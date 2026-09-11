"""Measure prefill and steady decode separately. --help needs only Python stdlib."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time


FIELDS = ["batch_size", "prompt_tokens", "output_tokens", "decode_steps",
          "prefill_ms_p50", "prefill_ms_p95", "decode_step_ms_p50", "decode_step_ms_p95",
          "per_user_tokens_s", "aggregate_output_tokens_s", "peak_allocated_gb", "status"]


def percentile(samples, fraction):
    """Linearly interpolate an empirical quantile; independent of numpy/PyTorch."""
    values = sorted(samples)
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Need nonempty samples and a fraction in [0, 1]")
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def summarize(batch, prompt, output, prefill_ms, decode_ms, peak):
    steps = output - 1
    step_ms = [duration / steps for duration in decode_ms]
    p50_step = statistics.median(step_ms)
    return dict(zip(FIELDS, [batch, prompt, output, steps,
                           statistics.median(prefill_ms), percentile(prefill_ms, 0.95),
                           p50_step, percentile(step_ms, 0.95),
                           1000 / p50_step, batch * 1000 / p50_step, peak / 1e9, "ok"]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run deterministic CPU FP32 correctness checks and exit")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=65, help="Includes first token produced by prefill; must be >=2")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=3, help="Full prefill+decode warmups per batch size")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--mlp-width", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-sequence", type=int, default=4096, help="Fixed learned-position capacity; hold fixed across context-length experiments")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    args = parser.parse_args()
    if min(args.batches) < 1 or args.prompt_tokens < 1 or args.output_tokens < 2:
        parser.error("batches/prompt must be positive; output-tokens must be >=2")
    if args.warmup < 1 or args.trials < 3:
        parser.error("Use >=1 warmup and >=3 measured trials (20 recommended)")
    if len(set(args.batches)) != len(args.batches):
        parser.error("Do not repeat batch sizes")
    if args.prompt_tokens + args.output_tokens - 1 > args.max_sequence:
        parser.error("prompt-tokens + output-tokens - 1 must fit max-sequence")
    return args


def main():
    args = parse_args()
    try:
        import torch
        from model import ModelConfig, TinyDecoder, check_correctness
    except ImportError as exc:
        raise SystemExit("Install PyTorch first; see README.md. --help works without PyTorch.") from exc
    if args.check:
        print(json.dumps(check_correctness(), indent=2))
        return
    if not torch.cuda.is_available():
        raise SystemExit("This timing experiment requires an NVIDIA CUDA GPU. CPU correctness: --check")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("This GPU does not support BF16. Choose --precision fp16, and report that precision.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # For an FP32 experiment, prevent TF32 from silently changing matrix precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    config = ModelConfig(layers=args.layers, width=args.width, heads=args.heads,
                         mlp_width=args.mlp_width, vocab_size=args.vocab_size,
                         max_sequence=args.max_sequence)
    model = TinyDecoder(config).to(device="cuda", dtype=dtype).eval()
    gpu = torch.cuda.get_device_properties(0)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "TinyDecoder, random weights; no language-quality interpretation",
        "config": asdict(config), "parameter_count": sum(p.numel() for p in model.parameters()),
        "seed": args.seed, "precision": args.precision, "warmup": args.warmup, "trials": args.trials,
        "prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens,
        "kv_cache_capacity": args.prompt_tokens + args.output_tokens - 1,
        "batches": args.batches, "python": platform.python_version(), "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": gpu.name,
        "gpu_total_memory_gb": gpu.total_memory / 1e9, "compute_capability": [gpu.major, gpu.minor],
        "sdpa_backend": "PyTorch automatic selection; no backend forced", "compile": False,
        "tf32": False, "timing": "perf_counter with CUDA synchronization before/after each phase",
        "includes": "Python dispatch, GPU work, KV writes, greedy argmax",
        "excludes": "model/cache allocation, tokenization, network, queueing, transfer to host",
        "percentiles": "Across repeated trials; each decode sample is a whole decode-loop time divided by its step count. NOT serving tail latency.",
        "memory": "PyTorch peak tensor allocation including model, preallocated full-capacity KV, inputs and temporaries; excludes driver/context and reserved unused pool",
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("model.py", "benchmark.py")},
        "raw_trials": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output.with_suffix(".json")
    rows = []

    def save():
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        json_path.write_text(json.dumps(metadata, indent=2) + "\n")

    @torch.inference_mode()
    def run_batch(batch):
        # Allocation is outside timing but is counted in peak tensor memory.
        prompt = torch.randint(config.vocab_size, (batch, args.prompt_tokens), device="cuda")
        caches = model.allocate_cache(batch, args.prompt_tokens + args.output_tokens - 1)

        def trial():
            torch.cuda.synchronize()
            start = time.perf_counter()
            next_token = model(prompt, caches).argmax(dim=-1)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - start) * 1000
            # Prefill logits predict output token 1. Feed that token to obtain
            # token 2, then continue; G output tokens require G-1 decode calls.
            start = time.perf_counter()
            for step in range(args.output_tokens - 1):
                pos = args.prompt_tokens + step
                next_token = model(next_token, caches, start_pos=pos).argmax(dim=-1)
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - start) * 1000
            return prefill_ms, decode_ms

        for _ in range(args.warmup):
            trial()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        prefill, decode = [], []
        for _ in range(args.trials):
            first, rest = trial()
            prefill.append(first)
            decode.append(rest)
        peak = torch.cuda.max_memory_allocated()
        metadata["raw_trials"][str(batch)] = {"prefill_ms": prefill, "decode_phase_ms": decode}
        return summarize(batch, args.prompt_tokens, args.output_tokens, prefill, decode, peak)

    print(f"{gpu.name} | {args.precision} | {metadata['parameter_count'] / 1e6:.1f}M random parameters")
    print("Batch  Prefill p50 ms  Decode avg-step p50 ms  Per-user tok/s  Total tok/s  Peak GB")
    for batch in args.batches:
        try:
            row = run_batch(batch)
        except torch.cuda.OutOfMemoryError:
            row = {field: "" for field in FIELDS}
            row.update(batch_size=batch, prompt_tokens=args.prompt_tokens,
                       output_tokens=args.output_tokens, decode_steps=args.output_tokens - 1, status="oom")
            print(f"{batch:5d}  OOM (recorded; no throughput fabricated)")
        else:
            print(f"{batch:5d}  {row['prefill_ms_p50']:14.2f}  {row['decode_step_ms_p50']:22.3f}"
                  f"  {row['per_user_tokens_s']:14.1f}  {row['aggregate_output_tokens_s']:11.1f}"
                  f"  {row['peak_allocated_gb']:7.3f}")
        rows.append(row)
        save()
        gc.collect()
        torch.cuda.empty_cache()
    print(f"Saved {args.output} and {json_path}")


if __name__ == "__main__":
    main()
