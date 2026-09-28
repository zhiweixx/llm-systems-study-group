#!/usr/bin/env python3
"""Print or run a controlled vLLM serving request-rate sweep.

Dry runs use only Python's standard library and never contact the server.
Actual runs require an existing server and a compatible vLLM benchmark CLI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Tokenizer/model ID or local path")
    p.add_argument("--served-model-name", help="API model name, if different")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--rates", nargs="+", type=positive_float, default=[0.5, 1.0, 2.0])
    p.add_argument("--num-prompts", type=positive_int, default=128)
    p.add_argument("--input-len", type=positive_int, default=512)
    p.add_argument("--output-len", type=positive_int, default=128)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-concurrency", type=positive_int, help="Optional client cap; changes the arrival experiment")
    p.add_argument("--ttft-slo-ms", type=positive_float, default=1000)
    p.add_argument("--tpot-slo-ms", type=positive_float, default=50)
    p.add_argument("--vllm-bin", default="vllm", help="vLLM executable in the benchmark client environment")
    p.add_argument("--output-dir", type=Path, default=Path("week4-results"))
    p.add_argument("--server-record", type=Path, help="Required with --run: text file recording server version, config, hardware and cache condition")
    p.add_argument("--timeout-seconds", type=positive_int, default=900, help="Per-rate command timeout; incomplete runs are not valid measurements")
    p.add_argument("--run", action="store_true", help="Execute printed commands against the existing server")
    return p


def command_for(args: argparse.Namespace, rate: float, index: int, output_dir: Path) -> list[str]:
    percentile_metrics = "ttft,tpot,itl,e2el"
    if args.max_concurrency is not None:
        percentile_metrics += ",client_queue_time,e2el_including_client_queue"
    command = [
        args.vllm_bin, "bench", "serve",
        "--backend", "openai", "--endpoint", "/v1/completions",
        "--base-url", args.base_url, "--model", args.model,
        "--dataset-name", "random", "--num-prompts", str(args.num_prompts),
        "--input-len", str(args.input_len), "--output-len", str(args.output_len),
        "--random-range-ratio", "0.0", "--random-prefix-len", "0",
        "--seed", str(args.seed), "--request-rate", str(rate), "--burstiness", "1",
        "--ignore-eos", "--num-warmups", "0",
        "--percentile-metrics", percentile_metrics, "--metric-percentiles", "50,95,99",
        "--goodput", f"ttft:{args.ttft_slo_ms}", f"tpot:{args.tpot_slo_ms}",
        "--save-result", "--save-detailed", "--result-dir", str(output_dir),
        "--result-filename", f"rate-{index:02d}-{rate:g}.json",
    ]
    if args.served_model_name:
        command += ["--served-model-name", args.served_model_name]
    if args.max_concurrency is not None:
        command += ["--max-concurrency", str(args.max_concurrency)]
    return command


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.output_len < 2:
        p.error("--output-len must be at least 2 to measure TPOT")
    if args.run and (args.server_record is None or not args.server_record.is_file()):
        p.error("--run requires an existing --server-record text file; see README.md")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir = args.output_dir.resolve() / stamp
    commands = [command_for(args, rate, i, output_dir) for i, rate in enumerate(args.rates, 1)]
    print("Existing server:", args.base_url)
    print("Mode:", "EXECUTE" if args.run else "PRINT ONLY; no requests or files created")
    print("SLOs are example targets, not measured hardware capabilities.")
    if args.max_concurrency:
        print("Client concurrency is capped: the server may receive requests more slowly than the offered rate.")
    for rate, command in zip(args.rates, commands):
        print(f"\n# {rate:g} requests/s; expected arrival-window scale ~{args.num_prompts / rate:.0f} s, plus drain time")
        print(shlex.join(command))
    if not args.run:
        return 0

    if shutil.which(args.vllm_bin) is None:
        p.error(f"cannot find vLLM executable: {args.vllm_bin}")
    help_result = subprocess.run([args.vllm_bin, "bench", "serve", "--help"], capture_output=True, text=True, timeout=60)
    if help_result.returncode:
        sys.stderr.write(help_result.stdout + help_result.stderr)
        p.error("vllm bench serve --help failed in the benchmark client environment")
    help_text = help_result.stdout + help_result.stderr
    needed_flags = {arg for command in commands for arg in command if arg.startswith("--")}
    missing_flags = sorted(flag for flag in needed_flags if flag not in help_text)
    if missing_flags:
        p.error("installed CLI does not advertise required flags: " + ", ".join(missing_flags))
    version = subprocess.run([args.vllm_bin, "--version"], capture_output=True, text=True, timeout=60)
    if version.returncode:
        p.error("could not record client vLLM version with vllm --version")

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "server-record.txt").write_text(args.server_record.read_text())
    (output_dir / "client-help.txt").write_text(help_text)
    manifest = {
        "started_utc": stamp,
        "python": sys.version,
        "client_vllm_version": (version.stdout + version.stderr).strip(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "commands": commands,
        "runs": [],
    }
    manifest_path = output_dir / "manifest.json"

    def save_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    for index, (rate, command) in enumerate(zip(args.rates, commands), 1):
        log_path = output_dir / f"rate-{index:02d}-{rate:g}.log"
        print(f"\nRunning {rate:g} requests/s; benchmark output: {log_path}", flush=True)
        run = {"rate": rate, "log": log_path.name, "status": "running"}
        manifest["runs"].append(run)
        save_manifest()
        try:
            with log_path.open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_seconds)
            run.update(status="completed" if result.returncode == 0 else "failed", returncode=result.returncode)
            save_manifest()
            print(log_path.read_text(), end="")
            if result.returncode:
                return result.returncode
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            run.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "timeout")
            save_manifest()
            print(f"\nStopped; inspect {log_path}. Do not use this incomplete run as a throughput result.", file=sys.stderr)
            return 130 if isinstance(exc, KeyboardInterrupt) else 124
    print(f"\nFinished. Keep the raw vLLM JSON, logs and manifest together: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
