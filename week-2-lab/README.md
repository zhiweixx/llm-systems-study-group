# Week 2 lab: measure prefill, decode, and batching

**Question:** how does batch size change first-token compute time, per-user generation speed, total throughput, and memory use?

Run a small decoder on one NVIDIA GPU, keep its architecture, weights, precision and sequence lengths fixed, and change only batch size. The decoder has **random weights** and about 113 million parameters with the default settings. It tests execution behavior; it does not produce meaningful language or represent the throughput of a trained production LLM.

No GPU timings are supplied. The CSV and plots are produced from your own run.

## Start in Colab

[Open in Colab](https://colab.research.google.com/github/zhiweixx/llm-systems-study-group/blob/main/week-2-lab/week2-inference-lab.ipynb), then select **Runtime → Change runtime type → GPU**. You can also [download the notebook](week2-inference-lab.ipynb). The notebook contains the code; no model download is needed. It checks correctness, identifies the assigned GPU, chooses a supported precision, runs the sweep, and plots the results. Keep the recorded GPU and precision with every figure.

Alternatively, download these files into one directory:

- [model.py](model.py): causal decoder with a preallocated KV cache.
- [benchmark.py](benchmark.py): benchmark and CPU numerical check.
- [plot_results.py](plot_results.py): plots measured CSV data.

Use Python 3.10 or newer and a CUDA-enabled PyTorch installation. For a local GPU, select the installation matching your environment from [PyTorch's installation page](https://pytorch.org/get-started/locally/). Colab normally provides PyTorch already; the notebook prints its version. Matplotlib is needed for plotting.

```bash
python -m pip install matplotlib
python benchmark.py --help
python benchmark.py --check
python benchmark.py --batches 1 2 4 8 16 32 --precision bf16 --output results.csv
python plot_results.py results.csv
```

If the assigned GPU does not support BF16, use `--precision fp16` consistently for the entire sweep. Record that change. A short exploratory run can use `--trials 3 --warmup 1 --batches 1 4 16`; use the default 20 trials and 3 warmups when preparing your presentation. A 95th percentile from only three trials is not a reliable tail estimate.

## What is held fixed?

The defaults are 8 layers, hidden width 1024, 16 attention heads, MLP width 4096, vocabulary size 4096, prompt length 512, and 65 generated tokens. The model uses causal multi-head attention, learned position embeddings with a fixed capacity of 4096 positions, LayerNorm and a GELU MLP. Dropout is zero. No external checkpoint is loaded; the random initialization is fixed by seed 2026. The JSON records the model dimensions, parameter count, versions, GPU, precision and code hashes. Keep `--max-sequence` fixed when comparing prompt lengths so that the position table and initialized model weights stay identical.

The same model remains on the GPU throughout the sweep. All requests in a batch have the same prompt/output lengths. Random token IDs keep the experiment independent of tokenization and text datasets. The cache is allocated to the maximum context length before timing, and new K/V values are written into their slots; the decode path does not repeatedly concatenate the growing cache.

## Precisely what do we time?

The benchmark uses `time.perf_counter()` with `torch.cuda.synchronize()` around each phase. It includes Python dispatch, GPU execution, KV writes and greedy `argmax`. It excludes model/cache allocation, tokenization, network traffic, queueing, and copying results to the host. This is an eager PyTorch benchmark; CUDA Graphs and `torch.compile` are not enabled.

1. **Prefill:** process the prompt and choose the first output token from the final prompt position's logits.
2. **Decode:** feed that first output token back to obtain the second, then continue. Generating 65 tokens therefore needs **64 decode calls** after prefill.
3. **Repeat:** warm up each batch size, then repeat the complete fixed-length run. Synchronization occurs at phase boundaries, not after every decode token.

For each trial:

```text
average decode-step time = measured decode-loop time / 64
per-user output tokens/s = 1000 / median average-step time in ms
aggregate output tokens/s = batch size × per-user output tokens/s
```

The throughput metrics describe **steady decode**, excluding prefill. The p50/p95 columns summarize repeated trials of a phase or average-step duration; they are not individual-token p95 or a production serving SLO. The prefill measurement is first-token *compute* time, not network-visible time to first token.

The peak-memory column is PyTorch's peak **allocated tensor memory** in decimal GB, including weights, full-capacity KV caches, inputs and temporaries. It excludes CUDA context/driver memory and unused reserved allocator memory. An out-of-memory batch is recorded with `status=oom` and empty measurement fields, never as zero latency or throughput. [PyTorch memory metric](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html)

## Why the decode attention mask matters

During prefill, many query positions are present, so a causal mask prevents an early token from seeing later prompt tokens. During a one-token decode call, the exposed K/V slice contains only the past and current positions: the new query may attend to **every key in that slice**.

The decode path therefore uses `is_causal=False`. Passing `is_causal=True` to non-square SDPA with one query and many keys uses upper-left alignment and would incorrectly mask all but the first key. The CPU FP32 correctness check compares cached logits against full causal recomputation at several positions and also checks greedy generation. [PyTorch SDPA masking semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

## Read the results

`results.csv` has one row per batch size. `results.json` stores the environment and raw repeated timings. The plot script writes PNG, SVG and PDF figures.

| CSV column | Meaning |
|---|---|
| `batch_size`, `prompt_tokens`, `output_tokens`, `decode_steps` | Workload sizes; decode steps = output tokens − 1 |
| `prefill_ms_p50`, `prefill_ms_p95` | Repeated prefill + first-token-selection durations |
| `decode_step_ms_p50`, `decode_step_ms_p95` | Repeated average decode-step durations |
| `per_user_tokens_s` | Steady decode speed for one sequence |
| `aggregate_output_tokens_s` | Steady decode throughput across the batch |
| `peak_allocated_gb` | Peak allocated tensor memory in decimal GB |
| `status` | `ok` or `oom` |

Before running, predict the directions of change. Afterwards, explain the measurements:

- Does total throughput rise with batch size? Does per-user speed rise as well?
- Where do gains flatten? Which measurements would distinguish bandwidth, compute and launch overhead?
- Under a fixed per-user latency limit, which measured batch sizes are acceptable?
- Repeat the sweep with `--prompt-tokens 2048`. Which costs grow even though the model weights are unchanged?

Batching can improve weight reuse, but a larger batch is not guaranteed to improve every metric. Small models can be dominated by framework/launch overhead. Attention backend selection, GPU clocks, memory pressure and other GPU users can affect the results. A bottleneck diagnosis needs evidence; the curves alone do not prove a specific hardware limit.

**For the meeting:** run in advance, keep the CSV and JSON, and present the measured plot with its configuration. Use the live notebook only as an optional repeat.
