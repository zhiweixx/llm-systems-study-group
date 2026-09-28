# Week 4 lab: find the useful serving rate

**Question:** at what incoming request rate does the service stop meeting its latency targets, even if output tokens/s continues to increase?

This experiment sends requests to an **already running vLLM server**. Keep the model, GPU allocation, server configuration and prompt/output lengths fixed; vary the offered request rate. It complements Week 2's GPU experiment by including queueing and the serving stack.

No GPU measurements are supplied. The script prints commands by default and does not launch a server, change its configuration, or send requests unless you add `--run`.

## Prepare the client and server

Download [run_benchmark.py](run_benchmark.py). Printing commands requires only Python 3.10 or newer. Execution requires a compatible `vllm` installation on the benchmark client and a running server that exposes `/v1/completions`. The client can run separately from the GPU server. Use the server's actual model/tokenizer ID; if its API name is an alias, add `--served-model-name`.

The commands use the [official serving benchmark CLI](https://docs.vllm.ai/en/latest/cli/bench/serve/), checked on September 28, 2026. Because the CLI evolves, the wrapper checks that the installed help advertises the required flags and saves that help alongside the results. It also records the client version. Use the matching documentation for your installed release if preflight fails.

Start with a small exploratory sweep:

```bash
python run_benchmark.py \
  --model YOUR_MODEL_ID \
  --base-url http://YOUR_SERVER:8000 \
  --rates 0.5 1 2 \
  --num-prompts 128 \
  --input-len 512 --output-len 128
```

Replace `YOUR_MODEL_ID` and the server address. The output shows complete `vllm bench serve` commands. By default they request fixed random prompt/output lengths, seed 2026, and example targets of **TTFT ≤ 1,000 ms and TPOT ≤ 50 ms**. Change the targets with `--ttft-slo-ms` and `--tpot-slo-ms` to match your application. Requested lengths should fit the server's context limit; retain the benchmark's reported token counts to check what was actually processed.

Before execution, create a small text file named `server-record.txt` recording:

```text
Server vLLM version / container digest:
Model and tokenizer revision:
GPU models, count, interconnect, and competing workloads:
Complete server launch command and relevant environment settings:
Model dtype and KV-cache dtype:
Tensor/data/pipeline parallel configuration:
Prefix caching enabled or disabled:
Initial prefix-cache state and preparation procedure:
Scheduler token budget, sequence limit, and chunked-prefill settings:
Client machine / network location:
```

Then repeat the printed command with:

```bash
python run_benchmark.py \
  --model YOUR_MODEL_ID \
  --base-url http://YOUR_SERVER:8000 \
  --rates 0.5 1 2 \
  --num-prompts 128 \
  --input-len 512 --output-len 128 \
  --server-record server-record.txt \
  --run
```

The script saves a timestamped directory inside `week4-results/`, including raw benchmark JSON, per-rate logs, exact commands, versions and your server record. It stops on the first failed or timed-out command. Its default timeout is 900 seconds **per rate**; override `--timeout-seconds` for longer experiments. After interrupting a run, confirm that outstanding server work has drained before starting another measurement.

## What is controlled?

Use the same prompt distribution and seed across configurations. The random workload has no intentionally shared prefix. Output length is held fixed with `--ignore-eos` to avoid confusing a shorter answer with a faster service. This is a workload-control choice, not a natural-chat quality evaluation.

The requested rate specifies an arrival process; it is not the number of simultaneous requests. The default experiment has no explicit client concurrency cap. If you add `--max-concurrency`, a slow service can make requests wait at the client, so the server may receive them more slowly than the requested rate. Label this as a capped experiment and inspect client-side waiting as well. [vLLM rate and concurrency controls](https://docs.vllm.ai/en/latest/cli/bench/serve/#--max-concurrency)

For a first capacity baseline, use a server with prefix caching disabled and record that setting. Otherwise the repeated seeded workload across sweep points may leave reusable prefixes in the cache. Keep server warmup and prefix-cache preparation identical across comparisons. **`--num-warmups 0` does not clear the server's cache.** This wrapper uses that setting because warmup/cache preparation should be performed and recorded explicitly.

The 128-request default is an exploratory run. Increase the request count and repeat runs when estimating a tail percentile; p99 from 128 requests is largely determined by the slowest one or two observations. If a high-rate run spends most of its time draining a growing queue, extend the observation or lower the rate before claiming a sustainable operating point. A finite batch completion rate alone does not establish stability.

## Read the measurements

The benchmark reports selected percentiles and saves detailed per-request information. Keep the raw files: this wrapper intentionally does not assume a particular release's JSON schema or silently reinterpret missing fields. [vLLM benchmark outputs](https://docs.vllm.ai/en/latest/cli/bench/serve/#--save-detailed)

For every offered rate, record:

| Measurement | What you use it for |
|---|---|
| Completed requests/s and output tokens/s | Aggregate work actually completed |
| p50/p95/p99 TTFT | Time from sending the request to receiving the first token |
| p50/p95/p99 TPOT | Distribution of each request's average output-token interval |
| p50/p95/p99 ITL | Token-stream stalls; do not substitute the TPOT percentile |
| Request goodput | Requests meeting **both** chosen TTFT and TPOT targets per second |
| Failure/timeout counts and observed lengths | Detect incomplete or incomparable experiments |
| Server waiting/running requests and preemptions | Evidence explaining the latency change |

TTFT and token timing are client-visible measurements, so transport and client overhead can contribute. With a client concurrency cap, waiting for permission to send is recorded separately: the wrapper also requests `client_queue_time` and `e2el_including_client_queue` percentiles. Its TTFT-based goodput does not include that pre-send wait, so do not treat capped-run goodput as an unrestricted-arrival SLO result. [Benchmark timing implementation](https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py)

Server telemetry helps localize the cause. Metric names and histogram availability depend on the release; inspect the deployed server's `/metrics` endpoint and its [metrics documentation](https://docs.vllm.ai/en/latest/usage/metrics/).

Make two plots with **offered requests/s** on the x-axis: (1) output tokens/s and request goodput, on separate axes or panels because their units differ; (2) p95 TTFT and p95 TPOT, with their respective target lines. Retain per-rate errors rather than dropping failed runs. Explain why the maximum tokens/s point may be unsuitable for interactive users. State the observation interval and whether queueing remained bounded.

## A separate experiment for prefix caching

The random workload above tests load sensitivity. It does **not** establish the benefit of prefix caching.

To study caching, create a controlled request trace with exact repeated token prefixes and distinct suffixes. Compare the same arrival trace and server configuration under an explicit cache preparation protocol:

1. **Cold:** ensure the test prefixes are absent before measurement, using a documented reset/restart procedure or genuinely fresh prefixes. Account for runtime warmup separately.
2. **Warm:** populate the intended reusable prefixes before the measured requests, then keep request order and routing consistent.
3. Record reused-token counts, TTFT, queue time and occupied KV blocks. A higher hit fraction does not imply the same proportional improvement in latency or physical memory use.

In a multi-replica experiment, cache state belongs to each replica: record routing and each replica's load, not just a cluster-wide average. Prefix caching reuses prior prefill work, while generated tokens still require decoding and attention to their history. [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

## Bring one diagnosis to the meeting

Choose an observed latency change, then propose one controlled follow-up experiment that could disprove your explanation. For example: keep offered rate and lengths fixed, change only the scheduler token budget, and compare TTFT, ITL, waiting requests and goodput. Keep the total GPU budget unchanged when comparing a shared prefill/decode service against separate pools.
