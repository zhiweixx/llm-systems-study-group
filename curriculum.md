# Four-week curriculum

For readers who know Transformer models and PyTorch but have little GPU systems background. The course focuses on GPUs, LLM inference, and model serving, with training memory and sharding as supporting topics.

Each meeting lasts 50 minutes. Week 1 uses a 22-slide, 30-minute presentation, followed by 15 minutes of discussion and 5 minutes of buffer. Later meetings reserve at least 15 minutes for discussion. Weeks 2–4 are planned; their slides are not published yet.

## Week 1: GPU memory and performance

**Meeting date: September 10, 2026.**

- Explain CPU dispatch, asynchronous launches, kernels, blocks, and SMs.
- Locate registers, shared memory, L1/L2 caches, and HBM on an H100, then use these locations to explain data movement.
- Explain compute and bandwidth limits using warehouse-and-factory illustrations.
- Show how lower precision, operator fusion, memory coalescing, and tiling can reduce transfer costs. Use thread/address and matrix-tile diagrams to make reuse concrete.
- Estimate weights, one explicit mixed-precision Adam layout, and inference KV payload using decimal GB and a supplied model-specific KV cost per token.
- Use execution timelines to identify launch overhead and validate an optimization.
- Calculate MFU and distinguish it from active GPU time.

**Worked questions:** compare 4,000- and 8,000-token contexts in a 24 GB inference budget; estimate 8B training MFU on eight H100 GPUs. Each question has an immediate solution slide. Week 1 retains basic within-request KV caching and its memory budget; cross-request prefix reuse and hit-rate metrics belong to Week 4.

**Discussion:** change context length, precision, concurrency, or measured token throughput and identify which quantities change.

## Week 2: Single-GPU inference performance

Build on the performance vocabulary introduced in Week 1. Focus on the behavior of prefill and decode, weight/KV traffic, batching, and actual measurements.

- Compare arithmetic intensity and latency across prefill and decode workloads.
- Estimate compute time and memory-transfer time under explicit assumptions.
- Build on Week 1's coalescing and tiling examples to explain how fusion, CUDA Graphs, and FlashAttention address different costs.
- Interpret CPU/GPU timelines and design a batch-size sweep.

**Discussion:** aggregate tokens/s increases with batch size, but each user's time between tokens becomes longer. Explain why both can happen and what to measure next.

**Reading:** [Scaling Book: rooflines](https://jax-ml.github.io/scaling-book/roofline/), [Scaling Book: inference](https://jax-ml.github.io/scaling-book/inference/), and [Horace He: performance from first principles](https://horace.io/brrr_intro.html).

## Week 3: Multi-GPU parallelism and sharding

Understand how extra GPUs change memory capacity, computation, and communication.

- Read diagrams of AllReduce, AllGather, and ReduceScatter.
- Distinguish data parallelism, serving replicas, tensor parallelism, and pipeline parallelism.
- Estimate persistent training state under ZeRO/FSDP-style sharding, with explicit exclusions for activations and transient buffers.
- Account for interconnect costs and pipeline bubbles when comparing schemes.

**Discussion:** an 8B inference model fits on one GPU. With four GPUs, compare four replicas, two groups of TP=2, and TP=4 under capacity, throughput, and latency objectives.

**Reading:** [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html) sections on DP, ZeRO, TP, and PP; [Stanford CS336 lectures](https://github.com/stanford-cs336/lectures).

## Week 4: LLM serving, scheduling, and KV management

Connect an inference engine to an online workload and explicit service objectives.

- Define TTFT, TPOT/ITL, throughput, tail latency, and SLOs.
- Compare static batching, continuous batching, and chunked prefill.
- Explain paged KV allocation, physical prefix sharing, and admission control.
- Distinguish token hit rate, request hit rate, and KV-pool occupancy, including cold/warm cache state and eviction.
- Design a load test varying request rate and prompt/output lengths.

**Worked exercise:** shared-prefix hits and physical KV storage, moved from Week 1. The [prefix-cache teaching notes](https://zhiweixx.github.io/llm-systems-study-group/week-4/prefix-cache.html) preserve the explanation, assumptions, question, and solution for this week.

**Discussion:** design an 8B chat service with four GPUs, mixed prompt lengths, and repeated system prompts. Specify model placement, KV budget, scheduling, and experiments that test the proposed latency targets.

**Reading:** [vLLM optimization](https://docs.vllm.ai/en/stable/configuration/optimization/), [prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/), and [Scaling Book: inference](https://jax-ml.github.io/scaling-book/inference/).

## Shared teaching model

Week 1 uses a hypothetical dense model with 8 billion parameters and BF16 weights. The KV-memory cost is supplied as approximately 0.131 MB per processed token across the whole model, so the audience can reason about context and concurrency without deriving attention-head layouts. The calculation reference uses 131,072 bytes per token. The Week 4 notes give the fuller architecture assumptions for the separate prefix-sharing exercise. These are teaching examples, not an exact checkpoint specification or measured benchmark.
