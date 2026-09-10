# Four-week curriculum

For readers who know Transformer models and PyTorch but have little GPU systems background. The course focuses on GPUs, LLM inference, and model serving, with training memory and sharding as supporting topics.

Each meeting lasts 50 minutes. Week 1 uses the current 19-slide, 30-minute presentation, followed by 15 minutes of discussion and 5 minutes of buffer. Later meetings reserve at least 15 minutes for discussion. Weeks 2–4 are planned; their slides are not published yet.

## Week 1: GPU memory and performance

**Meeting date: September 10, 2026.**

- Explain CPU dispatch, asynchronous launches, kernels, blocks, and SMs.
- Locate registers, shared memory, L1/L2 caches, and HBM on an H100.
- Estimate weights, explicit mixed-precision Adam state, and inference KV payload.
- Explain compute and bandwidth limits using warehouse-and-factory illustrations, fusion, and data reuse; use execution timelines to explain launch overhead.
- Calculate MFU and distinguish token hit rate, request hit rate, and cache occupancy.

**Worked questions:** inference in 24 GiB; 8B training MFU on eight H100 GPUs; shared-prefix hits and physical KV storage. Each question has an immediate solution slide.

**Discussion:** change one assumption—context length, precision, concurrency, or prefix warmth—and identify which quantities change.

## Week 2: Single-GPU inference performance

Build on the performance vocabulary introduced in Week 1. Focus on the behavior of prefill and decode, weight/KV traffic, batching, and actual measurements.

- Compare arithmetic intensity and latency across prefill and decode workloads.
- Estimate compute time and memory-transfer time under explicit assumptions.
- Explain how fusion, CUDA Graphs, and FlashAttention address different costs.
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
- Design a load test varying request rate and prompt/output lengths.

**Discussion:** design an 8B chat service with four GPUs, mixed prompt lengths, and repeated system prompts. Specify model placement, KV budget, scheduling, and experiments that test the proposed latency targets.

**Reading:** [vLLM optimization](https://docs.vllm.ai/en/stable/configuration/optimization/), [prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/), and [Scaling Book: inference](https://jax-ml.github.io/scaling-book/inference/).

## Shared teaching model

The numerical examples use a hypothetical dense model with 8 billion parameters, 32 layers, 32 query heads, 8 KV heads, and a head dimension of 128. BF16 weights and BF16 KV are the baseline. This is an explicitly specified teaching model, not an exact checkpoint specification or measured benchmark.
