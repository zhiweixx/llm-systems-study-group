# Four-week curriculum

For readers who know Transformer models and PyTorch but have little GPU systems background. The course focuses on GPUs, LLM inference, and model serving, with training memory and sharding as supporting topics.

The original meeting format is 50 minutes. Week 1 uses a 24-slide, 30-minute presentation, reserving 15 minutes for discussion and 5 minutes of buffer. Week 2 offers 31 slides with about 35 minutes of suggested full content or an approximately 32-minute route. The shorter route preserves 15 minutes of discussion and roughly 3 minutes of buffer. These are planning estimates rather than rehearsed durations. Week 3 is an expanded 45-slide teaching and reference deck with no fixed presentation duration; its sections can be presented separately. Week 4 remains planned.

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

## Week 2: LLM inference performance

**Meeting date: September 17, 2026.**

Build on the performance vocabulary introduced in Week 1. Explain why prefill is often compute-bound and small-batch decode is often bandwidth-bound, then connect those bottlenecks to optimizations and measurements. The core focuses on one GPU, with a preview of prefill–decode disaggregation.

- Map a linear layer to a matrix multiplication with M token rows: M = B × S during prefill and M = B for one decode step, where B is batch size and S is prompt length.
- Derive arithmetic intensity from matrix-multiplication FLOPs and bytes. Show how processing more token rows reuses the same weights and changes the likely bottleneck.
- Compare worked H100 compute-time and memory-transfer lower bounds under explicit assumptions; distinguish ideal bounds from measured runtime.
- Distinguish newly processed token rows from memory traffic. A decode step still reads weights and cached history, even though only one new token per request passes through the dense layers.
- Diagnose why aggregate decode throughput might plateau as batch size grows, distinguishing KV bandwidth from compute and host-work limits.
- Separate weight traffic from request-specific KV traffic: batching improves weight reuse, while longer contexts increase the KV data attention must read.
- Distinguish time to first token, inter-token latency, and aggregate throughput. Calculate the KV capacity needed by a set of active requests and show how continuous batching reuses a finished request's slot.
- Explain how PagedAttention maps logical KV blocks to physical GPU blocks, allocates blocks as a request grows, and reuses released capacity. Distinguish reduced allocation waste from the historical KV reads still required by attention.
- Derive online softmax from attention's weighted average: keep the maximum, denominator, and weighted-value sum; rescale when the maximum rises. Use FlashAttention to connect this recurrence to tiled GPU execution. The derivation remains on the slides.
- Diagnose a numerically correct paged-cache prototype that concatenates its K/V pages. Separate persistent allocation from temporary copies and explain why the attention kernel must consume the block table.
- Read the published interference experiment in DistServe Figure 2, then compare chunked prefill with separate prefill and decode GPU pools. Explain KV transfer and workload tradeoffs without promising an automatic throughput gain.
- Interpret CPU/GPU timelines and design a fixed-workload batch-size sweep measuring decode-step latency, aggregate output-token throughput, and peak memory.

**Presentation routes:** the full 31-slide deck has about 35 minutes of suggested content. An approximately 32-minute route skips the H100 numerical example (8) and optional experiment section (27–28). It retains arithmetic intensity, the complete online-softmax derivation (17–19), and the mixed-batch setup (23). The presenter chooses the route. The final two slides assign a take-home GEMM exercise and link to its solution; they add no planned lecture time.

**Discussion:** a paged KV implementation can produce correct outputs while still copying the entire history. Identify those copies, explain what a page-aware kernel changes, and distinguish fitting more requests from making each request faster.

**Take-home:** implement a tiled FP16 GEMM in Triton with FP32 accumulation, safe boundaries, correctness checks, and steady-state GPU timing. The final solution slide links to the [official Triton Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html). Grouped program ordering and autotuning are optional extensions.

**Reading:** [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10) for inference, [CS336 Lecture 5, pp. 52–54](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=52) for online softmax, [Berkeley Spring 2026 Lecture 18](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf) for serving, [PagedAttention](https://arxiv.org/abs/2309.06180), and [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin).

## Week 3: Multi-GPU parallelism and sharding

Understand how extra GPUs change memory capacity, computation, and communication. The 45-slide deck covers all five parallelism dimensions through worked tensor examples, with no fixed presentation duration.

- Establish separate capacity, latency, and throughput objectives. Define nodes, ranks, and communication groups, and show that each GPU has its own memory.
- Route different inference requests to serving replicas. Distinguish independent dense-model inference from synchronized data-parallel training.
- Partition a two-layer MLP across two GPUs: column-partition the first matrix, keep intermediate features local, row-partition the second matrix, and sum the partial output. Explain why the position of GELU matters. Extend the partition to attention heads and KV storage.
- Interpret AllReduce through a numerical example. Compare communication startup latency with transfer time, and follow exposed communication on the critical path rather than assuming linear speedup.
- Partition model layers into pipeline stages. Show how independent microbatches overlap, explain fill/drain bubbles, and retain the autoregressive token dependency for a single decode request. Distinguish pipeline parallelism from Week 2's prefill–decode disaggregation.
- Follow DDP's different local minibatches through gradient averaging and matching optimizer updates. Shard optimizer state, gradients, and parameters progressively with ZeRO, then trace FSDP's parameter gathering and gradient reduction.
- Calculate an explicit 8B mixed-precision Adam state budget across four GPUs: 128, 56, 44, and 32 GB per GPU for DDP and ZeRO stages 1–3. Separate persistent state from activation and temporary peak memory.
- Split positions of the same long sequence with context parallelism. Keep local queries while circulating remote K/V, merge stable attention summaries, and explain causal workload imbalance. Compare ring attention with Ulysses' sequence-to-head exchange and distinguish both from Megatron sequence parallelism.
- Extend the context partition to decode history: one new query can require attention across KV shards. State what is saved, what remains replicated, and what communication is still needed.
- Introduce an MoE expert as a learned MLP, then trace four tokens through top-2 routing, expert ownership, dispatch, local expert computation, and weighted output combination. Explain expert-load imbalance and how DP attention can coexist with EP experts.
- Combine explicit replica, TP, and PP groups. Compare different layouts under the same workload and GPU budget, with attention to per-GPU memory and latency targets.

**Worked questions:** complete a tensor-parallel MLP and identify its necessary communication; diagnose a hypothetical decode profile in which increasing TP from 2 to 4 barely improves latency. Each question has an immediate solution slide. These are authored interview-style exercises.

**Take-home:** simulate two tensor-parallel MLP ranks using ordinary PyTorch, test multiple shapes, and handle biases correctly. The runnable companion checks equivalence to the unpartitioned MLP on a CPU or one GPU. A real two-GPU AllReduce implementation is an optional extension; simulation timings are not distributed performance measurements.

**Discussion:** an 8B inference model fits on one GPU. With four GPUs, compare four replicas, two groups of TP=2, and TP=4 under capacity, throughput, and latency objectives. Then identify what would change for long contexts or a mixture-of-experts model.

**Reading:**

- [CS336 Lecture 8](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf), [Berkeley Lecture 4](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture4.pdf), and the [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html) for distributed execution and parallelism.
- [Megatron-LM](https://arxiv.org/abs/1909.08053) and the [PyTorch tensor-parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) for the paired MLP partition.
- [PyTorch DDP](https://docs.pytorch.org/docs/stable/notes/ddp.html), [ZeRO](https://arxiv.org/abs/1910.02054), and [PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) for replicated and sharded training state.
- [Insu Jang's context-parallelism overview](https://insujang.github.io/2024-09-20/introducing-context-parallelism/), [Ring Attention](https://arxiv.org/abs/2310.01889), and [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509) for distributed attention. The slides distinguish each algorithm's partition and communication.
- [Mixtral](https://arxiv.org/abs/2401.04088) and [vLLM expert-parallel deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) for expert routing and inference placement.

## Week 4: LLM serving, scheduling, and KV management

Connect an inference engine to an online workload and explicit service objectives.

- Define TTFT, TPOT/ITL, throughput, tail latency, and SLOs.
- Compare static batching, continuous batching, and chunked prefill.
- Extend Week 2's paged KV allocation example to physical prefix sharing and admission control.
- Develop Week 2's prefill–decode disaggregation preview with pool sizing, scheduling, KV-transfer costs, and network-aware placement under latency objectives.
- Distinguish token hit rate, request hit rate, and KV-pool occupancy, including cold/warm cache state and eviction.
- Design a load test varying request rate and prompt/output lengths.

**Worked exercise:** shared-prefix hits and physical KV storage, moved from Week 1. The [prefix-cache teaching notes](https://zhiweixx.github.io/llm-systems-study-group/week-4/prefix-cache.html) preserve the explanation, assumptions, question, and solution for this week.

**Discussion:** design an 8B chat service with four GPUs, mixed prompt lengths, and repeated system prompts. Specify model placement, KV budget, scheduling, and experiments that test the proposed latency targets.

**Reading:** [vLLM optimization](https://docs.vllm.ai/en/stable/configuration/optimization/), [prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/), and [Scaling Book: inference](https://jax-ml.github.io/scaling-book/inference/).

## Shared teaching model

Week 1 uses a hypothetical dense model with 8 billion parameters and BF16 weights. The KV-memory cost is supplied as approximately 0.131 MB per processed token across the whole model, so the audience can reason about context and concurrency without deriving attention-head layouts. The calculation reference uses 131,072 bytes per token. The Week 4 notes give the fuller architecture assumptions for the separate prefix-sharing exercise. These are teaching examples, not an exact checkpoint specification or measured benchmark.
