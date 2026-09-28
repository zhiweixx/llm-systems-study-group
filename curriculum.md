# Four-week curriculum

For readers who know Transformer models and PyTorch but have little GPU systems background. The course focuses on GPUs, LLM inference, and model serving, with training memory and sharding as supporting topics.

The original meeting format is 50 minutes. Week 1 uses a 24-slide, 30-minute presentation, reserving 15 minutes for discussion and 5 minutes of buffer. Week 2 offers 31 slides with about 35 minutes of suggested full content or an approximately 32-minute route. The shorter route preserves 15 minutes of discussion and roughly 3 minutes of buffer. These are planning estimates rather than rehearsed durations. Week 3 is an expanded 50-slide teaching and reference deck with no fixed presentation duration; its sections can be presented separately. Week 4 is a 34-slide teaching deck that opens with six visual slides and four theory slides on speculative decoding. Its full content needs roughly 55–60 minutes plus discussion; choose sections if retaining the original 50-minute meeting format.

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

Understand how extra GPUs change memory capacity, computation, and communication. The 50-slide deck covers all five parallelism dimensions through worked tensor examples, with no fixed presentation duration.

A separate [21-slide visual overview](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html) follows the same topics with less text: replicas and shards; DP and ZeRO/FSDP; TP and PP; CP and Megatron sequence parallelism; EP; combined groups and placement. Use the original 50-slide deck for the detailed derivations and exercises.

- Establish separate capacity, latency, and throughput objectives. Define nodes, ranks, and communication groups, and show that each GPU has its own memory.
- Route different inference requests to serving replicas. Distinguish independent dense-model inference from synchronized data-parallel training.
- Partition a two-layer MLP across two GPUs: column-partition the first matrix, keep intermediate features local, row-partition the second matrix, and sum the partial output. Explain why the position of GELU matters. Extend the partition to attention heads and KV storage.
- Interpret AllReduce through a numerical example. Compare communication startup latency with transfer time, and follow exposed communication on the critical path rather than assuming linear speedup.
- Partition model layers into pipeline stages. Show how independent microbatches overlap, explain fill/drain bubbles, and retain the autoregressive token dependency for a single decode request. Relate microbatch count to microbatch size under a fixed batch, including utilization and scheduling tradeoffs in pipeline frameworks. Distinguish pipeline parallelism from Week 2's prefill–decode disaggregation.
- Follow DDP's different local minibatches through gradient averaging and matching optimizer updates. Shard optimizer state, gradients, and parameters progressively with ZeRO, then trace FSDP's parameter gathering and gradient reduction.
- Calculate an explicit 8B mixed-precision Adam state budget across four GPUs: 128, 56, 44, and 32 GB per GPU for DDP and ZeRO stages 1–3. Separate persistent state from activation and temporary peak memory.
- Split positions of the same long sequence with context parallelism. Keep local queries while circulating remote K/V, merge stable attention summaries, and explain causal workload imbalance. Compare ring attention with Ulysses' sequence-to-head exchange and distinguish both from Megatron sequence parallelism.
- Extend the context partition to decode history: one new query can require attention across KV shards. State what is saved, what remains replicated, and what communication is still needed. Distinguish mathematical partitioning from TP/CP degree constraints imposed by head counts, layouts, and runtime implementations.
- Introduce an MoE expert as a learned MLP, then trace four tokens through top-2 routing, expert ownership, dispatch, local expert computation, and weighted output combination. Explain expert-load imbalance and how DP attention can coexist with EP experts.
- Combine explicit replica, TP, and PP groups. Compare different layouts under the same workload and GPU budget, with attention to per-GPU memory and latency targets. Place communication groups using the real network topology, and explain the costs and possible benefits of context and expert parallelism across nodes.

**Worked questions:** complete a tensor-parallel MLP and identify its necessary communication; diagnose a hypothetical decode profile in which increasing TP from 2 to 4 barely improves latency. Each question has an immediate solution slide. These are authored interview-style exercises.

**Take-home:** simulate two tensor-parallel MLP ranks using ordinary PyTorch, test multiple shapes, and handle biases correctly. The runnable companion checks equivalence to the unpartitioned MLP on a CPU or one GPU. A real two-GPU AllReduce implementation is an optional extension; simulation timings are not distributed performance measurements.

**Discussion:** an 8B inference model fits on one GPU. With four GPUs, compare four replicas, two groups of TP=2, and TP=4 under capacity, throughput, and latency objectives. Then identify what would change for long contexts or a mixture-of-experts model.

**Reading:**

- [CS336 Lecture 8](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf), [Berkeley Lecture 4](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture4.pdf), and the [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html) for distributed execution and parallelism.
- [Megatron-LM](https://arxiv.org/abs/1909.08053) and the [PyTorch tensor-parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) for the paired MLP partition.
- [PyTorch DDP](https://docs.pytorch.org/docs/stable/notes/ddp.html), [ZeRO](https://arxiv.org/abs/1910.02054), and [PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) for replicated and sharded training state.
- [Insu Jang's context-parallelism overview](https://insujang.github.io/2024-09-20/introducing-context-parallelism/), [Ring Attention](https://arxiv.org/abs/2310.01889), and [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509) for distributed attention. The slides distinguish each algorithm's partition and communication.
- [Mixtral](https://arxiv.org/abs/2401.04088) and [vLLM expert-parallel deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) for expert routing and inference placement.

## Week 4: Speculative decoding and LLM serving

Begin with how speculative decoding can emit several tokens per target-model pass. Follow a concrete draft through parallel verification, rejection, KV rollback, and a timing comparison. Derive why the sampler preserves the target distribution, how draft–target agreement affects accepted length, and when the resulting work saves time. Then follow one hypothetical 8B chat service on four GPUs. Assume one full replica fits on each GPU; apply the previous weeks’ inference and parallelism concepts to a continuously arriving workload.

| Slides | Content |
| --- | --- |
| 1 | Opening and teaching sequence |
| 2–7 | Speculative decoding: motivation, parallel target verification, greedy acceptance, rejected suffix and KV rollback, sampling correction, and the speedup tradeoff |
| 8–11 | Speculative-decoding theory: exact sampling proof, acceptance as distribution overlap and total variation distance, expected tokens per round, expected speedup and draft-length tradeoff |
| 12–13 | Four-GPU serving case, request lifecycle, router, engine scheduler, KV manager, latency targets |
| 14–20 | Exact prefix reuse, shared blocks, KV lifetime, hit metrics, Question 1 and solution |
| 21–28 | Arrival queues, three resource budgets, chunked scheduling, admission, routing, fixed-budget PD, Question 2 and solution |
| 29–33 | Controlled load testing, offered rate versus concurrency, latency attainment, diagnostic experiments, lab |
| 34 | Discussion: defend a serving decision and a falsifiable experiment |

Plan approximately 12–15 minutes for the speculative-decoding visual introduction, 8–10 minutes for its theory, and 55–60 minutes for the complete teaching sequence, followed by 15 minutes of discussion. These are planning estimates rather than rehearsed durations. For a 50-minute meeting, select a shorter route or continue the remaining sections in another session.

**Speculative-decoding learning goals:** explain why draft tokens can be verified together even though ordinary generation is sequential; trace accepted tokens, the first correction, and the discarded suffix; distinguish greedy equality from sampling-distribution preservation; derive the accepted and corrected probability mass; relate acceptance to distribution overlap; and estimate emitted tokens and speedup under stated assumptions about acceptance and timing.

**Question 1:** prefix hits increase after a routing change, but first-token tail latency worsens. Use per-replica queue and prefill observations to explain the outcome and propose a controlled routing experiment.

**Question 2:** short chat streams stall when long prompts arrive, while isolated decode kernels are unchanged. Use iteration timelines to propose a scheduling intervention, then compare shared and PD pools with the same four GPUs.

**Take-home:** use the [serving lab](https://zhiweixx.github.io/llm-systems-study-group/week-4/lab.html) to sweep offered request rate and report tail latency, failures, and requests meeting explicit targets. A fixed-length random workload is the baseline; repeated-prefix traces and cold/warm cache controls are a separate extension. No GPU benchmark measurements are fabricated for the teaching deck.

**Reading:** [vLLM prefix-cache design](https://docs.vllm.ai/en/stable/design/prefix_caching/), [scheduling configuration](https://docs.vllm.ai/en/latest/configuration/optimization/), [benchmark CLI](https://docs.vllm.ai/en/latest/cli/bench/serve/), [DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf), [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf), and [speculative decoding](https://proceedings.mlr.press/v202/leviathan23a.html). Slide-level notes identify supporting sources. The earlier [prefix-cache notes](https://zhiweixx.github.io/llm-systems-study-group/week-4/prefix-cache.html) remain an additional calculation reference.

## Shared teaching model

Week 1 uses a hypothetical dense model with 8 billion parameters and BF16 weights. The KV-memory cost is supplied as approximately 0.131 MB per processed token across the whole model, so the audience can reason about context and concurrency without deriving attention-head layouts. The calculation reference uses 131,072 bytes per token. The Week 4 notes give the fuller architecture assumptions for the separate prefix-sharing exercise. These are teaching examples, not an exact checkpoint specification or measured benchmark.
