# Week 1: 30-minute speaker notes

19 slides. Questions 1–3 each have a separate solution slide immediately afterward.

## 1. GPU memory and performance for LLMs

0.5 minutes

Use the first 30 minutes for the material and short numerical questions. Each question has its solution on the next slide. Use the remaining meeting time to revisit changed assumptions. The numerical model is an explicitly specified dense 8B GQA teaching model.



## 2. CPU execution, dispatch and asynchronous launches

2 minutes

x already resides on the GPU. CPU code selects a backend/implementation and submits arguments and tensor addresses. No whole-tensor CPU copy is implied. Dtype-specific selection may occur within the backend. A kernel is a GPU function executed by many threads. One PyTorch operation can launch zero, one or multiple kernels. The timeline illustrates allowed overlap, not measured duration. Same-stream operations stay ordered. CUDA events or synchronization are needed for GPU timing.

- [PyTorch dispatcher](https://docs.pytorch.org/tutorials/advanced/dispatcher.html)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)

## 3. GPU execution and memory model

3 minutes

Read the left panel as the logical programming model and the right panel as the hardware that executes it. A kernel launch creates a grid of blocks. Threads within a block cooperate using shared memory; warps group 32 threads. SM means Streaming Multiprocessor. A block executes on one SM, while an SM can host several resident blocks if registers, shared memory and other limits allow. The scheduling arrow illustrates the mapping, not a permanent one-to-one pairing. Each SM contains schedulers and compute units, a physical register file and a combined physical L1/shared-memory resource. Registers hold logically private thread values. L1 is hardware-managed caching; shared memory is a kernel-managed workspace. Device-wide global tensors normally reside in HBM, with L2 shared across SMs. HBM is stacked DRAM beside the compute die in the GPU package. Global memory is not exclusively owned by one grid and may survive kernel completion. CUDA local memory, omitted here, is thread-private device-memory-backed storage, for example register spills; it is not another name for on-chip SRAM. Cluster-level distributed shared memory is outside this introductory model. Native schematic adapted from CS336 Lecture 5, pages 10-12, and the NVIDIA CUDA programming guide.

- [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [CS336 Lecture 5, pp. 10–12](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=12)

## 4. H100 SXM 80 GB memory hierarchy

1.5 minutes

This conceptual pyramid shows proximity and scope, not capacity-scaled areas or a compulsory access path. Registers and combined L1/shared both have about 256 KB per SM, so do not infer strict capacity growth between every adjacent tier. NVIDIA specifies 64K 32-bit registers per SM, giving 256 KiB. Combined L1/texture/shared capacity is 256 KB per SM, including up to 228 KB shared memory. L2 is 50 MB device-wide. HBM3 is 80 GB and 3.35 TB/s peak for H100 SXM. Vendor KB/MB/GB labels are retained. HBM is stacked DRAM beside the compute die in the package. CPU main memory is a separate DRAM pool. SRAM is a technology used for L1, L2 and shared memory, not an extra hierarchy tier. Both SRAM and DRAM are volatile.

- [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)
- [Micron HBM](https://www.micron.com/products/memory/hbm)

## 5. Three performance limits: compute, memory and overhead

2.5 minutes

Conceptual framing follows Horace He: identify whether arithmetic, data movement or host/launch overhead limits execution before selecting an optimization. The equations and the insufficient-parallelism caveat are corroborated by NVIDIA GPU Performance Background, section 4. The figure is an analytic roofline, not measured data. It uses H100 SXM 80 GB peak HBM bandwidth of 3.35 TB/s and dense BF16 Tensor Core peak of 989 TFLOP/s. Intensity counts FLOPs per physical HBM byte under the declared traffic model. The ridge is 989 / 3.35 = 295.2238806 FLOP/byte. This Tensor Core ceiling is applicable only to eligible matrix computation; a pointwise scalar operator has a different instruction-throughput ceiling. The model assumes enough parallelism and omits launch and other latency limits. Caches and implementation details change the traffic count. The whole application may mix all three regimes. Do not infer a bottleneck from GPU-Util alone.

- [Horace He](https://horace.io/brrr_intro.html)
- [NVIDIA performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#understanding-performance)
- [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA dense compute peaks](https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput)

## 6. Fusion avoids writing and rereading an intermediate

1.5 minutes

The dataflow is a teaching schematic, corroborated by the PyTorch Performance Tuning Guide section on fusion. The dependent operations t = x * 2 followed by y = t + 1 can be combined into one pointwise kernel. Each thread can keep its temporary value in registers, so the entire tensor need not fit in shared memory. The logical accesses are read x, write t, read t, write y versus read x, write y. For N = 2^26 BF16 elements, one tensor is 2^27 bytes = 128 MiB. The stated model assumes all these passes reach HBM: unfused traffic is 512 MiB; fused traffic is 256 MiB. Actual HBM transactions may differ because of cache hits, write behavior and implementation choices. NVIDIA Best Practices supplies bytes-read plus bytes-written divided by time as an effective-bandwidth calculation. At 3.35 TB/s, the traffic-only times under this model are 160.26 and 80.13 microseconds; they are not measured latency or a guarantee of 2x acceleration. Compilers such as torch.compile can fuse eligible operations, but fusion is not guaranteed and excessive register pressure can matter. One PyTorch operator need not universally equal one GPU kernel.

- [Horace He](https://horace.io/brrr_intro.html)
- [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-operations)
- [CUDA bandwidth guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#effective-bandwidth-calculation)

## 7. Weight precision and training memory

2.5 minutes

Count unique parameters. The weight table gives ideal packed storage for exactly 8 billion parameters. Quantization adds scales and padding, and unquantized layers need their actual dtype. KV precision is independent. Training uses the explicitly listed 2+2+4+8 byte Adam layout. Real implementations may differ. Saved activations and transient allocations are extra. Checkpointing reduces saved activations through recomputation, not the listed states.

- [Stanford CS336, Lecture 2](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)
- [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html)
- [NVIDIA mixed precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)

## 8. Prefill, decode and the KV cache

2.5 minutes

Prefill processes the prompt and may produce the first output token from its last position. Decode processes generated tokens and appends their K/V. S counts processed prompt plus generated positions, so a sampled token enters KV when it is processed. GQA shares one K/V head pair across several query heads. The drawing shows one of eight groups in the teaching model. Past Q is not cached for ordinary autoregressive decode. New Q still attends to cached K/V. Formula assumes a conventional per-layer cache and no sliding window, latent cache or physical prefix sharing.

- [Hugging Face KV cache](https://huggingface.co/docs/transformers/main/cache_explanation)
- [Scaling Book: Transformers](https://jax-ml.github.io/scaling-book/transformers/)

## 9. Question 1: inference in a 24 GiB budget

1.5 minutes

Ask for the weight and KV formulas with units. Use the next slide for the solution. All cases have identical architecture and concurrency. Context lengths count currently processed cached positions. The case labels specify weight precision, while KV stays BF16. The task is to calculate payload and identify what remains unknown about actual peak memory.



## 10. Solution 1: memory payload and remaining budget

2 minutes

The per-token product is 131072 bytes across all layers. One 4096-token request is 0.5 GiB, so eight are 4 GiB. At 8192 tokens each, eight are 8 GiB. Weight bytes are 8e9 times bytes/value, then divide by 2^30. The 24 GiB device budget is explicitly binary. Quantization metadata, prefill workspace and runtime allocation remain outside payload. No case is guaranteed safe without these extras. A sufficient 3 GiB reserve for all extras would allow floor((24-14.901161-3)/0.5)=12 fixed-length requests in case A.



## 11. Model FLOPs utilization (MFU)

1.5 minutes

One multiply-add counts as two FLOPs. 6P per training token approximates dense weight matrix operations: about 2P forward and 4P backward. It is not three equal phases including the optimizer. Precise accounting must include attention and architecture details, especially long contexts. Use active computation for MoE rather than total parameters. Measured global token throughput is essential. MFU excludes extra checkpoint recomputation; a hardware FLOPs utilization measure may include it. Inference uses forward-only work, roughly 2P plus attention, not the training 6P numerator.

- [PaLM, Appendix B](https://jmlr.org/papers/volume24/22-1144/22-1144.pdf#page=90)
- [NVIDIA dense compute peaks](https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput)

## 12. Question 2: estimating MFU

1 minutes

Use the next slide for the solution. The throughput is global across the GPU group, not per device. Use dense BF16 Tensor Core peak. Assume the distributed state layout fits memory and ignore attention for this 6P estimate.

- [PaLM, Appendix B](https://jmlr.org/papers/volume24/22-1144/22-1144.pdf#page=90)
- [NVIDIA dense compute peaks](https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput)

## 13. Solution 2: 36.4% MFU

1.5 minutes

Useful model arithmetic is 2.880 PFLOP/s and matching group peak is 7.912 PFLOP/s. Do not multiply global token throughput by GPU count again. GPU-Util reports the fraction of a sampling interval with one or more kernels executing, not useful throughput relative to peak. Memory, communication, non-matmul work and launch gaps can reduce MFU, and MFU alone does not isolate the bottleneck. Attention is omitted from this simplified numerator.

- [PaLM, Appendix B](https://jmlr.org/papers/volume24/22-1144/22-1144.pdf#page=90)
- [NVIDIA dense compute peaks](https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput)
- [NVIDIA GPU-Util definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization)

## 14. Prefix caching across requests

1.5 minutes

The diagram shows one physical cached prefix referenced by several requests with private suffixes. Ordinary within-request KV caching, cross-request prefix reuse and hardware L1/L2 caching are distinct. Reuse requires matching token IDs and relevant model/cache identity, resident data and supported block granularity. Current vLLM token hit/query counters use tokens; use counter deltas from the same window and check the deployed version. Physical sharing and hit rate are different quantities. Prefix reuse mainly saves prefill, not all historical attention reads during decode.

- [vLLM prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM metrics](https://docs.vllm.ai/en/stable/usage/metrics/#general-metrics)

## 15. Question 3: prefix hits and shared KV

1 minutes

All prompts finish prefill sequentially. The first lookup is cold. Same model/cache identity, no eviction, no decode, matching full 16-token blocks, and all eight request states remain resident. Suffixes differ at their first token, excluding extra reusable suffix blocks. These assumptions establish the exact hit count and shared payload. Use the next slide for the solution.

- [vLLM prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)

## 16. Solution 3: cache reuse and physical storage

1.5 minutes

Cold token hits are 7 times 3072 = 21504, divided by 32768 queried tokens. Seven of eight requests have a hit. Count the shared prefix once and eight suffixes separately. Unique KV payload is exactly 1476395008 bytes, or 1.375 GiB. Unshared payload would be 4 GiB. Warm before all lookups gives 75% token hit rate and 100% request hit rate. The drawing is token-count proportional and excludes allocator reservations, metadata and unused capacity.

- [vLLM prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM metrics](https://docs.vllm.ai/en/stable/usage/metrics/#general-metrics)

## 17. Use timelines and measurements to find the bottleneck

1.5 minutes

These are original schematic timelines, not traces from a measured workload. The diagrams use an ordered GPU stream. Top: the short kernel finishes before the CPU has prepared the next launch, producing gaps on the illustrated GPU stream. A real trace needs inspection of all relevant streams and host activities before attributing gaps to CPU overhead. Bottom: the CPU queues later kernels while earlier kernels are running, hiding much of the submission cost. Arrow endpoints identify the launch-to-kernel correspondence; empty space is not a separate computation. PyTorch CUDA semantics corroborates asynchronous queues, stream ordering and the need for CUDA events or synchronization when timing GPU work. Warm up separately, then measure representative work; synchronize around end-to-end wall-clock timing or wait for event completion before reading GPU elapsed time. PyTorch memory management defines allocated as tracked live tensor allocations, reserved as the caching-allocator pool containing them, and peak as the maximum since reset. Resetting peak statistics does not remove pre-existing allocations. GPU-Util is active-kernel time, not model FLOPs utilization; use the measured token rate and the declared FLOP model for MFU. CUDA Graphs can reduce submission work for eligible workloads, but do not inherently fuse the GPU kernels.

- [Horace He](https://horace.io/brrr_intro.html)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
- [PyTorch memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)

## 18. Memory estimates and performance diagnosis

0.5 minutes

Use this as the closing reference. Capacity and performance are different questions. A payload calculation helps rule configurations out, but real peak allocations settle whether a configuration fits. Arithmetic intensity identifies a theoretical resource balance, while a profile identifies launch gaps, memory behavior and execution paths. Preserve all model and workload assumptions when comparing an optimization.



## 19. Discussion and references

0.5 minutes

Transition at minute 30. Discussion: choose a numerical question and change one assumption. Ask whether doubled context, different weight precision, a warm prefix, or lower launch overhead changes the answer. The supporting lecture and article motivate the diagrams, while NVIDIA and PyTorch documentation supplies architecture and implementation qualifications.

