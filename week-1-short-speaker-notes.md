# Week 1: 30-minute speaker notes

16 slides. 26 minutes of planned material, with 4 minutes for clarification within a 30-minute presentation. Questions 1–2 each have a separate solution slide immediately afterward. Prefix-cache hit rates and their worked exercise are reserved for Week 4.

## 1. GPU memory and performance for LLMs

0.5 minutes

The material and two short numerical questions total 26 minutes, leaving 4 minutes for clarification within the 30-minute presentation. Each question has its solution on the next slide. Use the remaining meeting time to revisit changed assumptions. The numerical model is an explicitly specified dense 8B GQA teaching model.



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

## 5. Memory bandwidth can limit GPU speed

2 minutes

Read the supplied Horace He illustration as a warehouse supplying a factory and receiving its output. The boxes and triangles represent input and output data, not literal GPU hardware. On an H100, the warehouse is HBM, which is DRAM; the factory includes GPU arithmetic units and nearby on-chip storage. Bandwidth is the number of bytes transferred per second across a specified interface. Imagine the data path supplies 10 items per second while the factory can process 100: the output cannot exceed the supply rate. Doubling arithmetic capacity alone changes little. With enough data reuse, arithmetic may instead become the limiting resource. These rates are an analogy, not hardware measurements. The terms describe a workload on particular hardware, not a permanent property of a GPU. Here memory-bound means memory-bandwidth limited, distinct from an out-of-memory capacity error. Small workloads may also be limited by submission overhead, covered in the later timeline slide. Source illustration supplied by the presenter from Horace He; technical distinctions corroborated by NVIDIA GPU Performance Background.

- [Illustration: Horace He (2022)](https://horace.io/brrr_intro.html)
- [NVIDIA performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#understanding-performance)

## 6. Less data movement leaves more time for useful compute

2 minutes

The supplied annotated illustration highlights traffic between GPU main memory and the compute die. HBM is GPU DRAM; SRAM provides fast on-chip storage such as caches and shared memory. The drawing simplifies actual access paths, which also include registers and caches. For t = x * 2 followed by y = t + 1, separate kernels may write the entire intermediate t and read it again. Fusion can keep temporary values on chip and produce the same result with fewer global-memory accesses. Cache hits and implementation details affect physical HBM traffic; neither two-times speedup nor a compute-bound result is guaranteed. A batch may reuse the same weight values across several tokens, increasing useful work per weight load. It can improve aggregate throughput while increasing per-request latency, so compare both. For a fixed algorithm, much of the useful arithmetic remains necessary, whereas some transfers and submission work can be removed. Compute-bound execution is a useful outcome when reached by reducing those costs, not by inserting extra arithmetic. Some well-optimized operations naturally remain bandwidth-bound. Measure end-to-end latency and throughput. The next material counts the storage needed by those weights and cached values. Illustrations: Horace He, supplied by the presenter. Fusion explanation: PyTorch Performance Tuning Guide; batching: Scaling Book inference analysis.

- [Illustration: Horace He (2022)](https://horace.io/brrr_intro.html)
- [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-operations)
- [Scaling Book: inference](https://jax-ml.github.io/scaling-book/inference/)

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

## 14. Use timelines and measurements to find the bottleneck

1.5 minutes

These are original schematic timelines, not traces from a measured workload. The diagrams use an ordered GPU stream. Top: the short kernel finishes before the CPU has prepared the next launch, producing gaps on the illustrated GPU stream. A real trace needs inspection of all relevant streams and host activities before attributing gaps to CPU overhead. Bottom: the CPU queues later kernels while earlier kernels are running, hiding much of the submission cost. Arrow endpoints identify the launch-to-kernel correspondence; empty space is not a separate computation. PyTorch CUDA semantics corroborates asynchronous queues, stream ordering and the need for CUDA events or synchronization when timing GPU work. Warm up separately, then measure representative work; synchronize around end-to-end wall-clock timing or wait for event completion before reading GPU elapsed time. PyTorch memory management defines allocated as tracked live tensor allocations, reserved as the caching-allocator pool containing them, and peak as the maximum since reset. Resetting peak statistics does not remove pre-existing allocations. GPU-Util is active-kernel time, not model FLOPs utilization; use the measured token rate and the declared FLOP model for MFU. CUDA Graphs can reduce submission work for eligible workloads, but do not inherently fuse the GPU kernels.

- [Horace He](https://horace.io/brrr_intro.html)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
- [PyTorch memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)

## 15. Memory estimates and performance diagnosis

0.5 minutes

Use this as the closing reference. Capacity and performance are different questions. A payload calculation helps rule configurations out, but real peak allocations settle whether a configuration fits. Arithmetic intensity identifies a theoretical resource balance, while a profile identifies launch gaps, memory behavior and execution paths. Preserve all model and workload assumptions when comparing an optimization.



## 16. Discussion and references

0.5 minutes

The planned material ends around minute 26. Use the remaining presentation time for clarification, then begin the discussion. Choose a numerical question and change one assumption. Ask whether doubled context, different weight precision, more concurrent requests, or lower launch overhead changes the answer. The supporting lecture and article motivate the diagrams, while NVIDIA and PyTorch documentation supplies architecture and implementation qualifications.

