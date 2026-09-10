# Week 1: 30-minute speaker notes

22 slides. 30 minutes of planned material. Questions 1–2 each have a separate solution slide immediately afterward. Prefix-cache hit rates and their worked exercise are reserved for Week 4.

## 1. GPU performance and memory for LLMs

0.5 minutes

The 30-minute presentation follows one story: where time is spent, how data movement can be reduced, and what must fit in memory. The two numerical questions keep a separate solution on the following slide. All memory calculations use decimal GB and MB. The dense 8B model and supplied KV cost are teaching assumptions.



## 2. CPU execution, dispatch and asynchronous launches

1.5 minutes

x already resides on the GPU. CPU code selects a backend/implementation and submits arguments and tensor addresses. No whole-tensor CPU copy is implied. Dtype-specific selection may occur within the backend. A kernel is a GPU function executed by many threads. One PyTorch operation can launch zero, one or multiple kernels. The timeline illustrates allowed overlap, not measured duration. Same-stream operations stay ordered. CUDA events or synchronization are needed for GPU timing.

- [PyTorch dispatcher](https://docs.pytorch.org/tutorials/advanced/dispatcher.html)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)

## 3. GPU execution and memory model

2 minutes

Read the left panel as the logical programming model and the right panel as the hardware that executes it. A kernel launch creates a grid of blocks. Threads within a block cooperate using shared memory; warps group 32 threads. SM means Streaming Multiprocessor. A block executes on one SM, while an SM can host several resident blocks if registers, shared memory and other limits allow. The scheduling arrow illustrates the mapping, not a permanent one-to-one pairing. Each SM contains schedulers and compute units, a physical register file and a combined physical L1/shared-memory resource. Registers hold logically private thread values. L1 is hardware-managed caching; shared memory is a kernel-managed workspace. Device-wide global tensors normally reside in HBM, with L2 shared across SMs. HBM is stacked DRAM beside the compute die in the GPU package. Global memory is not exclusively owned by one grid and may survive kernel completion. CUDA local memory, omitted here, is thread-private device-memory-backed storage, for example register spills; it is not another name for on-chip SRAM. Cluster-level distributed shared memory is outside this introductory model. Native schematic adapted from CS336 Lecture 5, pages 10-12, and the NVIDIA CUDA programming guide.

- [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [CS336 Lecture 5, pp. 10–12](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=12)

## 4. H100 SXM 80 GB memory hierarchy

1 minutes

The pyramid shows locality and sharing scope, not capacity-scaled areas or a compulsory path. Registers hold per-thread working values; shared memory is a cooperative workspace local to an SM. L1 and L2 are hardware-managed caches. HBM is stacked DRAM alongside the compute die in the package. H100 SXM has 80 GB HBM and peak bandwidth of 3.35 TB/s. The approximately 50 MB L2 label follows the vendor specification. SRAM is a memory technology used for on-chip caches and shared memory, not an extra hierarchy tier. Focus on where data can be reused, rather than per-SM capacity arithmetic.

- [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)
- [Micron HBM](https://www.micron.com/products/memory/hbm)

## 5. Memory bandwidth can limit GPU speed

1.5 minutes

Read the supplied Horace He illustration as a warehouse supplying a factory and receiving its output. The boxes and triangles represent input and output data, not literal GPU hardware. On an H100, the warehouse is HBM, which is DRAM; the factory includes GPU arithmetic units and nearby on-chip storage. Bandwidth is the number of bytes transferred per second across a specified interface. Imagine the data path supplies 10 items per second while the factory can process 100: the output cannot exceed the supply rate. Doubling arithmetic capacity alone changes little. With enough data reuse, arithmetic may instead become the limiting resource. These rates are an analogy, not hardware measurements. The terms describe a workload on particular hardware, not a permanent property of a GPU. Here memory-bound means memory-bandwidth limited, distinct from an out-of-memory capacity error. Small workloads may also be limited by submission overhead, covered in the later timeline slide. Source illustration supplied by the presenter from Horace He; technical distinctions corroborated by NVIDIA GPU Performance Background.

- [Illustration: Horace He (2022)](https://horace.io/brrr_intro.html)
- [NVIDIA performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#understanding-performance)

## 6. Four ways to spend less time moving data

1 minutes

Keep the factory illustration visible while introducing the next four techniques. Lower precision reduces bytes per value. Fusion removes intermediate global-memory traffic between operators. Coalescing packs useful accesses into fewer memory transactions. Tiling reuses a small working set near compute. The categories can overlap in an implementation. The goal is faster useful output, not forcing every workload to become compute-bound. Extra arithmetic, launch overhead, dependencies and finite on-chip resources still matter. The drawing simplifies the path through caches and registers. HBM is GPU DRAM, and SRAM implements on-chip caches and shared memory.

- [Illustration: Horace He (2022)](https://horace.io/brrr_intro.html)
- [CS336 Lecture 5: GPU optimization](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=27)

## 7. Lower precision moves fewer bytes per value

1.5 minutes

This is a storage and traffic comparison for the same number of values. FP32 uses four bytes and BF16 or FP16 uses two. The input values may round differently; precision choices require accuracy checks. A 16-bit format reduces ideal weight storage for eight billion parameters from 32 GB to 16 GB. Eligible matrix operations can also use high-throughput Tensor Core paths, but low precision does not guarantee a two-times end-to-end speedup. Accumulation, optimizer states and some numerically sensitive operations may remain FP32. Tensor Cores are specialized matrix-arithmetic units within an SM; actual supported precision and peak depend on the hardware.

- [NVIDIA mixed precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [CS336 Lecture 5: GPU optimization](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=27)

## 8. Fusion avoids a round trip for an intermediate tensor

1.5 minutes

Compare t = x * 2 followed by y = t + 1. Two separate GPU kernels logically read x, write t, read t, and write y. A fused elementwise kernel can keep the intermediate value in registers and only read x and write y. The schematic counts logical global-memory accesses, not measured HBM transactions; caches and compiler choices affect physical traffic. The arithmetic is unchanged in this example, but other fusions may change numerical rounding. Fusion also reduces the number of kernel launches. It does not make every operation compute-bound, and PyTorch eager mode does not promise fusion from writing both operators in one Python expression.

- [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-operations)
- [Horace He](https://horace.io/brrr_intro.html)

## 9. Coalescing: neighboring threads read nearby values

2 minutes

Each thread requests one value in one load instruction. Dark cells mark requested values; outlined groups show transfer chunks. Both examples request eight values. With the deliberately simplified four-value chunk, contiguous addresses 0 through 7 touch two chunks, while addresses 0, 4, 8, ..., 28 touch eight. The hardware combines accesses into transactions; the programmer influences this through layout and thread-to-data mapping. The drawing is not an actual eight-thread warp or a hardware transaction-size specification. CUDA warps contain 32 threads. On modern CUDA devices the guide describes coalescing in terms of the aligned 32-byte segments needed to service the active lanes. Exact thread order need not match address order: what matters is the set of addresses accessed together. Alignment, caching and reuse also affect physical traffic and measured speed. This is an efficiency comparison, not a promised four-times end-to-end speedup. Native teaching schematic informed by CS336 Lecture 5, pages 37–39, with the transaction explanation checked against NVIDIA CUDA Best Practices, Coalesced Access to Global Memory.

- [CS336 Lecture 5, pp. 37–39](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=37)
- [CUDA memory coalescing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

## 10. Tensor layout determines which addresses are nearby

1 minutes

Both panels use the same contiguous row-major 4 by 4 array. Cell numbers are element offsets from the array start, not stored tensor values or byte addresses. The offset formula is 4 times row plus column. Increasing the column within the same row advances by one element; increasing the row while holding the column fixed advances by four. Each selected cell is read by one of four neighboring threads in the same illustrative load. The left mapping varies the column index and requests 0, 1, 2, 3. The right varies the row index and requests 0, 4, 8, 12. The bottom strip shows the shared physical element order 0 through 15, grouped by matrix row. This tiny array illustrates address mapping, not actual transaction counts or a measured performance ratio. In particular, some of these offsets may still fit in one hardware segment depending on element size and alignment. Coalescing concerns the addresses a warp accesses together, rather than whether one individual thread eventually walks a contiguous row. Libraries choose mappings for high-level operations; PyTorch tensor strides describe the layout and a transpose view need not copy data. A larger row stride can spread accesses across many transfer chunks. Native diagram based on CS336 Lecture 5, pages 37–39, with the precise coalescing rule checked against NVIDIA CUDA Best Practices.

- [CS336 Lecture 5, pp. 37–39](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=37)
- [CUDA memory coalescing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

## 11. Tiling: load a small region, then reuse it

2 minutes

Allow 2 minutes. Tiling divides a large matrix multiplication C = A × B into smaller output regions and input stages. This is a simplified shared-memory implementation, not a claim that every GEMM uses exactly this data path. A thread block cooperatively loads an A tile and a B tile from global memory into shared memory, reuses these values to update its output tile, and keeps partial sums in registers. Move along the reduction dimension k: load the next A/B tile pair, then add its contribution to the same partial sums. Only after all k stages does the block write the completed output tile. Global loads may be served by caches; arrows describe the conceptual storage roles, not a mandatory uncached HBM transaction per load. The drawings show one stage and one output tile, not all concurrently executing blocks. Whole matrices need not fit in shared memory. Shared memory and registers are finite: larger tiles can improve reuse but consume more resources and may reduce the number of resident blocks. Synchronization is required when collaborating threads exchange staged data. This concept introduces reuse without CUDA syntax. Based on CS336 Lecture 5, pages 40–43, and the NVIDIA matrix-multiplication shared-memory example.

- [CS336 Lecture 5, pp. 40–43](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=40)
- [CUDA shared-memory matrix multiply](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)

## 12. Tiling reuses inputs across an output tile

2 minutes

Allow 2 minutes. A and B are both 4×4, and this example computes only C[0:2,0:2], the top-left 2×2 output tile. The reduction dimension has four entries and is traversed in two stages of width two. At stage 1, load A[0:2,0:2] and B[0:2,0:2]; at stage 2, load A[0:2,2:4] and B[2:4,0:2]. The solid outlines mark stage 1, and dashed outlines mark stage 2. The a cell means A[0,0] and contributes to C[0,0] and C[0,1], so it is reused across two output columns. The b cell means B[0,0] and contributes to C[0,0] and C[1,0], so it is reused across two output rows. Without explicit reuse, computing four dot products independently issues 4×(4 A values + 4 B values)=32 logical global-memory input loads. Staging each input tile once issues 2×(4 A values + 4 B values)=16 such loads, followed by shared-memory/register reuse. Both versions calculate the same four outputs with the same 16 multiplications and reductions. Output stores are excluded and unchanged. These are illustrative logical input-load counts, not measured HBM transactions: caches can already satisfy repeated global loads, and shared-memory reads and synchronization also have costs. Do not infer a 2× runtime improvement or a halving of all memory traffic. Tile size is constrained by shared-memory/register capacity and occupancy; this toy tile size is explanatory, not a performance recommendation. Based on CS336 Lecture 5, pages 40–43, and NVIDIA’s shared-memory C=AB example.

- [CS336 Lecture 5, pp. 40–43](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=40)
- [CUDA shared-memory matrix multiply](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)

## 13. Training stores much more than the weights

1.5 minutes

Use one explicitly specified mixed-precision Adam layout: BF16 weights, BF16 gradients, FP32 master weights, and two FP32 moment tensors. These persistent states use 2+2+4+8 = 16 bytes per parameter, or 128 GB for an eight-billion-parameter model. This is an example layout, not a universal Adam constant; actual gradient and master-copy policies differ. Activations and temporary workspace add to the persistent state. Activation checkpointing stores fewer intermediate activations by recomputing them later. It does not shrink the stated weights or optimizer tensors. Multi-GPU state sharding is covered in Week 3.

- [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html)
- [NVIDIA mixed precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)

## 14. The KV cache grows as generation continues

2 minutes

Prefill processes the prompt and creates cached key/value vectors for its processed positions. During decode, a new token contributes its own key/value vectors and its query attends to the cached history. This avoids recomputing past key/value projections; it does not eliminate reading historical keys and values. Past queries are not needed for ordinary autoregressive decode. A sampled token contributes K/V when it is processed by the model. Model-specific bytes per cached token are supplied in the exercise; the architecture derivation is intentionally outside this introductory presentation. Use the number of currently processed positions, not only generated-output length. No cross-request sharing is assumed.

- [Hugging Face KV cache](https://huggingface.co/docs/transformers/main/cache_explanation)

## 15. Question 1: what fits in a 24 GB memory budget?

1.5 minutes

Use a decimal 24 GB memory budget. Both cases use the same dense eight-billion-parameter model with BF16 weights and eight concurrent requests. Supply the model-specific KV rate as approximately 0.131 MB per processed token across all layers. The reference rate is 131072 bytes; there is no need to derive it from attention-head architecture here. Use 4000 and 8000 cached positions, not powers of two. Both weight and KV payload are ideal stored data before activations, workspace and runtime memory. Compare whether a payload fits before assuming the actual workload fits.



## 16. Solution 1: doubling context can exhaust the budget

1.5 minutes

Weights are eight billion times two bytes, or 16 GB. With a reference rate of 131072 bytes per cached token, eight requests at 4000 tokens use 4.194304 GB KV; at 8000 tokens they use 8.388608 GB. Round results to one decimal for presentation. Using the supplied rounded rate 0.131 MB gives the same one-decimal answers. Case A leaves about 3.8 GB for activations, workspace and other runtime allocations; this is not a guaranteed fit. Case B exceeds the 24 GB budget with weights and KV alone. The model and concurrency are fixed, so the difference isolates cached context length.



## 17. Use timelines and measurements to find the bottleneck

1.5 minutes

These are original schematic timelines, not traces from a measured workload. The diagrams use an ordered GPU stream. Top: the short kernel finishes before the CPU has prepared the next launch, producing gaps on the illustrated GPU stream. A real trace needs inspection of all relevant streams and host activities before attributing gaps to CPU overhead. Bottom: the CPU queues later kernels while earlier kernels are running, hiding much of the submission cost. Arrow endpoints identify the launch-to-kernel correspondence; empty space is not a separate computation. PyTorch CUDA semantics corroborates asynchronous queues, stream ordering and the need for CUDA events or synchronization when timing GPU work. Warm up separately, then measure representative work; synchronize around end-to-end wall-clock timing or wait for event completion before reading GPU elapsed time. PyTorch memory management defines allocated as tracked live tensor allocations, reserved as the caching-allocator pool containing them, and peak as the maximum since reset. Resetting peak statistics does not remove pre-existing allocations. GPU-Util is active-kernel time, not model FLOPs utilization; use the measured token rate and the declared FLOP model for MFU. CUDA Graphs can reduce submission work for eligible workloads, but do not inherently fuse the GPU kernels.

- [Horace He](https://horace.io/brrr_intro.html)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
- [PyTorch memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)

## 18. MFU measures useful arithmetic against a compute peak

1 minutes

MFU is a model-level measure. Divide useful model FLOPs per second by the matching aggregate device peak. Approximate dense-transformer training weight-matrix work as 6P FLOPs per token: about 2P forward and 4P backward. Count a multiply-add as two FLOPs. This approximation omits attention and other model details and is not the forward-only inference formula. Use measured global throughput across the group, and match the hardware peak to precision and computation path. GPU activity time is a separate denominator. High activity can coexist with low useful arithmetic throughput because kernels may spend time moving data or doing other work.

- [PaLM, Appendix B](https://jmlr.org/papers/volume24/22-1144/22-1144.pdf#page=90)
- [NVIDIA dense compute peaks](https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput)
- [NVIDIA GPU-Util definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization)

## 19. Question 2: estimating MFU

1 minutes

Use the next slide for the solution. The throughput is global across the GPU group, not per device. Use dense BF16 Tensor Core peak. Assume the distributed state layout fits memory and ignore attention for this 6P estimate.

- [PaLM, Appendix B](https://jmlr.org/papers/volume24/22-1144/22-1144.pdf#page=90)
- [NVIDIA dense compute peaks](https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput)

## 20. Solution 2: 36.4% MFU

1.5 minutes

Useful model arithmetic is 2.880 PFLOP/s and matching group peak is 7.912 PFLOP/s. Do not multiply global token throughput by GPU count again. GPU-Util reports the fraction of a sampling interval with one or more kernels executing, not useful throughput relative to peak. Memory, communication, non-matmul work and launch gaps can reduce MFU, and MFU alone does not isolate the bottleneck. Attention is omitted from this simplified numerator.

- [PaLM, Appendix B](https://jmlr.org/papers/volume24/22-1144/22-1144.pdf#page=90)
- [NVIDIA dense compute peaks](https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput)
- [NVIDIA GPU-Util definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization)

## 21. Match the optimization to the bottleneck

0.5 minutes

Close the story by tying each technique to the cost it changes. Lower precision reduces bytes per value, subject to accuracy and supported compute paths. Fusion can avoid temporary global-memory round trips and reduce launches. Coalescing combines neighboring accesses into fewer transactions. Tiling reuses a working set on chip and may work together with coalesced loads. Large tiles can consume too much shared memory or registers, so benchmark rather than maximizing tile size blindly. None of these techniques guarantees a particular speedup. Capacity still constrains the viable model and workload.



## 22. Discussion and further reading

0.5 minutes

The planned presentation totals 30 minutes. Use the next 15 minutes for discussion, with five minutes of meeting buffer. Ask the audience to explain in words what cost an optimization removes before asking for more calculations. The separately prepared interview problem proposals remain for presenter review and have not been inserted. Prefix-cache hit-rate metrics remain in Week 4.

