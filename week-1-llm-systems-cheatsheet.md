# Week 1 - LLM Systems Cheatsheet

Reference notes accompanying the Week 1 presentation: follow one 8B model from a GPU operation to a training or inference memory budget, then ask how efficiently it runs. Hardware constants use NVIDIA H100 SXM 80 GB. The model is a teaching example, not an exact checkpoint. All capacities below use decimal units: 1 KB = 1,000 bytes, 1 MB = 1,000,000 bytes, and 1 GB = 1,000,000,000 bytes.

## 1. GPU execution & memory hardware

Start with what happens when Python asks the GPU to compute.

### 1  From a Python operation to GPU work

**Host = CPU; device = GPU.** For a CUDA-resident tensor, `y = x * 2` runs CPU framework code, then GPU work. **Dispatch** selects an implementation appropriate for the device and tensor properties. **Launch** queues a kernel with addresses, arguments and a thread configuration; it does not copy the whole tensor from the CPU each time.

**CUDA kernel:** a GPU function run by many **threads** (execution instances). A **thread block** groups threads that cooperate through SMEM and normally execute on one **SM (Streaming Multiprocessor)**. An SM contains schedulers, compute units and local storage. CUDA cores perform general arithmetic; Tensor Cores accelerate matrix operations. An operation can launch zero, one or several kernels.

**Asynchronous** means the CPU can continue before queued GPU work finishes. A **CUDA stream** is an ordered queue: dependent operations in the same stream execute in order without a CPU wait between them. `torch.cuda.synchronize()` blocks the CPU until pending work on the selected/current CUDA device completes. Reading a GPU result on the CPU can also require waiting.

### 2  Storage roles and sharing

| Storage | Who uses/manages it? | What it holds |
| --- | --- | --- |
| Registers | Logically private to each thread; physical register file on an SM | Current operands and accumulators |
| SMEM / shared memory | Kernel-managed workspace, normally shared within a thread block | Tiles/data that threads reuse together |
| L1 cache | Hardware-managed; local to an SM | Copies of recently accessed data |
| L2 cache | Hardware-managed; shared across SMs | Cached data reused across GPU work |
| HBM | Device-wide GPU main memory | Weights, KV cache and large tensors |

**DRAM (Dynamic Random Access Memory)** needs refresh; **SRAM (Static Random Access Memory)** retains its state while powered without periodic refresh. Both are volatile. These are technologies: L1/L2 caches and SMEM are roles implemented with SRAM. Register files are specialized working storage.

**HBM = High Bandwidth Memory, a stacked DRAM architecture.** GPU HBM sits beside the compute die in the package. CPU main memory is a separate DRAM pool. FlashAttention Fig. 1 separates these physical pools; its label "CPU DRAM" does not exclude HBM from the DRAM technology family.

### 3  H100 SXM 80 GB constants - keep the scope attached

| Resource | Specification | Scope / caveat |
| --- | --- | --- |
| HBM3 | 80 GB; peak 3.35 TB/s | Whole GPU; not per SM |
| Dense BF16 Tensor Cores | 989 TFLOP/s | Whole GPU; no structured sparsity |

The first number tells us how much data fits; the second tells us how quickly HBM can transfer data; the third describes peak arithmetic throughput. **Latency** measures how long an operation or access takes. Cache hits can avoid HBM transfers, so HBM bandwidth is not a universal "DRAM-to-SRAM speed."

Sources: [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html), [PyTorch dispatcher](https://docs.pytorch.org/tutorials/advanced/dispatcher.html), [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution), [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html), [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/), [NVIDIA dense compute peaks](https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput), [Micron: HBM is stacked DRAM](https://www.micron.com/products/memory/hbm), [FlashAttention, Fig. 1 and Sec. 2.1](https://arxiv.org/pdf/2205.14135#page=2).

---

## 2. Precision & training memory

Count unique stored values, then account for the other allocations.

### 4  Count stored values

```text
Tensor payload = number of stored values x bytes per value
Weight payload M_weights = P x b_w
```

### 5  Weight-only storage for exactly 8,000,000,000 parameters

| Storage format | Bytes/value | Weight storage |
| --- | --- | --- |
| FP32 | 4 | 32 GB |
| BF16 or FP16 | 2 | 16 GB |

**BF16 and FP16 occupy the same memory:** 16 bits = 2 bytes per value. BF16 represents a wider range of magnitudes; FP16 provides finer precision within its narrower range. Both use half the storage of FP32.

Storage and computation formats can differ. In particular, BF16 model weights do not imply that every training state also uses BF16. Count shared storage once.

### 6  Full-parameter training: the explicit Adam example

```text
M_train = weights + gradients + optimizer states + master weights (if used)
          + saved activations + temporary / runtime allocations
```

| Persistent state | Chosen dtype | Bytes/parameter |
| --- | --- | --- |
| Model weights | BF16 | 2 |
| Gradients | BF16 | 2 |
| Master weights for updates | FP32 | 4 |
| Adam first moment m | FP32 | 4 |
| Adam second moment v | FP32 | 4 |
| TOTAL for this layout |  | 16 |

**8B example:** 16 x 8 x 10^9 = **128 GB** of states, before activations and extras. This layout cannot fit on one 80 GB H100.

**16 bytes/parameter applies to this stated layout.** Different state dtypes or optimizer implementations change the total. Peak memory depends on which allocations coexist.

Training also keeps selected forward values for backward. **Activation checkpointing** saves fewer of these values and recomputes them during backward; the weights, gradients and Adam moments still need storage. Parameter count alone cannot determine activation memory. Dividing states across GPUs changes the per-device budget, which we will revisit in the parallelism week.

Sources: [CS336: systems and memory](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py), [NVIDIA mixed precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html), [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html).

---

## 3. Inference memory & KV calculations

Inference avoids most training states, but it still needs more than model weights.

**Given for this teaching example:** 8B parameters, BF16 weights, and **131,072 bytes of KV storage per cached token**, about **0.131 MB**, across the whole model. This rate is supplied for the example; actual models differ. Requests keep independent KV storage. We compare 8 active requests at 4,000 and 8,000 cached tokens each.

### 7  What is cached, and when?

**Prefill** processes the prompt and populates K/V; its last prompt position can produce the first output token. **Decode** processes generated tokens iteratively. A sampled token enters the KV cache when the model processes it. S counts processed prompt + generated positions, not just the number of outputs.

**KV cache** stores past keys and values at each attention layer. A new query attends to these cached K/V; old queries are not needed for ordinary autoregressive decode. Caching avoids recomputing the past projections but does not eliminate reading historical K/V for attention.

### 8  Multiply the given KV rate by the cached token count

```text
KV storage = bytes per cached token x total cached tokens
M_KV = m_token x sum_i(S_i)
     = m_token x B x S  [B requests with equal cached length S]
```

Sum the cached lengths when requests differ. This counts stored K/V values; metadata and unused allocation can add memory.

**One request at 4,000 tokens:** 131,072 x 4,000 = 524,288,000 bytes = **0.524288 GB**.
**Eight requests:** 131,072 x 8 x 4,000 = **4.194304 GB**, approximately **4.19 GB**.
**At 8,000 tokens each:** double that to **8.388608 GB**, approximately **8.39 GB**.

### 9  Question 1: does it fit in a 24 GB budget?

```text
M_infer = weights + KV + working activations / workspaces + runtime overhead
```

| Cached tokens per request (8 requests) | Weights | KV | Weight + KV | Left for extras |
| --- | --- | --- | --- | --- |
| 4,000 | 16.00 | 4.19 | 20.19 | 3.81 |
| 8,000 | 16.00 | 8.39 | 24.39 | -0.39 |

**All table values are GB, rounded after calculating with the exact rate.** The shorter case leaves about 3.8 GB for other allocations, so it needs measurement before promising a fit. The longer case already exceeds the budget before those extras and cannot fit under these assumptions. Weights are stored once per model replica, not once per request.

For mental arithmetic, use **16 + 4.2 = 20.2 GB** and **16 + 8.4 = 24.4 GB**. Context growth and prefill workspaces also need room.

Sources: [Hugging Face: KV cache](https://huggingface.co/docs/transformers/main/cache_explanation), [Scaling Book: Transformer accounting](https://jax-ml.github.io/scaling-book/transformers/).

---

## 4. Model FLOPs utilization

Define the numerator, denominator and measurement window before calculating.

### 10  Model FLOPs utilization (MFU)

**FLOP** = one floating-point operation; **FLOP/s** = a rate. A multiply-add conventionally counts as 2 FLOPs. MFU compares useful model arithmetic per second with the matching theoretical peak of the GPU group.

```text
MFU = useful model FLOPs / (N_GPU x F_peak x elapsed seconds)
Dense training approximation: MFU = (6 x P x R) / (N_GPU x F_peak)
```

**R** is the measured **global** training token rate, in tokens/s; F_peak is FLOP/s **per GPU**. Equivalently use 6P x global tokens in a measured step as the numerator and include step time in the denominator.

**Why 6P?** Weight matrix operations cost roughly 2P FLOPs/token in forward + 4P in backward. This estimate omits attention work, which matters more at long context. For inference, count forward-only work instead (roughly 2P plus attention).

### 11  Question 2: 8B training on 8 H100 SXM GPUs

**Given:** 60,000 global training tokens/s; dense BF16 Tensor Core peak = 989 x 10^12 FLOP/s per GPU. Assume distributed model states fit; ignore attention in this estimate.
**Useful rate:** 6 x 8 x 10^9 x 60,000 = 2.880 x 10^15 FLOP/s.
**Group peak:** 8 x 989 x 10^12 = 7.912 x 10^15 FLOP/s.
**MFU = 2.880 / 7.912 = 36.4%.**

**MFU checks:** use the peak appropriate for the chosen arithmetic, and do not multiply global throughput by GPU count again. The usual MFU numerator excludes extra checkpoint recomputation. A measured token rate or step time is required.

**GPU-Util is different:** it measures the fraction of a sampling interval with one or more kernels running, not useful FLOPs / peak. 95% GPU-Util can coexist with 36.4% MFU. Low decode MFU can reflect a bandwidth limit; MFU alone does not diagnose the bottleneck or service quality.

Sources: [PaLM: MFU, Appendix B](https://jmlr.org/papers/volume24/22-1144/22-1144.pdf#page=90), [NVIDIA dense compute peaks](https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput), [NVIDIA GPU utilization definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization).

---

## 5. Why data movement changes performance

Having enough memory does not ensure fast execution. The GPU must also get useful data to its compute units. Coalescing, tiling and fusion address different sources of wasted movement.

### 12  Coalescing, tiling and fusion

**Coalescing makes a group of accesses efficient.** Arrange threads that execute together to read adjacent addresses, so the hardware can serve their requests with fewer memory transactions. Scattered addresses can transfer data that the threads never use. Coalescing improves how we access data; it does not mean we perform fewer mathematical operations.

**Tiling keeps reused data nearby.** Divide a large matrix multiplication into small pieces. Threads load input tiles into shared memory, reuse them for several calculations, and accumulate partial results in registers. The large matrices stay in HBM; only the working tiles need to fit on chip. Tile loads can also be coalesced.

**Fusion avoids storing an intermediate between operations.** Instead of launching one kernel for `t = x * 2` and another for `y = t + 1`, one kernel can compute both and keep a small intermediate in a register. It can save a kernel launch and the write/read of `t`.

These techniques can be combined: fetch adjacent values efficiently, reuse a tile locally, and avoid unnecessary intermediate tensors. They must preserve the required numerical behavior, and their benefit depends on the workload. Sources: [NVIDIA: coalesced access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory), [NVIDIA: shared memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory), [CS336 Lecture 5: GPUs, slides 26-39](https://stanford-cs336.github.io/spring2025-lectures/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf#page=26).

### Optional calculation: how much traffic can fusion save?

Let `x` contain **100 million BF16 values**, so each full tensor is **200 MB**. Model each listed read or write as a full transfer to or from HBM.

| Implementation | Modeled full-tensor passes | HBM traffic |
| --- | --- | --- |
| Two kernels | Read x; write t; read t; write y | 4 x 200 = 800 MB |
| One fused kernel | Read x; write y | 2 x 200 = 400 MB |

```text
Traffic-model time bound = modeled HBM bytes / peak HBM bytes per second
800 x 10^6 / (3.35 x 10^12) = 238.81 microseconds
400 x 10^6 / (3.35 x 10^12) = 119.40 microseconds
```

These bounds assume every listed transfer reaches HBM, no cache benefit or extra transactions, and the H100 SXM peak bandwidth. They are estimates, not measured kernel times. Half the modeled traffic does not guarantee twice the real speed: launch overhead, computation and achieved bandwidth also matter.

Sources: [CUDA bandwidth guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#bandwidth), [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/), [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html).

---

## 6. Measurement & memory tradeoffs

Use estimates to explain the budget, then validate the workload that will actually run.

### 13  PyTorch memory measurements

| API / setting | What it means |
| --- | --- |
| memory_allocated() | Current live tensor allocation tracked by the PyTorch CUDA allocator |
| memory_reserved() | Memory held by the caching allocator; includes allocated memory |
| max_memory_allocated() | Peak tracked tensor allocation since the peak-statistics reset |
| model.eval() | Evaluation behavior for modules; does not disable autograd |
| torch.inference_mode() | Disables gradient recording and additional inference-unneeded tracking |

```python
model.eval()
# Warm up separately with representative inputs.
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    result = representative_inference_workload()
torch.cuda.synchronize()
peak_gb = torch.cuda.max_memory_allocated() / 10**9
```

The workload function is a placeholder for your prefill/decode code. Resetting statistics does not remove baseline tensors. Allocated + reserved double-counts memory. CUDA/runtime or library allocations may sit outside PyTorch tracking, so the peak is not the whole process footprint. Measure initialization/graph-capture peaks separately if relevant.

**Timing:** a CPU timer around an asynchronous launch may only measure submission. Use CUDA events or synchronize the boundaries of a warmed-up measurement. Measure representative sequence lengths, concurrency and a complete training step when estimating training peak.

### 14  What changes when you change the workload?

| Change (hold other choices fixed) | Direct effect |
| --- | --- |
| Double active requests or their cached lengths | Double KV storage; weights per replica stay fixed |
| FP32 weights -> BF16 weights | Halve weight storage; other state formats must be specified separately |
| Activation checkpointing | Fewer saved activations; more recomputation |
| Divide model states across GPUs | Change the per-device budget; account for what each device actually stores |

Sources: [PyTorch memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management), [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution), [CS336: systems and memory](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py), [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html).

---

## 7. Quick lookup & interview checklist

Keep units, scope and assumptions visible.

### Unit conversions

| Quantity | Meaning / conversion |
| --- | --- |
| Bit vs byte | 8 bits = 1 byte; 16 bits = 2 bytes; 32 bits = 4 bytes |
| Storage | 1 KB = 10^3 bytes; 1 MB = 10^6 bytes; 1 GB = 10^9 bytes |
| Bandwidth | 1 TB/s = 10^12 bytes/s |
| Convert | GB = bytes / 10^9; seconds = bytes / (bytes per second) |

### Core equations

```text
Weights:    M_weights = P x b_w
Training:   M_states = 16P bytes  [only the stated Adam layout]
KV/token:   m_token = 131,072 bytes  [given for this teaching model]
KV total:   M_KV = m_token x sum_i(S_i)  [independent per-request storage]
Fit:        weights + KV + all other live allocations <= budget
Dense training MFU (approx): (6 x P x global tokens/s) / (N_GPU x F_peak)
Time bound: modeled transferred bytes / peak bytes per second
```

**Symbols:** P = stored parameter count; b_w = bytes per weight; m_token = KV bytes per cached token across the model; S_i = currently cached tokens for request i; N_GPU = GPU count; F_peak = matching FLOP/s per GPU. Equations count stored values unless stated otherwise.

### 15  A reliable interview answer in five steps

**1. State the target.** Weight storage, inference peak, training peak or token throughput?
**2. State assumptions.** GPU, units, parameter count, precision for each state, KV bytes per token and active request lengths.
**3. Write the equation with units.** Separate hardware constants, chosen inputs and measured quantities.
**4. Calculate the known payload.** Add a stated reserve or identify what needs measurement. Round only at the end.
**5. Explain the limit.** A payload fit is not a peak-memory guarantee; a theoretical traffic time is not a benchmark; a utilization percentage is not a latency prediction.

**Numbers to remember for this teaching model:** BF16 weights **16 GB**; assumed Adam states **128 GB**; given KV rate **about 0.131 MB/token**; 8 x 4,000 cached tokens require **about 4.2 GB** of KV; base inference weight + KV is **about 20.2 GB**; example MFU **36.4%**. H100 SXM has **80 GB HBM**, **3.35 TB/s** peak HBM bandwidth and **989 TFLOP/s** peak dense BF16 throughput.

**Scope:** covers the Week 1 talk and additional reference material for discussion. Worked questions use authored teaching numbers, not verified verbatim frontier-lab interview questions. Each section links to primary documentation; hardware constants and implementation details must be checked for other variants or software versions.

Sources: [CS336: systems and memory](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py), [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html), [PaLM: MFU, Appendix B](https://jmlr.org/papers/volume24/22-1144/22-1144.pdf#page=90).

---
