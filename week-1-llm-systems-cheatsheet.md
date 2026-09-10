# Week 1 - LLM Systems Cheatsheet

Extended reference notes accompanying the 19-slide Week 1 presentation dated September 10, 2026. These notes retain additional detail from the original longer deck. Hardware example: NVIDIA H100 SXM 80 GB. Model example: an explicitly specified dense 8B GQA model, not an exact checkpoint.

## 1. GPU execution & memory hardware

Week 1 reference  |  Concepts first; constants are for H100 SXM 80 GB.

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
| Registers | 65,536 x 32 bits = 256 KiB | Per SM |
| L1 / texture / SMEM | 256 KB combined; up to 228 KB SMEM | Per SM; do not add 256 + 228 |
| L2 cache | 50 MB | Whole GPU |
| HBM3 | 80 GB; peak 3.35 TB/s | Whole GPU; not per SM |
| Dense BF16 Tensor Cores | 989 TFLOP/s | Whole GPU; no structured sparsity |

**Capacity** = bytes that fit; **bandwidth** = bytes/s across a named interface; **latency** = time for an access. HBM bandwidth is not a universal "DRAM-to-SRAM speed." Cache hits can avoid HBM traffic. The hierarchy is not a compulsory route for every access. KB/MB/GB above retain vendor labels; KiB is an explicit binary conversion.

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

| Storage format | Bytes/value | Ideal GB | Ideal GiB |
| --- | --- | --- | --- |
| FP32 | 4 | 32 | 29.80 |
| BF16 or FP16 | 2 | 16 | 14.90 |
| INT8 | 1 | 8 | 7.45 |
| Packed INT4 | 0.5 | 4 | 3.73 |

**BF16 and FP16 occupy the same memory.** FP16 has 1 sign / 5 exponent / 10 fraction bits; BF16 has 1 / 8 / 7. BF16 has a wider exponent range; FP16 has more fraction bits. FP32 has 1 / 8 / 23. Storage, computation and accumulation formats can differ.

**Quantization accounting:** add scale/zero-point metadata and padding; budget unquantized layers at their actual dtypes. W4A16 describes 4-bit weights and 16-bit activations; it does not make KV or optimizer states 4-bit. Tensor views/tied weights can share storage; avoid double-counting.

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

**8B example:** 16 x 8 x 10^9 = **128 GB = 119.21 GiB** of states, before activations and extras. This unsharded layout cannot fit on a 24 GiB device or an 80 GB H100.

**16 bytes/parameter is an assumption, not a universal law.** Gradient dtype, master copies, optimizer and implementation can change it. Autocast alone does not establish this layout. Peak memory depends on which allocations coexist.

**Autograd** retains selected forward values needed by backward. **Activation checkpointing** saves fewer values and recomputes them during backward; it does not directly remove weights, gradients or Adam moments. Parameter count alone cannot determine activation memory. Sharding/offload and adapter training change the budget and must be stated explicitly.

Sources: [CS336: systems and memory](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py), [NVIDIA mixed precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html), [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html).

---

## 3. Inference memory & KV calculations

One teaching model throughout - this is not an exact checkpoint specification.

**Model:** P = 8B; L = 32 layers; 32 query heads; h_KV = 8 KV heads; d_h = 128 values/head; BF16 weights and BF16 KV (2 bytes/value). Base case: B = 8 active requests, S = 4096 cached positions/request. Full attention; no prefix sharing or sliding window.

### 7  What is cached, and when?

**Prefill** processes the prompt and populates K/V; its last prompt position can produce the first output token. **Decode** processes generated tokens iteratively. A sampled token enters the KV cache when the model processes it. S counts processed prompt + generated positions, not just the number of outputs.

**KV cache** stores past keys and values at each attention layer. A new query attends to these cached K/V; old queries are not needed for ordinary autoregressive decode. Caching avoids recomputing the past projections but does not eliminate reading historical K/V for attention.

**GQA (grouped-query attention):** here, 4 query heads share each KV head. Use the **KV-head count** in the storage formula. At the same head dimension and dtype, 32 KV heads (MHA) use 4x the cache of 8 KV heads.

### 8  Derive KV bytes from one token

```text
m_token = 2 x L x h_KV x d_h x b_KV
M_KV = m_token x sum_i(S_i) = m_token x B x S  [equal lengths]
```

The leading 2 counts **K and V**; L counts layers; h_KV counts KV heads; d_h counts values/head; b_KV is bytes/value. Sum the active lengths for unequal requests. This is tensor payload; physical blocks, metadata and unused reservation can add memory.

**m_token** = 2 x 32 x 8 x 128 x 2 = **131,072 B = 128 KiB**, across all layers.
**One request:** 4096 x 128 KiB = 512 MiB = **0.5 GiB**.
**Eight requests:** 8 x 0.5 = **4 GiB**. At 8192 positions each: **8 GiB**.

### 9  Question 1: does it fit in a 24 GiB budget?

```text
M_infer = weights + KV + working activations / workspaces + runtime overhead
```

| Case (8 requests) | Weights | KV | Payload | Left for extras |
| --- | --- | --- | --- | --- |
| A: BF16, S = 4096 | 14.90 | 4.00 | 18.90 | 5.10 |
| B: BF16, S = 8192 | 14.90 | 8.00 | 22.90 | 1.10 |
| C: INT4, S = 8192 | 3.73 | 8.00 | 11.73 | 12.27 |

**All table values are GiB; KV stays BF16 in all three cases.** Ideal weight + KV payload is only part of the budget. C still needs quantization metadata. Remaining capacity is not a fit guarantee: prefill peaks, workspace and allocator behavior matter.

```text
If 3 GiB is reserved for all extras and S = 4096:
B_max = floor[(24 - 14.901161 - 3) / 0.5] = 12 requests
```

This admission estimate assumes the stated reserve is sufficient and lengths stay fixed. Context growth or real allocation peaks can lower the safe limit. Weights are stored once per model replica, not once per request.

Sources: [Hugging Face: KV cache](https://huggingface.co/docs/transformers/main/cache_explanation), [Scaling Book: Transformer accounting](https://jax-ml.github.io/scaling-book/transformers/).

---

## 4. MFU & cache metrics

Define the numerator, denominator and measurement window before calculating.

### 10  Model FLOPs utilization (MFU)

**FLOP** = one floating-point operation; **FLOP/s** = a rate. A multiply-add conventionally counts as 2 FLOPs. MFU compares useful model arithmetic per second with the matching theoretical peak of the GPU group.

```text
MFU = useful model FLOPs / (N_GPU x F_peak x elapsed seconds)
Dense training approximation: MFU = (6 x P x R) / (N_GPU x F_peak)
```

**R** is the measured **global** training token rate, in tokens/s; F_peak is FLOP/s **per GPU**. Equivalently use 6P x global tokens in a measured step as the numerator and include step time in the denominator.

**Why 6P?** Dense weight matrix operations cost roughly 2P FLOPs/token in forward + 4P in backward. This is not three equal forward/backward/optimizer passes. Attention and architecture details need a more precise model, especially for long context. For inference, use forward-only work (roughly 2P plus attention).

### 11  Question 2: 8B training on 8 H100 SXM GPUs

**Given:** 60,000 global training tokens/s; dense BF16 Tensor Core peak = 989 x 10^12 FLOP/s per GPU. Assume distributed model states fit; ignore attention in this estimate.
**Useful rate:** 6 x 8 x 10^9 x 60,000 = 2.880 x 10^15 FLOP/s.
**Group peak:** 8 x 989 x 10^12 = 7.912 x 10^15 FLOP/s.
**MFU = 2.880 / 7.912 = 36.4%.**

**MFU checks:** match precision and dense/sparse peak; do not multiply global throughput by GPU count again. The usual MFU numerator excludes extra checkpoint recomputation. Total MoE parameters cannot be inserted into the dense formula without accounting for active computation. A measured token rate or step time is required.

**GPU-Util is different:** it measures the fraction of a sampling interval with one or more kernels running, not useful FLOPs / peak. 95% GPU-Util can coexist with 36.4% MFU. Low decode MFU can reflect a bandwidth limit; MFU alone does not diagnose the bottleneck or service quality.

### 12  Three different meanings of "cache"

| Concept | What is reused? | Metric / accounting |
| --- | --- | --- |
| Per-request KV cache | Past K/V within autoregressive generation | Payload from token count and architecture |
| Prefix cache | K/V for an identical token prefix across requests | Reused prompt tokens / queried prompt tokens |
| Hardware L1 / L2 cache | Memory data handled by GPU cache hardware | Cache accesses/hits, separate from prefix hits |

```text
Token hit rate = delta(prefix_cache_hits) / delta(prefix_cache_queries)
Request hit rate = requests with any prefix hit / queried requests
```

**Use counter deltas from the same time window.** vLLM prefix hit/query counters count tokens; verify the deployed version. Request hit rate uses requests. **KV pool occupancy** is used blocks/capacity, a third quantity. GPU type and model size alone cannot determine a prefix hit rate.

**Reuse requires** matching token IDs and model/cache identity, resident data and supported block granularity. Arrival order, warm/cold state, eviction and routing matter. Prefix caching saves prefill work; 60% hits does not mean 60% less end-to-end latency.

Sources: [PaLM: MFU, Appendix B](https://jmlr.org/papers/volume24/22-1144/22-1144.pdf#page=90), [NVIDIA dense compute peaks](https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput), [NVIDIA GPU utilization definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization), [vLLM cache metrics](https://docs.vllm.ai/en/stable/usage/metrics/#general-metrics), [vLLM prefix-cache design](https://docs.vllm.ai/en/stable/design/prefix_caching/), [vLLM prefix-cache limits](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/#limits).

---

## 5. Worked cache & bandwidth questions

Count tokens for reuse; count bytes for traffic. State the physical sharing assumptions.

### 13  Question 3: cold prefix cache, eight prompts

**Setup:** eight requests each have 4096 prompt tokens: an identical 3072-token prefix and a distinct 1024-token suffix (different from its first token). Prefills finish sequentially. The first lookup is cold; no eviction occurs. Retain all eight request states and exclude decode. Same model/cache identity; 16-token blocks; matching blocks are stored once and shared by reference.

| Quantity | Calculation | Result |
| --- | --- | --- |
| Token hits | 0 + 7 x 3072 | 21,504 tokens |
| Token queries | 8 x 4096 | 32,768 tokens |
| Token hit rate | 21,504 / 32,768 | 65.625% |
| Request hit rate | 7 / 8 | 87.5% |
| Unique cached positions | 3072 + 8 x 1024 | 11,264 |
| Shared KV payload | 11,264 x 128 KiB | 1.375 GiB |
| Without physical sharing | 8 x 4096 x 128 KiB | 4 GiB |

**Warm-cache variant:** if the prefix is resident before all eight lookups, the token hit rate is 3072 / 4096 = **75%**, and the request hit rate is **100%**. The final unique payload is still **1.375 GiB** under the same sharing assumptions.

The shared payload excludes metadata, unused reserved blocks and later decode growth. Identical prompt text alone is insufficient if tokenization or preceding tokens differ. Hit rate describes reuse events; physical KV sharing describes stored bytes. One does not uniquely determine the other.

### 14  Optional challenge: fusion and HBM traffic

**Fusion** combines operations into one kernel while preserving required numerical semantics. Compare `t = x * 2; y = t + 1` with a fused kernel. Let N = 2^26 BF16 elements. Each tensor contains N x 2 bytes = **128 MiB**.

| Implementation | Modeled full-tensor passes | HBM traffic |
| --- | --- | --- |
| Two kernels | Read x; write t; read t; write y | 4 x 128 = 512 MiB |
| One fused kernel | Read x; write y | 2 x 128 = 256 MiB |

```text
Traffic-model time bound = modeled HBM bytes / peak HBM bytes per second
512 x 2^20 / (3.35 x 10^12) = 160.26 microseconds
256 x 2^20 / (3.35 x 10^12) =  80.13 microseconds
```

**Assumptions:** every listed read/write reaches HBM; no cache benefit or extra transactions; use the H100 SXM 3.35 TB/s peak. These are conditional bounds for the specified byte counts, not measured kernel times. Half the modeled traffic does not guarantee twice the real speed.

**Where did t go?** Small per-thread intermediate values can stay in registers. The whole 128 MiB tensor does not need to fit in one SM. Cache hits, launch overhead, achieved bandwidth, computation and register pressure can change the result.

**Tiling** is a related reuse technique: load a small matrix tile into SMEM, reuse it cooperatively, and keep working accumulators in registers. It reduces repeated distant accesses; it does not require the entire matrix to fit on chip.

Sources: [vLLM prefix-cache design](https://docs.vllm.ai/en/stable/design/prefix_caching/), [vLLM cache metrics](https://docs.vllm.ai/en/stable/usage/metrics/#general-metrics), [vLLM prefix-cache limits](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/#limits), [CUDA bandwidth guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#bandwidth), [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/), [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html).

---

## 6. Measurement & memory tradeoffs

Use estimates to explain the budget, then validate the workload that will actually run.

### 15  PyTorch memory measurements

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
peak_gib = torch.cuda.max_memory_allocated() / 2**30
```

The workload function is a placeholder for your prefill/decode code. Resetting statistics does not remove baseline tensors. Allocated + reserved double-counts memory. CUDA/runtime or library allocations may sit outside PyTorch tracking, so the peak is not the whole process footprint. Measure initialization/graph-capture peaks separately if relevant.

**Timing:** a CPU timer around an asynchronous launch may only measure submission. Use CUDA events or synchronize the boundaries of a warmed-up measurement. Measure representative sequence lengths, concurrency and a complete training step when estimating training peak.

### 16  What changes when you change the workload?

| Change (hold other choices fixed) | Direct effect |
| --- | --- |
| Double B or cached S | Double unshared KV payload; weights per replica stay fixed |
| BF16 weights -> INT4 weights | Ideal weight bytes / 4; KV unchanged unless separately changed |
| BF16 KV -> 1-byte KV | Ideal KV bytes / 2; weights unchanged; support/metadata matter |
| 32 KV heads -> 8 KV heads | KV bytes / 4 at fixed head dimension and KV dtype |
| Activation checkpointing | Fewer saved activations; more recomputation |
| Prefix block sharing | Fewer unique KV blocks when prefixes match and remain cached |
| Sharding / offload / adapters | Changes state placement or trainable subset; redo the budget |

Sources: [PyTorch memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management), [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution), [CS336: systems and memory](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py), [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html).

---

## 7. Quick lookup & interview checklist

The one-page recap  |  Keep units, scope and assumptions visible.

### Unit conversions

| Quantity | Meaning / conversion |
| --- | --- |
| P; b_w | P = unique stored parameter count; b_w = bytes per weight |
| Bit vs byte | 8 bits = 1 byte; a packed 4-bit value ideally uses 0.5 byte |
| Decimal | 1 GB = 10^9 bytes; 1 TB/s = 10^12 bytes/s |
| Binary | 1 KiB = 2^10 B; 1 MiB = 2^20 B; 1 GiB = 2^30 B |
| Convert | GiB = bytes / 2^30; seconds = bytes / (bytes per second) |

### Core equations

```text
Weights:    M_weights = P x b_w
Training:   M_states = 16P bytes  [only the stated Adam layout]
KV/token:   m_token = 2 x L x h_KV x d_h x b_KV
KV total:   M_KV = m_token x sum_i(S_i)  [no physical prefix sharing]
Fit:        weights + KV + all other live allocations <= budget
Dense training MFU (approx): (6 x P x global tokens/s) / (N_GPU x F_peak)
Token hits: reused prompt tokens / queried prompt tokens
Time bound: modeled transferred bytes / peak bytes per second
```

**Symbols:** P = stored parameter count; b_w / b_KV = bytes per weight / KV value; L = layers; h_KV = KV heads; d_h = values/head; S_i = currently cached positions for request i; N_GPU = GPU count; F_peak = matching FLOP/s per GPU. Equations count ideal payload unless stated otherwise.

### 17  A reliable interview answer in five steps

**1. State the target.** Weight storage, inference peak, training peak, token throughput or hit rate?
**2. State assumptions.** Exact GPU variant, units, model architecture, precision for each state, concurrency/context; cache warmness and sharing if relevant.
**3. Write the equation with units.** Separate hardware constants, chosen inputs and measured quantities.
**4. Calculate the known payload.** Add a stated reserve or identify what needs measurement. Round only at the end.
**5. Explain the limit.** A payload fit is not a peak-memory guarantee; a theoretical traffic time is not a benchmark; a utilization percentage is not a latency prediction.

**Numbers to remember for this teaching model:** BF16 weights **14.90 GiB**; assumed Adam states **119.21 GiB**; KV **128 KiB/token**; 8 x 4096 BF16 KV **4 GiB**; base inference payload **18.90 GiB**; example MFU **36.4%**; cold prefix hits **65.625%**.

**Scope:** covers the Week 1 talk and additional reference material for discussion. Worked questions use authored teaching numbers, not verified verbatim frontier-lab interview questions. Each section links to primary documentation; hardware constants and implementation details must be checked for other variants or software versions.

Sources: [CS336: systems and memory](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py), [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html), [PaLM: MFU, Appendix B](https://jmlr.org/papers/volume24/22-1144/22-1144.pdf#page=90), [vLLM cache metrics](https://docs.vllm.ai/en/stable/usage/metrics/#general-metrics).

---
