# Week 2 speaker notes

Single-GPU LLM inference. 21 slides, 30 minutes, followed by 15 minutes of discussion.

The deck and its interactive diagrams work offline. Use arrow keys to change slides, the in-slide buttons to advance examples, and Notes for the explanation. Print shows the completed interactive examples. The batch-size chart starts with an explicitly illustrative model. The optional lab produces real measurements on a CUDA GPU.

## 1. Single-GPU LLM inference

**Suggested time: 0.5 minutes.**

Connect to Week 1: GPU compute can run ahead of its ability to move data. Today applies that idea to a causal dense Transformer on one GPU. The goal is to explain a bottleneck, predict a useful change, and measure it. The displayed date assumes the weekly meeting after September 10. The two questions are authored teaching exercises, not attributed interview reports.


## 2. A prompt produces the first token

**Suggested time: 2 minutes.**

Use Next step twice. The four words are illustrative token labels, not a tokenizer demonstration. Prefill processes all prompt positions with a causal mask and returns logits at the final prompt position. Sampling those logits yields the first generated token. Decode step 1 feeds that generated token, adds its K/V at each layer, and predicts token 2. Decode step 2 feeds token 2 and predicts token 3. The newest sampled token enters the KV cache only when it is subsequently processed. Cache blocks represent per-token K/V across layers, not complete hidden activations. Times are schematic.

- [CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)
- [Berkeley Lecture 18](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_18.pdf#page=9)

## 3. Prefill and decode have different performance limits

**Suggested time: 1.5 minutes.**

Inference is the overall task, with both prefill and decode phases. Compute-bound means arithmetic throughput is the main rate limit; memory-bandwidth-bound means the rate of moving bytes is the main limit. It does not mean running out of memory. Flatten the request and new-token dimensions into M token rows for a dense projection or MLP. Prefill has M=B×S; ordinary decode has M=B because only one new token per request is ready. Long prompts often provide enough reuse to reach compute limits, whereas small decode batches often do not. Short prefills and large batched decode can behave differently, and operators within a pass can have different bottlenecks. The following three slides derive the claim rather than treating it as a rule to memorize.

- [CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)
- [Berkeley Lecture 18](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_18.pdf#page=9)

## 4. Many token rows reuse the same weights

**Suggested time: 2 minutes.**

The difference is matrix-vector versus matrix-matrix work, not different learned weights. With one decode request, a weight contributes to the output for one new token. With a prompt, the same weight contributes to many output rows. Tiled matrix multiplication exploits this reuse in on-chip storage. The entire weight matrix does not need to fit on chip, and real kernels can reload tiles. Only the ideal accounting on the next slides assumes one HBM read for each input. Causal masking restricts attention dependencies, but the tokenwise dense projections and MLP at each layer can still process all available prompt positions together.

- [CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)
- [NVIDIA matmul performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-and-memory-bounds)

## 5. Arithmetic intensity connects reuse to the bottleneck

**Suggested time: 2 minutes.**

M is the number of token rows, D the input dimension, and F the output dimension. There are MF output values, each formed from D multiply-add pairs. Counting one multiply and one add as two FLOPs gives 2MDF. BF16 uses two bytes, giving 2MD bytes for X, 2DF for W, and 2MF for Y in this idealized cold-HBM accounting. Divide to obtain intensity. When M is small relative to D and F, DF dominates the denominator, so I≈M. This approximation is not valid for arbitrarily large M. Compute time floor is FLOPs/C, and HBM time floor is bytes/BW. With overlap, the idealized time is their maximum; their crossover is I=C/BW. These are theoretical resource ceilings/lower bounds, not measured runtimes. Actual traffic, cache behavior, launch latency and hardware utilization matter.

- [NVIDIA matmul performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-and-memory-bounds)
- [CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)

## 6. H100 example: same weights, different bottlenecks

**Suggested time: 2 minutes.**

This is one dense linear layer, not the whole model. Use the exact intensity formula rather than I≈M at M=2048. W contains 67,108,864 values and 134,217,728 BF16 bytes. M=1 requires 134,217,728 FLOPs and 134,250,496 ideal HBM bytes: I=0.999756, compute floor 0.135711 μs, HBM floor 40.074775 μs. M=2048 requires 274,877,906,944 FLOPs and 201,326,592 bytes: I=1365.333, compute floor 277.935194 μs, HBM floor 60.097490 μs. Thus the dominant resource floor switches in this model. 989 TFLOP/s is dense BF16 throughput, not the higher sparsity-assisted headline. The skinny M=1 operation cannot attain that Tensor Core peak, so its 0.14 μs compute floor is emphatically not an achievable benchmark. Peak bandwidth is also a ceiling. The ratio 989/3.35≈295 is a FLOPs-per-byte threshold, not an exact batch-size threshold. Real tiles may reread data, and cached data can reduce HBM traffic.

- [H100 SXM specifications](https://www.nvidia.com/en-sg/data-center/h100/)
- [NVIDIA dense compute peaks](https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput)
- [NVIDIA matmul performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-and-memory-bounds)

## 7. The KV cache saves work on earlier tokens

**Suggested time: 1.5 minutes.**

For a fixed causal model, earlier token representations do not depend on newly generated future tokens, so their K/V can be retained. At each layer we compute the current token’s q, k, and v, append k and v to that layer’s cache, and attend over the relevant past positions plus the current position. Only the current hidden state proceeds through the rest of the network. We normally do not cache old queries for standard next-token decoding. Requests use common read-only model weights but different K/V because their token histories differ.

- [CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)

## 8. Question 1: Why does a longer context slow decode?

**Suggested time: 0.5 minutes.**

Pause before advancing. Invite a distinction between constructing the historical K/V and reading it for a new query. Hold model, hardware, precision, and batch size fixed. We ask why latency might increase, not for an exact proportionality. The rectangles represent historical positions. This is an authored discussion question.

- [CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)

## 9. Solution 1: The cache still has to be read

**Suggested time: 1 minutes.**

The new query changes at each decode step. Its dot products with the historical keys and its weighted sum of historical values must therefore be computed again. KV caching eliminates rebuilding those historical states, not using them. Under ordinary full-context attention, K/V traffic and attention arithmetic grow with context length. Whole-model latency need not double when context doubles, because other costs remain. Sliding-window, sparse, and compressed attention can change this pattern and are outside this example.

- [CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)

## 10. Batching affects weight traffic and KV traffic differently

**Suggested time: 2 minutes.**

For standard multi-head decode attention, consider one query head with its corresponding KV head per request. QK costs approximately 2BSh FLOPs and the weighted-value sum costs another 2BSh. K and V each contain BSh BF16 values, for 4BSh bytes in total. Ignoring the smaller query/output traffic and softmax arithmetic for a long history gives I≈1 FLOP/byte. Batch B appears in both work and KV bytes because different requests have independent histories. Batching does not create the same across-request reuse that exists for model weights. It can still improve attention throughput by improving occupancy, parallelism or effective bandwidth. Shared KV heads, prefix sharing, different precisions and cache residency can change this model; they are outside the current example. Do not generalize the cancellation into a claim that attention cannot be optimized.

- [CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)

## 11. Total throughput and per-user speed

**Suggested time: 1.5 minutes.**

The horizontal axis is generation tokens per second per request, so moving right means a faster user experience. The vertical axis counts output tokens per second across all active requests. Each point is one fixed synchronous batch. The default chart is an explicitly illustrative model with decode-step duration t = 4 + 0.5B milliseconds, where B is batch size. Rates are 1000/t per user and 1000B/t in aggregate. This is not a hardware prediction. Import the accompanying benchmark CSV on slide 19 to replace this chart with local measurements. The Berkeley source inspired the axes; its disaggregated-serving curves are not reused as single-GPU measurements.

- [Berkeley Lecture 19, part 1, p. 10](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_19_1.pdf#page=10)
- [CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)

## 12. Question 2: Which batch meets the user’s target?

**Suggested time: 0.5 minutes.**

This authored question requires comparison, not a lengthy calculation. All values are hypothetical and consistent with the illustrative chart on slide 11 after rounding. Batch 64 maximizes throughput among these rows but only delivers about 28 tokens/s per request. Ask which workload would accept that tradeoff. Offline bulk processing can prioritize aggregate throughput, while interactive use needs an explicit per-request target. These average rates are not p95 serving guarantees.

- [Berkeley Lecture 19, part 1, p. 10](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_19_1.pdf#page=10)

## 13. Solution 2: Batch 32 meets both objectives

**Suggested time: 1 minutes.**

Among the offered configurations, batch 32 delivers the highest aggregate throughput while meeting 40 tokens/s per user. A batch of 64 would be a reasonable choice if maximizing offline throughput were the objective and memory capacity allowed it. Real serving has variable arrivals and lengths, scheduling, queueing, and tail latency. Those complications belong to Week 4. The demonstration on slide 19 will measure repeated fixed-batch decode loops rather than production per-request percentiles.

- [Berkeley Lecture 19, part 1, p. 10](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_19_1.pdf#page=10)
- [Berkeley Lecture 18](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_18.pdf#page=9)

## 14. Ordinary attention materializes large intermediates

**Suggested time: 1.5 minutes.**

A denotes scores and S denotes sequence length. Q, K, V have already been projected; attention proper computes two matrix products separated by a row softmax. The drawing shows dense matrices. Causal attention masks future keys, and optimized kernels can skip corresponding work. The problem is intermediate storage and traffic. Computing more efficiently does not require changing the attention definition. Modern PyTorch SDPA may already choose a fused implementation, so ordinary means the explicit separate-operator baseline.

- [CS336 Lecture 5, pp. 50–54](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=50)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)

## 15. FlashAttention keeps one tile near compute

**Suggested time: 2.5 minutes.**

Use Next step through four states: load the first K/V tile, compute its scores and partial state, load the second K/V tile, and normalize/write the result. This diagram follows one query row with four keys and scalar values. A real kernel handles query blocks and vector-valued outputs. It retains query data and running state on chip while looping over K/V tiles. It discards each score/probability tile after using it. Other query blocks repeat the procedure and may reread K/V. “Write once” refers to this conceptual output block, not a claim about every implementation or every memory transaction. The next slide shows the rescaling that makes softmax correct.

- [CS336 Lecture 5, pp. 50–54](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=50)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)

## 16. Online softmax combines tiles correctly

**Suggested time: 3 minutes.**

The toy log scores are chosen so exponentials are simple. First tile: m=ln2, exp(scores-m)=[0.5,1], l=1.5, u=25. Second tile raises the maximum to ln4, so old contributions must be multiplied by exp(ln2-ln4)=0.5. New weights are [0.75,1]. Updated l=0.5×1.5+0.75+1=2.5. Updated u=0.5×25+0.75×30+1×40=75. Output u/l=30. In general m′=max(m,max tile), α=exp(m−m′), p=exp(tile−m′), l′=αl+sum(p), u′=αu+sum(pv). The real u is a vector. Do not average independently normalized tile outputs equally: their normalization masses differ. Masked keys contribute zero weight.

- [CS336 Lecture 5, pp. 50–54](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=50)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)

## 17. What FlashAttention changes

**Suggested time: 0.5 minutes.**

FlashAttention is an IO-aware algorithm for exact dense attention, not sparse attention or an approximation that drops keys. Different execution and accumulation orders can change floating-point rounding. It retains quadratic attention arithmetic but avoids full quadratic score/probability materialization. Training implementations also recompute intermediates in backward. The largest motivating savings here are for prefill and training. Decode has only one new query per request and continues streaming relevant KV; optimized decode kernels have additional design concerns. Do not promise constant traffic, a universal speedup, or that every input value is fetched exactly once.

- [Berkeley Lecture 2, pp. 44–45](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture2.pdf#page=44)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)

## 18. CUDA Graphs reduce repeated launch overhead

**Suggested time: 1.5 minutes.**

These timelines are schematic, not profiler measurements and not to scale. A CUDA Graph captures GPU operations and dependencies, then replays them with lower repeated host submission cost. The kernels can remain separate, which distinguishes graphs from operator fusion. Standard PyTorch capture needs stable memory addresses, capture-compatible operations, and static shapes/control flow for that captured graph. Input values can change in existing storage. Prefill lengths and growing decode history require deliberate shape/buffer handling or multiple graphs. Capture/compilation warm-up costs are separate from replay latency. Replay cannot eliminate the underlying arithmetic or required HBM traffic.

- [PyTorch CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [CS336 Lecture 6](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)

## 19. Experiment: sweep batch size on one GPU

**Suggested time: 1.5 minutes.**

The lab is a small causal Transformer with random weights. It studies execution behavior, not language quality or production model throughput. It preallocates KV storage and avoids copying a growing cache with torch.cat. The benchmark records synchronized wall-clock time including host dispatch, GPU work, KV writes, and greedy token selection. Prefill includes the first prediction. Decode rates refer to later tokens only. Reported trial percentiles describe repeated trial-average step durations, not serving tail latency. Hardware, software, model shape, precision and lengths are recorded in metadata. The implementation stays fixed, but automatic SDPA backend selection can change with tensor shapes; profile the selected kernels if explaining a performance change. Import the generated CSV to replace slide 11’s illustrative chart for this browser session; no file upload or network request occurs. No measured GPU results are bundled.

- [CS336 Lecture 6](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)
- [PyTorch CUDA timing](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
- [Berkeley Lecture 18](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_18.pdf#page=9)

## 20. A measurement should test a bottleneck hypothesis

**Suggested time: 1 minutes.**

Each row is a hypothesis, not a diagnosis from one symptom. A profiler timeline helps reveal launch gaps and which operators consume time. Both bandwidth-bound and compute-bound kernels can keep a GPU continuously busy. Use operator shapes, FLOP/byte reasoning, achieved compute or memory metrics where available, and controlled comparisons to distinguish limits. PyTorch SDPA is a dispatching API and may select different kernels, so inspect its actual backend rather than assuming FlashAttention. Keep numerical behavior and workload comparable. A lower kernel time does not guarantee lower application latency if another cost dominates.

- [CS336 Lecture 6](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)
- [Berkeley Lecture 18](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_18.pdf#page=9)

## 21. Discussion

**Suggested time: 0.5 minutes.**

The planned talk totals 30 minutes including short question pauses. Use the following 15 minutes for discussion, with five minutes of meeting buffer. Ask participants to choose a metric and workload before suggesting an optimization. Revisit the two questions if time is short. Week 3 covers multi-GPU execution and sharding. Week 4 adds arrivals, continuous batching, paged KV management, prefix reuse, and serving latency objectives. Sources below also appear on the relevant slides.

- [CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)
- [CS336 Lecture 5, pp. 50–54](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=50)
- [Berkeley Lecture 18](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_18.pdf#page=9)
