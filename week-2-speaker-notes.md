# Week 2 speaker notes

LLM inference performance. 31 slides, 35.25 minutes of suggested full content. A roughly 32.25-minute route skips slides 8, 27, 28. Reserve 15 minutes for discussion.

The HTML deck works offline. The online-softmax derivation remains on the slides. The generation and block-table diagrams have step controls. The DistServe plot is an attributed published experiment; the H100 bars are theoretical bounds, and the remaining diagrams are schematic.

## 1. LLM inference performance

**Suggested time: 0.5 minutes.**

Audience: Transformer/PyTorch familiarity with little GPU systems background. Follow the causal chain from latency metrics and generation to KV reuse, matrix shapes, traffic, attention IO, and serving schedules. The complete deck has 35.25 minutes of suggested content. A roughly 32.25-minute route skips slides 8, 27, 28; it retains arithmetic-intensity prerequisites, the visible online-softmax derivation, and the mixed-batch setup. These are planning estimates, not rehearsed timings. Reserve 15 minutes for discussion. Slides 30–31 are a take-home GEMM assignment and its reference solution, outside the lecture timing. Questions are authored exercises, not attributed company interview reports. The only empirical figure is clearly attributed to DistServe; other diagrams are schematic and H100 bars are theoretical resource bounds.


## 2. Latency and throughput measure different things

**Suggested time: 1 minutes.**

Start with user-visible behavior before optimization names. Arrival is the chosen service boundary. TTFT runs to the first delivered output token and includes queueing, tokenization, prefill, sampling and delivery overhead. ITL is an interval between successive delivered tokens; a per-request average of these intervals is often called TPOT. Aggregate output throughput counts tokens across requests in a time window. The drawn distances have no numeric time scale. Rates in the fixed-batch lab exclude the first output token and are not production latency percentiles.

- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)
- [Berkeley L18, PDF pp. 9–16](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf#page=9)

## 3. A prompt produces the first token

**Suggested time: 1.5 minutes.**

Use Next step twice. The four words are illustrative token labels, not a tokenizer demonstration. Prefill processes all prompt positions with a causal mask and returns logits at the final prompt position. Sampling those logits yields the first generated token. Decode step 1 feeds that generated token, adds its K/V at each layer, and predicts token 2. Decode step 2 feeds token 2 and predicts token 3. The newest sampled token enters the KV cache only when it is subsequently processed. Cache blocks represent per-token K/V across layers, not complete hidden activations. Times are schematic.

- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)
- [Berkeley L18, PDF pp. 9–16](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf#page=9)

## 4. The KV cache saves work on earlier tokens

**Suggested time: 1 minutes.**

For a fixed causal model, earlier token representations do not depend on newly generated future tokens, so their K/V can be retained. At each layer we compute the current token’s q, k, and v, append k and v to that layer’s cache, and attend over the relevant past positions plus the current position. Only the current hidden state proceeds through the rest of the network. We normally do not cache old queries for standard next-token decoding. Requests use common read-only model weights but different K/V because their token histories differ.

- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)

## 5. New token rows are only part of the workload

**Suggested time: 1 minutes.**

M is a count of newly processed token positions, not bytes transferred. At one layer a dense projection or MLP acts on M hidden-state vectors, each with many scalar values. In an uncached prefill every prompt position must be processed to build its per-layer K/V and support later causal positions, even when only the last position produces the next-token prediction. For B=2 and S=4 this is eight positions. In ordinary autoregressive decode, each request feeds its most recently generated token, so two requests supply two new positions. Earlier positions do not need to pass through the projections and MLP again because their K/V has been retained. However each new query still attends to the relevant cached keys and values plus the current position. Weight reads also remain: model weights already reside in GPU HBM, but arithmetic needs to load weight tiles into on-chip storage. This is separate from CPU-to-GPU transfer of token IDs. New hidden-state traffic, weight traffic, historical KV reads, new KV writes, and other intermediates all contribute to physical memory traffic. The cache is not a promise that all historical KV fits on chip. Prefix-cache hits, speculative multi-token decoding, and ragged prompt batches are outside this simple shape comparison.

- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)
- [Berkeley L18, PDF pp. 9–16](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf#page=9)

## 6. Many token rows reuse the same weights

**Suggested time: 1 minutes.**

The difference is matrix-vector versus matrix-matrix work, not different learned weights. With one decode request, a weight contributes to the output for one new token. With a prompt, the same weight contributes to many output rows. Tiled matrix multiplication exploits this reuse in on-chip storage. The entire weight matrix does not need to fit on chip, and real kernels can reload tiles. Only the ideal accounting on the next slides assumes one HBM read for each input. Causal masking restricts attention dependencies, but the tokenwise dense projections and MLP at each layer can still process all available prompt positions together.

- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)
- [NVIDIA matmul performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-and-memory-bounds)

## 7. Arithmetic intensity connects reuse to the bottleneck

**Suggested time: 1.5 minutes.**

M is the number of token rows, D the input dimension, and F the output dimension. There are MF output values, each formed from D multiply-add pairs. Counting one multiply and one add as two FLOPs gives 2MDF. BF16 uses two bytes, giving 2MD bytes for X, 2DF for W, and 2MF for Y in this idealized cold-HBM accounting. Divide to obtain intensity. When M is small relative to D and F, DF dominates the denominator, so I≈M. This approximation is not valid for arbitrarily large M. Compute time floor is FLOPs/C, and HBM time floor is bytes/BW. With overlap, the idealized time is their maximum; their crossover is I=C/BW. These are theoretical resource ceilings/lower bounds, not measured runtimes. Actual traffic, cache behavior, launch latency and hardware utilization matter.

- [NVIDIA matmul performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-and-memory-bounds)
- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)

## 8. H100 example: same weights, different bottlenecks

**Suggested time: 1.5 minutes.**

This is one dense linear layer, not the whole model. Use the exact intensity formula rather than I≈M at M=2048. W contains 67,108,864 values and 134,217,728 BF16 bytes. M=1 requires 134,217,728 FLOPs and 134,250,496 ideal HBM bytes: I=0.999756, compute floor 0.135711 μs, HBM floor 40.074775 μs. M=2048 requires 274,877,906,944 FLOPs and 201,326,592 bytes: I=1365.333, compute floor 277.935194 μs, HBM floor 60.097490 μs. Thus the dominant resource floor switches in this model. 989 TFLOP/s is dense BF16 throughput, not the higher sparsity-assisted headline. The skinny M=1 operation cannot attain that Tensor Core peak, so its 0.14 μs compute floor is emphatically not an achievable benchmark. Peak bandwidth is also a ceiling. The ratio 989/3.35≈295 is a FLOPs-per-byte threshold, not an exact batch-size threshold. Real tiles may reread data, and cached data can reduce HBM traffic.

- [H100 SXM specifications](https://www.nvidia.com/en-sg/data-center/h100/)
- [NVIDIA dense compute peaks](https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput)
- [NVIDIA matmul performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-and-memory-bounds)

## 9. KV memory limits how many requests fit

**Suggested time: 1 minutes.**

The formula counts both K and V for a batch of B sequences, each with S cached tokens, across L layers. d_KV is the number of KV heads times the head dimension: the total K dimension for one token in one layer, and equally the total V dimension. It need not equal the full model hidden dimension. p is the storage size in bytes per element. In the example B=16, S=4096, L=32, d_KV=32×128=4096, and p=2 for BF16. Direct substitution gives 2×16×4096×32×4096×2=34359738368 bytes, or approximately 34.4 GB. The first factor 2 counts K and V; the final factor 2 is BF16 storage per element. S counts cached prompt and generated positions already processed, not the maximum permitted sequence length or only the newly generated tokens. The newest sampled token enters the cache when it is subsequently processed. For unequal lengths, replace B×S with the total cached tokens across the batch. The slide assumes uniform layer widths, equal K and V dimensions, and no prefix sharing between requests. This is stored KV payload, not per-step memory traffic or total GPU allocation; unused cache capacity, metadata, model weights and workspace require additional memory.

- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)
- [Berkeley L18, PDF pp. 9–16](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf#page=9)
- [Scaling Book: KV memory](https://jax-ml.github.io/scaling-book/inference/)

## 10. Batching affects weight traffic and KV traffic differently

**Suggested time: 1.5 minutes.**

The left side uses the previously derived ideal dense-layer intensity I≈M with M=B in ordinary one-token decode and weight-dominated traffic. The right side counts one query head and its corresponding KV head in ordinary multi-head attention. q[1,d] times K transpose[d,S] forms S scores at approximately 2Sd FLOPs. The normalized row p[1,S] times V[S,d] costs another 2Sd FLOPs. Reading both S×d KV arrays in BF16 costs 2 arrays×Sd elements×2 bytes=4Sd bytes. For a sufficiently long history, ignoring query/output traffic, small intermediates and softmax arithmetic yields approximately one FLOP per byte. This is not a claim about prefill or the full model. Multiply both work and KV traffic by B for B independent histories and the ratio stays the same. Batching can still improve occupancy, parallelism and achieved bandwidth. Grouped query heads, shared prefixes, alternative precisions, cache residency and repeated reads change the traffic model; the slide does not assume them. The pedagogical distinction is shared model weights versus separate request histories, not that attention is impossible to optimize.

- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)

## 11. Question 1: Why did a larger decode batch stop helping?

**Suggested time: 1 minutes.**

Ask for a diagnosis with competing explanations, not a memorized prefill/decode label. This is an authored hypothetical case, not measured data or a reported company interview question. All batches fit without offloading; model, precision and implementation remain fixed. Sequence length is held fixed within a comparison. With ordinary one-token decode, a B-request batch produces B output tokens per step. Little throughput improvement after doubling B means the step duration has risen close to twofold. That can happen because per-request KV traffic scales with B, because arithmetic saturates compute throughput, or because per-request host/dispatch work grows with B. Fixed per-step launch overhead alone would usually allow throughput to grow with B rather than explain a plateau. The observation alone cannot distinguish these. An ideal weight-read budget is amortized over the batch, while unrelated requests generally have separate KV histories. Ask what evidence would disprove the proposed diagnosis before selecting an optimization.

- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)
- [NVIDIA matmul performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-and-memory-bounds)

## 12. Solution 1: A throughput plateau does not identify the limit

**Suggested time: 1 minutes.**

weight_bytes is the number of bytes of weights ideally fetched once during one complete decode step, and kv_bytes is the required K/V traffic for one request across layers at the chosen cached length S. These names distinguish byte counts from W and K matrices. Both quantities are traffic under stated assumptions, not reserved capacity. Ideal weight-plus-KV reads are weight_bytes+B×kv_bytes. A bandwidth-limited step takes approximately those bytes divided by effective bandwidth; dividing the B emitted tokens by this duration gives bandwidth/(weight_bytes/B+kv_bytes). If the KV term dominates, doubling B approximately doubles both duration and output tokens, so aggregate throughput can stay flat while bandwidth remains the limit. This is one hypothesis, not an inference from a plateau alone. Compute can instead dominate. Profile time attributed to dense layers versus attention, measured traffic and achieved bandwidth, relevant compute throughput, and host/GPU idle gaps. A large byte count alone does not prove bandwidth saturation; distinguish achievable bandwidth ceilings from memory-latency or parallelism problems. Batch-independent launch overhead is generally amortized as B grows; per-request host work that grows with B can instead create a rate limit. Vary B at fixed S, then S at fixed B, keeping model, GPU, dtype and timing boundaries fixed. Both attention arithmetic and KV traffic increase with S, so a length sweep alone is not proof. Compare counters and kernel timings together. Caches, rereads, ragged lengths, shared prefixes and extra intermediates can change this ideal traffic model. Faster GEMMs help only if their execution is a material part of the measured bottleneck.

- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)
- [NVIDIA matmul performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-and-memory-bounds)

## 13. Continuous batching replaces finished requests

**Suggested time: 1 minutes.**

The drawing isolates iteration-level membership changes. Each column is one iteration, not a fixed number of milliseconds, so it does not claim a numeric speedup. A needs two more decode tokens, B five, and C three. Under the illustrated static policy, the completed A slot cannot be reused until B finishes. Under continuous membership C takes the slot at the next boundary. Assume C has completed prefill elsewhere/earlier; real scheduling must also budget that work. A vLLM-style scheduler also checks available KV blocks and the step token budget. Different context lengths require appropriate ragged/paged attention handling. Fixed slot count two is purely illustrative.

- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)
- [Orca, §4.1](https://www.usenix.org/system/files/osdi22-yu.pdf#page=5)
- [Berkeley L18, PDF pp. 9–16](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf#page=9)

## 14. PagedAttention reduces wasted KV capacity

**Suggested time: 1.5 minutes.**

Compare an illustrative strategy reserving space for up to 12 tokens per request against on-demand fixed-size KV blocks. Requests A, B, C currently have 6, 3, 5 cached tokens, respectively: 14 used slots. Fixed reservations allocate 36 slots and leave 22 unused. With 4-token blocks, the requests use 2, 1, 2 blocks: 20 allocated slots with 6 unused tail slots. The unused tail of a block can be filled as that request grows. PagedAttention also avoids requiring each request’s physical blocks to be contiguous, reducing external fragmentation; the next slide explains the lookup. The reservation baseline is a teaching example, not a claim that every contiguous allocator always reserves a fixed maximum. Actual blocks hold vector-valued K/V at model layers, and metadata has a small additional cost. Four tokens per block is a toy choice, not a recommended deployment setting. Paging does not compress K/V, move it automatically to CPU memory, or remove the reads needed for attention. More efficient capacity use permits more concurrent requests when KV capacity is the limiting resource; throughput gains depend on the workload and kernels.

- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [vLLM: PagedAttention](https://vllm-project.github.io/2023/06/20/vllm.html)

## 15. PagedAttention: logical blocks and physical storage

**Suggested time: 1.5 minutes.**

Use Next step four times. The logical blocks and physical pool are two views of the same stored K/V, not copies. The block table stores a mapping. One physical block holds K/V for four cached token positions. Request C keeps physical blocks 0 and 2 throughout. Initially A has 6 cached tokens: logical blocks 0 and 1 map to physical blocks 4 and 1. Read table [4,1]: P4 supplies A1–A4, followed by P1 supplying A5–A6; the two unused slots in P1 are ignored. In particular, logical token position 5 lies in logical block 1, whose table entry points to the first slot of P1. A grows to 8 tokens by filling its existing tail block; no old K/V moves. The ninth cached token allocates another free block, physical block 5, with three tail slots unused. A then finishes and releases its unshared blocks. Request B arrives with 3 cached tokens and reuses physical block 4; old A contents are no longer valid. The example deliberately disables prefix retention and sharing; reference-counted shared or cached blocks need different release rules. Token labels count processed positions, not tokens merely sampled. A logical block preserves token order despite nonconsecutive physical addresses. The attention kernel uses the table and sequence length to fetch valid historical K/V, never treating unused slots as valid keys. A storage block here is not a CUDA thread block. PagedAttention can coexist with tiled attention kernels: it addresses KV allocation and lookup, while FlashAttention addresses attention computation and intermediate IO.

- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [vLLM: PagedAttention](https://vllm-project.github.io/2023/06/20/vllm.html)

## 16. Ordinary attention materializes large intermediates

**Suggested time: 1 minutes.**

Explicitly return from the preceding single-query decode examples to full-prompt prefill. One request and one head have S query rows and S key rows, giving S-by-S scores and probabilities. For ordinary decode there is one new query, so the corresponding row has 1-by-S scores. The 4-by-4 illustration labels queries down the rows and keys across columns; future positions are masked. The last query row is blue and stays the chosen row in the next slides. A denotes scores. Q, K, V have already been projected; attention computes two matrix products separated by a row softmax. In the explicit separate-operator baseline these score and probability arrays are written and reread in HBM. They are temporary intermediates, not the persistent per-request K/V stored for future generation. Modern PyTorch attention APIs may already choose fused implementations; ordinary here means this explicit baseline. Causal kernels can skip masked work. The mathematical attention result need not change.

- [CS336 Lecture 5, pp. 52–54](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=52)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)

## 17. Attention is a weighted average

**Suggested time: 1.25 minutes.**

Keep the final blue query row from the preceding 4-token prefill matrix. Its four valid keys are split into two illustrative compute tiles. The same argument applies to any chosen query and its valid keys, including a decode query. The denominator must normalize over the whole valid row, not separately over each tile. Substituting p_j=exp(s_j)/sum_k exp(s_k) into sum_j p_j v_j gives the ratio on screen. The scale sqrt(d) is already included in the scores. The denominator is one scalar; the numerator and output are vectors of the value dimension. Both unnormalized sums are additive across disjoint tiles, so retain and add each tile’s contributions, then divide once. Independent per-tile outputs cannot simply be averaged because different tiles have different total unnormalized weights. The two tiles shown are kernel compute tiles, not the storage-allocation blocks on the preceding PagedAttention slide. This establishes why streaming is mathematically possible; the next two slides make the accumulation numerically stable.

- [CS336 Lecture 5, pp. 52–54](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=52)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)

## 18. Stable streaming needs three running values

**Suggested time: 1.5 minutes.**

The equations describe the state for the same fixed query after processing at least one valid key. Every sum on this page covers the keys processed so far. The ratio u/l would be attention over that partial set; it becomes the final output only after all valid keys have been included. m is the maximum of the scores processed so far. On this reference scale every exp(s_j−m) is at most one, avoiding overflow of positive exponentials. Multiplying both sums by the same positive factor does not alter their ratio. The state uses two scalars and one vector per query row; it does not grow with the number of processed keys. A real query tile has one such state per row. Online means an incremental scan within a kernel, not online learning or a server processing internet requests. The remaining obstacle is that m is not known globally until all tiles are visited.

- [Online normalizer, 2018](https://arxiv.org/abs/1805.02867)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)

## 19. A new maximum rescales the old contributions

**Suggested time: 2 minutes.**

The identity at the top is the reason this works, not an extra heuristic. For every old score s, exp(s−m′)=exp(s−m) exp(m−m′). Linearity lets us multiply the old denominator and numerator by alpha without retaining old scores or values. The sums in the update range only over the new tile. If the maximum does not increase, alpha is one. Initialization is m=−infinity, l=0, u=0; the first tile with a finite valid maximum gives alpha=0. All-masked tiles require special handling to avoid exp(−infinity−(−infinity)), and are skipped in this introductory derivation. Each state should be updated together, using old l,u in the right-hand sides. The numerator is a vector in real attention.

- [Online normalizer, 2018](https://arxiv.org/abs/1805.02867)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)

## 20. FlashAttention streams tiles through the running state

**Suggested time: 1.5 minutes.**

This reconnects the algebra to the GPU memory hierarchy. One query tile and its state are kept on chip, while K/V tiles are streamed from HBM. The score/probability tile is temporary; full S-by-S arrays are never required in HBM. The running state is per query row, and u is a vector. A real kernel partitions storage across registers and shared memory and uses many threads. Do not imply the whole sequence fits on chip, or that all K/V values are read exactly once for the whole attention layer. Loop orders and implementations differ; this is an explanatory query-tile organization consistent with FlashAttention-2-style processing. Causal masks restrict each row’s valid keys.

- [CS336 Lecture 5, pp. 52–54](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=52)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)

## 21. Question 2: Copying paged K/V at every decode step

**Suggested time: 1.5 minutes.**

Authored diagnostic exercise grounded in the PagedAttention kernel design, not a verified interview question from any company. The motivation is to connect paged storage to the way an attention kernel reads it. Explain the copying premise before asking the questions: torch.cat allocates new storage and copies the page contents. Reuse slide 15 if helpful: table=[4,1], block_size=4, T=6. P4 holds tokens 1–4 and P1 holds tokens 5–6 plus two unused positions. Concatenation copies all eight positions, then [:T] keeps the first six for attention; slicing does not undo the copy or allocation. The code is executable for a one-head float tensor setting with the imports and inputs provided. T includes the current processed token. The current query may attend to all T keys, so no future-position mask is needed here. The block table must preserve token order and every view in the list has shape block_size×d. Ask participants to separate numerical correctness, persistent allocation, temporary allocations, and bandwidth. The question targets the O(Td) K/V reconstruction. This unfused reference also creates O(T) scores and probabilities; that is a separate cost. No timing or speedup is supplied because they depend on the implementation and workload.

- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)

## 22. Solution 2: Read K/V directly from the pages

**Suggested time: 1.5 minutes.**

The prototype can be mathematically correct. Page allocation already reduces some reservation waste, but every concatenate materializes the requested history again, adding O(Td) temporary storage and extra reads/writes per layer and step. Dense K and V can coexist with the page pool. A page-aware kernel translates token ranges using the block table and consumes page contents directly, retaining online softmax state as necessary. It does not need to recreate a whole contiguous K/V history in HBM. Appending new tokens updates the current tail/new page; old cache contents need not move. The benefit is not a promise of faster single-request attention. Compare the same model, dtype, lengths and batch first, then study admitted concurrency under a fixed memory budget. For numerical checking compare against the dense reference with an appropriate floating-point tolerance.

- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [vLLM: PagedAttention](https://vllm-project.github.io/2023/06/20/vllm.html)

## 23. One serving iteration can mix prefill and decode

**Suggested time: 1.5 minutes.**

This is a new serving example, distinct from the earlier continuous-batching illustration where C had already completed prefill. A and B each provide their most recently generated token; C provides all eight prompt positions. The batch therefore has three requests and ten newly processed positions. Tokenwise projections and MLP operations can pack these positions for shared weight reuse. Attention must retain request boundaries and each request’s valid causal context, through metadata and a suitable kernel; the requests must not attend to one another. Packing does not imply one kernel, or that every operation executes simultaneously. The new input row count excludes weight reads and historical KV reads. At the end A and B each have their next token and C has its first answer token. The scheduler repeats this selection at iteration boundaries. This schematic execution model provides the common setup for the DistServe interference plot, chunked prefill and phase disaggregation.

- [Orca: selective batching, Fig. 5](https://www.usenix.org/system/files/osdi22-yu.pdf#page=7)
- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)

## 24. A prefill can delay an ongoing decode batch

**Suggested time: 1.5 minutes.**

Figure reproduced from DistServe Figure 2, physical PDF p5, proceedings p196, with original axes, legends and annotations retained. Section 2.3 and the caption compare a decode batch with that batch plus one prefill request: the x-axis count refers to ongoing decode requests, with one extra prefill for the blue curve. Orange is decode-only batch execution time; blue is mixed-batch execution time; dashed is prefill-alone time. The y-axis is milliseconds for execution, not tokens/s or end-to-end latency including serving queues. At fixed decode count, blue minus orange is the extra wait until ongoing decode outputs complete. Blue minus dashed is how much later the prefill completes than when run alone. These are different reference comparisons, not additive components of mixed runtime or proof that a fixed individual kernel became slower. The extra work itself can increase duration; the figure does not isolate intrinsic execution inefficiency. The larger gap in the 1024-token panel illustrates why long prompt work can interrupt token delivery. The panels use different y ranges. The caption identifies a 13B LLM but does not supply a complete hardware/backend configuration for this specific figure, so we do not attribute it to a particular GPU. This is the authors’ published workload, not our experiment and not a benchmark of modern chunked-prefill scheduling.

- [DistServe, Fig. 2 (OSDI 2024)](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=5)
- [Berkeley L18, PDF pp. 39–41](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf#page=39)

## 25. Chunked prefill limits work between decode steps

**Suggested time: 1 minutes.**

This continues the A/B/C serving example. Each row is one scheduled model iteration on the shared GPU, not a separate GPU and not a measured equal-duration time interval. A and B each advance one decode step in every row. The first iteration processes C positions 1–4 and stores their per-layer KV. The next processes positions 5–8, attending to previous cached positions plus the causal prefix of its new chunk. Only the last prompt position is used to predict the first answer token. The third iteration processes that answer token, grows C’s cache to nine positions and predicts its second token. Token chunking does not create independent requests or reset context. A scheduler can reserve a token budget for ongoing decodes and fill the remainder with prefill work; four prompt tokens per chunk is only a toy choice. Too-small chunks increase launch/scheduling overhead and can reduce GEMM efficiency; larger chunks can prolong decode iterations. Chunking and disaggregation address different resource decisions and can coexist. No equal runtime or numeric speedup is implied.

- [Berkeley L18, PDF p. 18](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf#page=18)
- [vLLM: chunked prefill](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill)

## 26. Prefill–decode disaggregation

**Suggested time: 1.5 minutes.**

PD means prefill–decode disaggregation. Prefill and decode instances use distinct GPU resources and can choose their resource allocation and batch or parallelism strategies independently. They need compatible access to the model weights, often as separate replicas or sharded replicas. For a request, the prefill side builds K/V across layers and passes that state plus relevant request metadata to the decode side. The first sampled token can be produced by prefill and handed over as well; the figure focuses on KV rather than the exact API ownership of first-token delivery. Decode then continues autoregressive generation without recomputing the prompt. A transfer-time lower bound is payload bytes divided by effective link bandwidth; setup, contention, layout conversion and additional queues can add delay. Layerwise transfer may overlap some communication with prefill, so the full transfer duration is not necessarily extra exposed latency. Benefits include reducing prefill-induced tail ITL and independent TTFT/ITL tuning. Gains in goodput—the request rate that meets latency targets—depend on the workload and placement; disaggregation does not inherently increase raw tokens/s. Long KV transfers or poorly balanced pools can outweigh isolation benefits. Compare this design with the preceding chunked-prefill schedule under the same GPU budget and service objectives. Week 4 develops placement and scheduling policies in more detail.

- [DistServe (OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [vLLM: disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/)

## 27. Experiment: sweep batch size on one GPU

**Suggested time: 1 minutes.**

The lab is a small causal Transformer with random weights. It studies execution behavior, not language quality or production model throughput. It preallocates KV storage and avoids copying a growing cache with torch.cat. The benchmark records synchronized wall-clock time including host dispatch, GPU work, KV writes, and greedy token selection. Prefill includes the first prediction. Decode rates refer to later tokens only. Reported trial percentiles describe repeated trial-average step durations, not serving tail latency. Hardware, software, model shape, precision and lengths are recorded in metadata. The implementation stays fixed, but automatic SDPA backend selection can change with tensor shapes; profile the selected kernels if explaining a performance change. Use the lab plotting script on the saved CSV; the presentation does not invent a performance curve. No local GPU measurements are bundled. The DistServe figure elsewhere in this deck is a separately attributed published measurement, not a run of this lab.

- [CS336 Lecture 6](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)
- [PyTorch CUDA timing](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
- [Berkeley L18, PDF pp. 47–53](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf#page=47)

## 28. A measurement should test a bottleneck hypothesis

**Suggested time: 0.5 minutes.**

Use the fixed-batch experiment to test Question 1 rather than infer a bottleneck from one curve. Sweep B with the same cached-length trajectory, then sweep initial context at fixed B and generation length. Longer context increases both attention work and KV reads, so a length sweep alone cannot prove bandwidth saturation. Inspect attention and dense-kernel time, achieved traffic/bandwidth or compute where available, and host/GPU gaps. A busy GPU can be limited by memory or arithmetic. For Question 2 compare numerically equivalent dense reconstruction and direct paged attention at matching shapes and precision; inspect copy traffic, peak live allocations and step latency. Profiling a smaller sub-operation is useful only when its cost matters to the end-to-end objective. These experiments exclude new arrivals and serving queues. The preceding notebook is a fixed-batch execution microbenchmark, not a production scheduler or a supplied measured result.

- [CS336 Lecture 6](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)
- [Berkeley L18, PDF pp. 47–53](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf#page=47)

## 29. Discussion

**Suggested time: 0.5 minutes.**

Use the remaining discussion time to ask what observation could falsify a proposed bottleneck. The full route is 35.25 minutes; a roughly 32.25-minute route skips 8, 27, 28. Week 3 develops multi-GPU parallelism and Week 4 studies scheduling, prefix reuse and serving policies in depth. Revisit the paged-prototype diagnostic if participants confuse memory allocation with the attention kernel.

- [CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10)
- [CS336 Lecture 5, pp. 52–54](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=52)
- [Berkeley L18, PDF pp. 9–16](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf#page=9)

## 30. Question 3: Take-home — implement GEMM in Triton

**Suggested time: 0 minutes.**

This is a take-home assignment adapted from the official Triton Matrix Multiplication tutorial, not an in-class coding task or a reported company interview question. Zero minutes means no additional planned lecture time; the assignment can be announced at the end. Implement a positive-dimension, contiguous row-major FP16 GEMM with an FP32 accumulator and FP16 output. tl.dot is permitted; calling torch.matmul or cuBLAS as the implementation is not. PyTorch is the validation baseline. Start with fixed tile sizes, map each program to one output tile, and iterate over K tiles. Handle invalid M/N input positions safely and zero-fill invalid K positions; mask output stores. The official reference wraps M/N load indices, so load masks are not the only valid approach. Test the three visible (M,N,K) cases, state rtol/atol, and report maximum error rather than requiring bitwise equality. Warm up and use CUDA events or a GPU benchmark utility, excluding compilation and autotuning from steady-state timing. Compare against torch.matmul under matching conditions. Grouped program ordering and autotuning are optional extensions. No requirement to outperform cuBLAS; the goal is a correct implementation and an explanation supported by measurements.

- [Triton: Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton: first kernel](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)

## 31. Solution 3: Triton GEMM tutorial

**Suggested time: 0 minutes.**

The large underlined title and footer both link directly to the user-requested official tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html. Use it as the reference solution after attempting the preceding take-home exercise. The tutorial supplies a blocked FP16 GEMM with FP32 accumulation, pointer arithmetic, boundary handling, grouped program ordering, autotuning, correctness checks, and benchmark code. Its displayed speedups are examples for particular environments, not a target or promise for every GPU and shape. This reference slide is outside the planned lecture time. The slide itself works offline; opening the external solution requires internet access.

- [Triton: Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
