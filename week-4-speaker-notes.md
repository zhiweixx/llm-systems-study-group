# Week 4 — Speculative decoding and LLM serving

30 slides. Slide 2 compares hardware bandwidths. Slides 3–17 cover speculative decoding and agent caching. Slides 18–30 develop a worked Llama 3.1 405B serving case. Hardware assumption for that case: eight servers, each with eight H100 SXM 80 GB GPUs, for 64 GPUs total.

Route: 2 hardware bandwidth; 3–4 speculative decoding and exact sampling; 5–6 KV continuation and cost; 7–8 sampling quiz and solution; 9–12 correctness and performance theory; 13 request lifecycle; 14–17 agent context and compaction. The 405B case then covers fit (19), replica placement (20), TP ownership (21), KV budget (22), TP16 (23), PP timing (24), decode/prefill costs (25–26), context parallelism (27), FP8 (28), configuration (29), and deployment choice (30).

This is an expanded teaching deck. Select a route for the meeting rather than treating the page count as a rehearsed duration. The first 17 slides and the 405B case can also be presented as separate sections.

The slide 2 chart distinguishes cited hardware specifications, published cache measurements, and architectural estimates. Other diagrams and numerical estimates are authored teaching examples, not GPU measurements. The applied-inference chapter supplies a worked-problem structure; all 405B calculations are rederived from primary model/hardware sources and distinguish one-request latency from pipelined throughput. Versioned vLLM documentation is 0.19.1; verify the selected backend and installed version.

The optional [serving lab](https://zhiweixx.github.io/llm-systems-study-group/week-4/lab.html) remains available separately. No GPU benchmark or cluster deployment was run to produce these slides. The HTML works offline; reference links need internet.

## 1. Week 4 · From model execution to an online service

**Section: Opening**

Slide 2 compares hardware bandwidths. Slides 3–12 cover speculative decoding, exact sampling, a multiple-choice quiz and theory. Slide 13 introduces a small serving example; slides 14–17 explain agent context and prefix-cache reuse. Slides 18–30 introduce a separate 405B deployment on eight 8-H100 servers: storage, TP/PP, KV capacity, latency versus throughput, context parallelism and precision choices. This is expanded teaching material; select a route for the available meeting time. Numerical deployment estimates are analytical and were not GPU-benchmarked.


## 2. Bandwidth: on-chip memory, HBM and GPU links

**Section: Hardware bandwidth**

This chart compares bandwidth scales, not latency. It is an original chart of published specifications, published microbenchmarks, and explicitly stated architectural estimates; we ran no GPU benchmark. The horizontal axis is logarithmic and all plotted values are decimal TB/s (1 TB/s = 1,000 GB/s). H100 and H200 mean SXM, and the networking reference is an eight-GPU DGX configuration. The entire set of NVLink interfaces on one GPU supplies 900/900/1,800 GB/s bidirectionally for H100/H200/B200. The plotted one-way rates are therefore 450/450/900 GB/s, not per-link rates or independent simultaneous bandwidth to every peer. HBM is 3.35/4.8/up to 8 TB/s per GPU. Do not double HBM bandwidth as if it were a full-duplex network interface. InfiniBand is a server configuration rather than a fixed GPU property. These DGX systems have eight 400 Gb/s ConnectX-7 compute-network adapters: 400/8 = 50 GB/s per adapter per direction, or 400 GB/s per node per direction when all eight adapters are used. Only the per-adapter rate is plotted. Usable application and collective bandwidth will be lower and topology-dependent. The hollow SRAM markers are not official bandwidth guarantees or a controlled generational benchmark. FlashAttention-3 Table 1 uses 12 TB/s for H100 L2 and estimates SMEM bandwidth as 128 bytes/cycle/SM × 132 SMs × 1.83 GHz = 30.92 TB/s. Other L2 microbenchmarks obtain different rates because clock, working-set size, read/write mix, partition locality, and kernel design differ. We leave H200 L2 unplotted because an independent comparable value was not verified. The H200 SMEM marker uses the same Hopper throughput, 132 SMs, and an assumed 1.83 GHz; it is an architectural estimate, not a measurement. The B200 SMEM range uses the FlashAttention-4 values of 128 bytes/cycle/SM and 148 SMs with assumed clocks of 1.85–2.0 GHz, yielding 35.05–37.89 TB/s. This range reflects clock assumptions, not a statistical confidence interval. Chips and Cheese measured B200 L2 at 21 TB/s for local-partition working sets and 16.8 TB/s when accesses cross partitions. That plotted range represents these access patterns. RRZE-HPC reports H100 L1 hits approaching 128 bytes/cycle/SM. Applying the SXM SM count and 1.83 GHz gives the approximate 31 TB/s L1 aggregate mentioned below the chart; the microbenchmark itself used a PCIe H100, so this is a normalized SXM estimate, not its measured total. No matched H200/B200 L1 result is plotted. L1 and SMEM share on-chip storage resources but have different access semantics; their rates must not be added. All-SM aggregate means independent SMs accessing their own local storage concurrently. One H100 SM at 1.83 GHz contributes only about 234 GB/s, not 31 TB/s by itself. Register and B200 tensor-memory datapaths are not represented by the SMEM numbers. Use this figure to motivate data reuse within a GPU and to understand why moving tensors between GPUs or nodes can dominate serving costs.

- [H100 specs](https://www.nvidia.com/en-us/data-center/h100/)
- [H200 specs](https://www.nvidia.com/en-us/data-center/h200/)
- [DGX B200](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html)
- [B200 HBM](https://www.nvidia.com/en-us/data-center/dgx-b200/)
- [DGX Hopper](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html)
- [FA3 Table 1](https://tridao.me/publications/flash3/flash3.pdf)
- [FA4](https://tridao.me/blog/2026/flash4/)
- [B200 L2 test](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut)
- [L1 test](https://github.com/RRZE-HPC/gpu-benches)

## 3. Speculative decoding: draft, verify, commit

**Section: Speculative decoding · overview**

h is the committed history and d_<i is the draft prefix before position i. The classical draft model samples gamma proposals autoregressively. Since those candidates are now known, the target can score all candidate-conditioned contexts together using causal attention, as in a short prefill. This computes probability distributions, not independently sampled target tokens or argmax matches. In a common KV convention, the final committed token has not yet been processed; feed it plus the draft tokens to obtain gamma candidate scores and a bonus distribution. Equivalently, an already computed first-position distribution can be reused. Each proposal and its target score condition on the same history. Verification is one model call, not one CUDA kernel, and layers remain sequential. At the first rejection the subsequent draft history is invalid, so discard that suffix. If every proposal is accepted, sample one bonus from the final target distribution. The exact stochastic acceptance and residual correction appear next. The range of emitted tokens ignores EOS and output limits. Lower-batch target decode can benefit because multi-position verification amortizes weight movement; extra draft and verification work can outweigh those savings.

- [Leviathan et al., §2–3](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)
- [Chen et al., Algorithm 2](https://arxiv.org/html/2302.01318v1)

## 4. Exact speculative sampling

**Section: Speculative decoding · algorithm**

This is the stochastic sampling algorithm in Leviathan et al. Section 2.3, written in native notation rather than using an argmax illustration. Fix one history, draw X from q and an independent U uniformly on (0,1), then accept if U is at most min(1,p(X)/q(X)). A proposal necessarily has q(X)>0, so the evaluated ratio is defined. On rejection draw a new token from the normalized positive residual p-q. The residual normalizer equals the overall rejection probability; if it is zero, p=q and the correction branch is never reached. Tokens outside the proposal support can still be generated by the residual. For a speculative sequence apply this rule in order to each reached candidate, using p_i and q_i conditioned on the identical prefix. Stop after the first rejection and discard later draft candidates and invalid scores. If all candidates pass, sample the bonus directly from the target distribution after the full accepted draft. Ordinary probabilities after temperature, top-p or other intended sampling transformations are required; raw logits cannot be used in the ratio. The exactness claim is about the ideal probability algorithm. It does not require the same sampled text for a fixed RNG seed or numerically identical floating-point implementations. The proof follows the multiple-choice quiz.

- [Leviathan et al., §2.3 / Algorithm 1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)
- [Chen et al., Algorithm 2](https://arxiv.org/html/2302.01318v1)

## 5. After verification: rollback, continuation, and the bonus

**Section: Speculative decoding**

Before verification the target cache covers h except its last token u. Verification evaluates u and all four draft input positions, so h is now fully cached. When sat is rejected, the target KV for The and cat is valid because their histories are unchanged. KV for sat and quietly belongs to an invalid branch and must no longer be visible to attention. The replacement slept is selected from a distribution already computed at the previous position; slept itself has not been fed through the target, so its KV is pending until continuation. In the all-accepted alternative, the final draft position provides the distribution for one bonus token. That bonus is also not yet an evaluated target input. In practice engines track valid lengths or block mappings; rollback does not necessarily erase bytes. The draft model has its own weights and KV state, which must be synchronized with the committed sequence; accepted draft KV cannot be substituted for target KV. We omit EOS and length-limit termination, which can truncate the emitted block.

- [Leviathan et al., §2–3](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)
- [Transformers assisted decoding](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py)

## 6. When does the extra work pay off?

**Section: Speculative decoding**

Start at two accepted proposals, matching the rollback example. Four sequential draft steps cost 4 ms, verification costs 14 ms and bookkeeping costs 2 ms, for 20 ms to emit three tokens; ordinary target decoding would take 30 ms. Move accepted-prefix length to zero: the draft round still costs 20 ms but emits only the correction, versus 10 ms normally. Move to four: all four proposals plus a bonus yield five outputs. Then increase verification time: good acceptance can still fail to pay off. For this illustrative single round, emitted tokens = accepted prefix length + one correction or bonus; EOS and output limits are excluded. The sliders vary independent assumptions and are not hardware predictions. Across actual rounds use total elapsed time divided by total emitted tokens, rather than averaging speedup ratios. A separate draft also uses weight memory and its own KV cache; a configuration that speeds one stream can reduce capacity or throughput under concurrency. Verification time need not equal one-token target latency.

- [Leviathan et al., §2–3](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)
- [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)

## 7. Quiz: Resampling from the target after rejection

**Section: Speculative decoding theory · multiple-choice quiz**

Allow approximately 2–3 minutes for discussion and the immediate solution. There is one universally valid expression, C: Pr(Y=x)=min(p(x),q(x))+Z*p(x), where Z=Pr(reject)=1-sum_x min(p(x),q(x)). The quiz deliberately changes the correct algorithm by sampling the replacement from p. A is the desired target law but not the law of this altered procedure in general. B mistakenly uses q as the conditional law of accepted proposals; accepting with token-dependent probabilities changes that law. C adds the accepted joint mass and the replacement joint mass. D uses q for the replacement, contrary to the stated procedure. Some options can agree for special distributions such as p=q, but C is the identity valid for all p,q. The acceptance ratio is evaluated only for sampled X with q(X)>0. This is an authored assessment based on Leviathan Section 2.3, not a reported company interview question.

- [Adapted from Leviathan et al., §2.3](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)

## 8. Quiz solution: account for both output paths

**Section: Speculative decoding theory · quiz solution**

C follows by summing the joint probabilities of two disjoint events: emitting x from an accepted proposal, and emitting x from the replacement branch. The former is q(x)*min(1,p(x)/q(x))=min(p(x),q(x)); the latter is Z*p(x) because the replacement is an independent draw from p and rejection occurs with probability Z. B is a common misuse of a mixture: the accepted component is not q. For 0<Z<1, its conditional law is min(p,q)/(1-Z). If Z=0 the branch weights are degenerate and the same unconditional expression still applies. The displayed four-token counterexample has accepted mass (0.20,0.20,0.10,0.10), total acceptance 0.60 and rejection 0.40. Adding Z*p gives (0.28,0.40,0.18,0.14), not p. A therefore fails in general; D is the law of a different rule that resamples from q. Correct residual resampling replaces Z*p by max(p-q,0), yielding min(p,q)+max(p-q,0)=p. The following proof derives the residual normalizer. Equal distributions and disjoint-support cases can make several distractors coincide numerically with C; the question explicitly asks for the identity valid for arbitrary distributions.

- [Adapted from Leviathan et al., §2.3](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)
- [Leviathan et al., App. A.1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)

## 9. Why speculative sampling is exact

**Section: Speculative decoding theory · 8–10 min**

This proof completes the correction motivated by the multiple-choice quiz. Fix a history and let p and q be normalized target and proposal distributions over a common vocabulary, after any intended sampling transformations. Propose X from q and accept with probability min(1,p(X)/q(X)). For any vocabulary item x, the joint probability of proposing and accepting x is min(p(x),q(x)). Its sum is the total acceptance probability. Since p and q each sum to one, the omitted target mass Z=sum_x max(p(x)-q(x),0) equals the rejection probability 1-sum_x min(p(x),q(x)). On rejection draw Y from r(x)=max(p(x)-q(x),0)/Z. This contributes unconditional mass Z*r(x), exactly filling the missing target mass. Thus min(p,q)+max(p-q,0)=p pointwise. When q(x)=0 the algorithm cannot propose x, so it never evaluates that ratio for such an x; positive target mass there comes through the residual. When Z=0, the rejection branch has zero probability. At successive positions, use the distributions for the actual accepted history and discard the invalid suffix after a rejection. This yields the target autoregressive joint distribution by the chain rule. The proof concerns the ideal probability algorithm; floating-point implementations can differ numerically. Chen uses the opposite p/q naming; this deck consistently uses p for target and q for draft.

- [Leviathan et al., App. A.1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)
- [Chen et al., §4.2](https://arxiv.org/html/2302.01318v1#S4.SS2)

## 10. Acceptance measures distribution overlap

**Section: Speculative decoding theory**

For a fixed context, sum the accepted mass min(p(x),q(x)) over the vocabulary. Using min(p,q)=(p+q-|p-q|)/2 and the fact that both distributions sum to one, alpha=1-(1/2)sum|p-q|. The subtracted term is total variation distance. In our example TV=(|0.6-0.8|+|0.4-0.2|)/2=0.2 and total acceptance is 0.8. Distinguish this 80% overall acceptance from the 75% acceptance conditional on having proposed A. Perfectly equal distributions give alpha=1; disjoint supports give alpha=0. This is a fixed-context result for stochastic speculative sampling, not simply the fraction of matching argmax predictions. Real histories give different acceptance probabilities. A draft with greater overlap may also be more expensive, so overlap alone does not determine speedup. For one-token exact couplings, this overlap is also the largest possible probability that a q-distributed proposal and p-distributed output agree; no mass at x can be shared beyond min(p(x),q(x)).

- [Leviathan et al., §3](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)

## 11. Expected tokens per verification round

**Section: Speculative decoding theory**

Let A be the consecutive accepted-prefix length among gamma proposals. N=A+1 includes one correction on rejection or one bonus when all proposals pass. This counts emitted tokens before EOS or a maximum-output cutoff. The tail-sum identity for a positive integer N gives E[N]=sum_{k=1}^{gamma+1}P(N>=k)=1+sum_{i=1}^{gamma}P(A>=i). No independence assumption is needed for this first formula. If P(proposal i accepted | proposals 1 through i-1 accepted) equals the same alpha at every reached depth, P(A>=i)=alpha^i and the finite geometric series follows. This is a model assumption, not a consequence of knowing one global average acceptance rate. At alpha=1 use the continuous limit gamma+1; alpha=0 gives one output per round. The chart displays the probability that each output position exists: 1,0.8,0.64,0.512,0.4096. Their sum is 3.3616, not 1+4*0.8=4.2, because a later accepted draft token is useful only if all earlier proposals survive. Actual context- and depth-dependent survival rates can be inserted directly into the general identity. A single round emits an integer; this fractional result is the average over many rounds. The paper derives the same geometric model using an i.i.d. acceptance assumption.

- [Leviathan et al., §3](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)

## 12. Expected speedup and draft length

**Section: Speculative decoding theory**

Use a stationary fixed-workload approximation and enough generation rounds for average costs to be meaningful. In the displayed cost model a round costs C(gamma)=gamma*D+V(gamma)+O. Its average time per output token is C/E[N], versus T for target-only decoding, so speedup is T*E[N]/C. If round cost also varies, use E[C]/E[N] for the long-run average time per token, not E[C/N] or the average of per-round speedups. For variable contexts and system load, measure aggregate elapsed time and emitted tokens against a matched baseline. The illustrative curve assumes alpha=0.8, T=10 ms, D=1 ms, V(gamma)=10+gamma ms and O=2 ms. At gamma=4, E[N]=3.3616 and C=20 ms, giving speedup 1.6808. Integer gamma=4 maximizes this example over 1 through 12. The earlier slider chose exactly two accepted tokens in one round, hence 1.5x; this theory uses an average over random acceptance, hence 1.68x. The classic simplified formula follows if verification takes exactly one target step and overhead is zero: S=(1-alpha^(gamma+1))/((1-alpha)*(1+gamma*c)), c=D/T. For gamma=1 this is (1+alpha)/(1+c), so alpha>c is the break-even condition under those simplified assumptions only. For alpha<1, E[N] approaches 1/(1-alpha), but positive draft cost grows with gamma. The marginal benefit of one more proposal is alpha^(gamma+1); increasing gamma helps only when this extra progress per extra millisecond exceeds current E[N]/C. Context length, KV traffic, concurrency, draft memory, synchronization and sampling implementation affect the measured costs. Faster generation for one stream does not by itself imply higher throughput or better latency under load.

- [Leviathan et al., §3](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)

## 13. Follow one request from arrival to completion

**Section: Setup · 3 min**

Transition from accelerating one generation to serving many concurrent requests under a fixed GPU budget. Assume one 8B model fits on each GPU at the chosen precision. A replica is a complete model instance, not a tensor-parallel shard; the router selects ONE of the four replicas for a request. Each replica owns its own queue and local KV cache, with no cross-GPU cache sharing in this baseline. A long-running agent issues many model calls as its task progresses; repeated transcript prefixes create temporal reuse within that session. Different sessions compete for the same service. Keep the model, precision and hardware fixed when comparing serving policies. The lower diagram follows one request through the selected engine, including its earlier routing time. This is a logical sequence, not a scale drawing or a promise of one GPU kernel per box. Prefill may be split across engine iterations and interleaved with other requests. Prefix caching can skip a portion of input computation, but routing, queuing, uncached work, and the output-generation path remain. The first token is produced using the prompt processing result; subsequent tokens are generated by decode steps. On completion, release the request’s references and retain reusable prefix blocks if policy allows. For n>1 output tokens, client mean TPOT=(last-token time−first-token time)/(n−1), with streaming event/token conventions stated by the benchmark. Individual ITL samples need not equal the mean. Network buffering can change observed streaming gaps. Define timestamp boundaries before comparing server and client metrics. The later 405B deployment is a separate hardware case; it applies these latency metrics to multi-node replicas.

- [vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics/)
- [Berkeley L18](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf)

## 14. An agent session contains many model calls

**Section: Agent cache reuse**

Start with one coding task, not several unrelated users. Each model call returns an action or response; the harness executes tools and sends a new input containing the previous transcript plus new content. Blue marks an earlier input assumed retained by the same model/cache; tan marks material newly supplied in this diagram. The 20k includes the action serialization and test log; the next16k consist of the subsequent model action and returned file excerpts. Each row is one successive model call with the earlier input retained, so its blue region can be reused. At the end, assume the full 40k input is cached. The next slides use this exact snapshot. We conservatively count previous input reuse: whether generated output is immediately reusable as the next input depends on the serving implementation and exact token serialization. Bars show transcript structure, not proportional lengths. This scenario is explicitly supported by the vLLM multiround conversation example and Manus’s append-only agent loop; no prevalence ranking across industry workloads is implied.

- [Manus: agent context](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- [vLLM: prefix reuse](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

## 15. Hit rate needs a denominator and an explanation

**Section: Agent cache reuse**

Here H is cached input tokens divided by total input tokens for one model call, not a hardware L1/L2 hit rate and not a request-level Boolean. U counts fresh input positions, not bytes read from HBM or equal-cost units of GPU time. The two rows are alternative next calls after the same 40k snapshot; if those two calls are included in a reporting window, the aggregate ratio is 80/102=78.43%, whereas averaging their percentages gives about80.95%. Both requests have some cache reuse, so an any-hit request metric would report100% and conceal the difference. vLLM exposes queried/hit-token counters; for a specific engine verify which eligible tokens and boundaries its denominator covers. Provider usage fields can separate uncached tokens, cache creation and cache reads: reconstruct total input according to that provider rather than dividing by an uncached-only field. At fixed model/configuration, cache residency and complete-block alignment are assumed for these illustrative counts.

- [vLLM: cache metrics](https://docs.vllm.ai/en/latest/design/metrics/#prefix-cache-metrics)
- [vLLM: prefix reuse](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

## 16. Editing the past moves the cache-reuse boundary

**Section: Agent cache reuse**

This is selective compaction of an old test interaction (the action and its tool result), not a requirement that all compaction systems use this exact format. Ordinary exact-prefix reuse is the assumption. The first summary token differs from the old action/log range, so only the fixed4k prefix matches. At the first Transformer layer, token embeddings may not yet mix all previous context, but the full multilayer KV state of the unchanged later text generally depends on the changed earlier context; positions may also shift when20k becomes2k. Thus neither textual identity of the later messages nor retaining their old cache blocks makes those blocks valid for this new request. Model weights, adapters, token serialization and relevant position/input settings must also agree. Engines use parent-prefix identities in cache keys; full-block boundaries may round the reusable prefix down. The old branch can remain valid for the original transcript. Figure widths are schematic; the next slide uses a proportional token scale. Tool-result clearing can use a placeholder rather than a generated summary; both change past tokens. Full-history compaction may replace a larger range and move the boundary differently.

- [vLLM: cache identity](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [Anthropic: context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing#context-editing-and-prompt-caching)

## 17. Compaction causes a rebuild, then reuse recovers

**Section: Agent cache reuse**

Advance through retain, compact, and the next append. Starting cached input is4k fixed prefix+20k old test interaction+16k later history=40k. Keeping it and appending2k requires42k total input with40k reused and2k newly prefilling. Replacing the old20k test interaction by a2k summary gives24k input: only4k match, so20k must be newly computed. After this edited input is processed and cached, another2k append gives26k input,24k reused and2k uncached. This last suffix includes the previous generated action and new observation under our conservative previous-input convention. The first two rows are alternatives at the same point, not two sequential executions; the third follows the second. Treat cache residency, eligibility, compatible model settings and stable exact serialization as assumptions. Counts idealize block rounding and any final-token recomputation. Summary generation is a separate cost and is not included in these token numbers. The bars show input positions, not GPU memory traffic, elapsed time or retained physical cache allocation. A rebuilt shorter active history can reduce subsequent attention and active KV needs, even though old unused cached blocks may remain in the pool until eviction.

- [Anthropic: context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing#context-editing-and-prompt-caching)
- [vLLM: prefix reuse](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

## 18. Serve a 405B model on eight H100 nodes

**Section: Case study · Llama 3.1 405B on 64 H100s**

This is a new deployment case, separate from the four-GPU 8B example on slide 13. Here a node explicitly means an eight-GPU server: eight nodes contain 64 GPUs. The hardware is a DGX-like H100 SXM 80 GB system with an intra-node NVSwitch fabric and an inter-node RDMA-capable network whose actual bandwidth and latency must be measured. The H100 BF16 peak is the dense peak, not the doubled sparse peak. The linked applied-inference chapter supplies the progression from capacity to latency and deployment, but its 70B numerical examples are not reused. All 405B values are derived independently. Batch B later means the number of sequences in one decode microbatch on one replica; sequence length S means the number of currently cached positions. S grows during generation. Storage units are decimal: 1 GB = 10^9 bytes. The baseline uses the canonical architecture with 8 KV heads, BF16 weights, and BF16 KV unless explicitly changed. P = 405 billion is rounded; equal weight partitioning is a planning approximation. The baseline excludes prefix sharing, CPU offload, and a speculative draft model. No GPU measurements were run.

- [Applied inference chapter](https://liltom-eth.github.io/scaling-book-pytorch/chapters/applied-inference.html)
- [DGX H100 system guide](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html)
- [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)

## 19. First constraint: one BF16 copy exceeds one node

**Section: Case study · Llama 3.1 405B on 64 H100s**

The bar compares total storage, not a physically pooled memory address space. Each GPU must hold its assigned tensors. ceil(810 / 80) = 11 is only a necessary byte-capacity condition; it is not a suggested tensor-parallel degree. TP degrees must suit the model heads and matrix partitions, while runtime allocations and KV also need space. A practical two-node option is TP8 × PP2: each node owns 63 of the 126 Transformer layers and shards its layers over eight GPUs. The embedding and output matrices at the two endpoints, plus other unsharded tensors, can make the true rank allocations unequal. Section 6.1 of the Meta paper reports this two-machine BF16 inference design. Storage units are decimal: 1 GB = 10^9 bytes. The baseline uses the canonical architecture with 8 KV heads, BF16 weights, and BF16 KV unless explicitly changed. P = 405 billion is rounded; equal weight partitioning is a planning approximation. The baseline excludes prefix sharing, CPU offload, and a speculative draft model. No GPU measurements were run.

- [Meta model configuration](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)
- [Llama 3 paper §6](https://arxiv.org/html/2407.21783v3#S6)
- [Week 3: combine groups](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-17)

## 20. A starting layout: four replicas of TP8 × PP2

**Section: Case study · Llama 3.1 405B on 64 H100s**

Read one horizontal row first: it is one complete 405B model. Node 0 stores the first 63 layers and node 1 stores the last 63. Within either node, eight GPUs cooperate on every local layer. Duplicate this layout to create four replicas using the 64-GPU budget. These are inference replicas, the serving analogue of data parallelism from Week 3, rather than DDP training jobs. KV belongs to the chosen replica. A session can be routed back to its replica for cache reuse, subject to load and cache residency. The PP arrow represents a logical hidden-state transfer; actual wire volume depends on whether the backend transfers replicated activations or gathers/scatters a sharded boundary. One request does not span all four replicas. Storage units are decimal: 1 GB = 10^9 bytes. The baseline uses the canonical architecture with 8 KV heads, BF16 weights, and BF16 KV unless explicitly changed. P = 405 billion is rounded; equal weight partitioning is a planning approximation. The baseline excludes prefix sharing, CPU offload, and a speculative draft model. No GPU measurements were run.

- [Llama 3 paper §6](https://arxiv.org/html/2407.21783v3#S6)
- [Week 3: serving replicas](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-4)
- [Week 3: PP](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-10)

## 21. Apply Week 3 tensor parallelism to this model

**Section: Case study · Llama 3.1 405B on 64 H100s**

The diagram depicts one layer on its assigned PP stage. GQA associates 16 query heads with each of eight KV heads. TP8 assigns one complete KV head and its 16 associated query heads to each rank. The row-parallel attention output projection sums contributions from all ranks. A conventional paired column-parallel/row-parallel SwiGLU FFN similarly has a reduction after the down projection: 53,248 / 8 = 6,656 intermediate features per rank. Both gate and up projections follow that split. Thus the standard replicated-activation TP forward layout has two AllReduces per layer. Sequence-parallel and fused implementations may replace or combine collectives; the diagram is a reasoning baseline, not a profiler trace. Each PP node performs its own 126 layer reductions; 252 lie along a full forward path. Storage units are decimal: 1 GB = 10^9 bytes. The baseline uses the canonical architecture with 8 KV heads, BF16 weights, and BF16 KV unless explicitly changed. P = 405 billion is rounded; equal weight partitioning is a planning approximation. The baseline excludes prefix sharing, CPU offload, and a speculative draft model. No GPU measurements were run.

- [Meta model configuration](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)
- [Week 3: attention TP](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-9)
- [Week 3: TP reduction](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-8)

## 22. Budget KV memory on each GPU

**Section: Case study · Llama 3.1 405B on 64 H100s**

KV bytes for the whole model per cached token are 2 × 126 × 8 × 128 × 2 = 516,096. One 8,192-token request uses 4.227858432 GB of unique KV across the replica. TP8 head sharding and PP2 layer sharding divide this by 16: 0.264241152 GB per GPU per request. At B = 32, this is 8.455716864 GB per GPU. The assumed rank budget for KV is 80 − 50.625 − 8 = 21.375 GB, so floor(21.375 / 0.264241152) = 80. Four replicas would fit approximately 320 such requests under this simplified capacity model. This is not a throughput claim, a max-num-seqs recommendation, or a promise of an ITL target. Context grows during generation. At 32,768 cached tokens, the same calculation gives 20 requests per replica. Runtime workspace may grow with the batch and prefill token budget; the 8 GB reserve is not a guarantee. No prefix sharing is credited. Storage units are decimal: 1 GB = 10^9 bytes. The baseline uses the canonical architecture with 8 KV heads, BF16 weights, and BF16 KV unless explicitly changed. P = 405 billion is rounded; equal weight partitioning is a planning approximation. The baseline excludes prefix sharing, CPU offload, and a speculative draft model. No GPU measurements were run.

- [Meta model configuration](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)
- [Week 3: KV ownership](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-9)

## 23. Why not simply use TP16?

**Section: Case study · Llama 3.1 405B on 64 H100s**

Compare one replica using exactly the same two nodes. In plain head-sharded TP16, each rank handles 8 of the 128 query heads. Each group of 16 query heads shares one KV head, so two ranks need the same KV history. Without context sharding or another specialized attention scheme, each KV head has two copies across the 16 ranks. Each rank now holds one head across 126 layers, versus 63 layers in TP8 × PP2, so per-rank KV doubles. The exact checkpoint and implementation matter: Meta also distributed an MP16 configuration with 16 KV heads, reflecting duplication; this case starts from the canonical architecture with 8 KV heads. The duplicated K/V projections add about 0.528 GB per rank beyond 810 / 16, so TP16 weights occupy approximately 51.2 GB per rank; this duplication remains with TP16 + DCP2. All 252 conventional TP reductions now span both nodes. Wider TP can lower local layer time, but exposed network cost can negate it. Do not treat the 900 GB/s aggregate bidirectional NVLink specification as inter-node or usable AllReduce bandwidth. Context parallelism later gives a way to remove the cache duplication. Storage units are decimal: 1 GB = 10^9 bytes. The baseline uses the canonical architecture with 8 KV heads, BF16 weights, and BF16 KV unless explicitly changed. P = 405 billion is rounded; equal weight partitioning is a planning approximation. The baseline excludes prefix sharing, CPU offload, and a speculative draft model. No GPU measurements were run.

- [Meta model configuration](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)
- [vLLM Q/K/V sharding](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/layers/linear.py#L1009)
- [Week 3: placement](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-18)

## 24. Pipeline throughput is different from token latency

**Section: Case study · Llama 3.1 405B on 64 H100s**

This teaching timeline uses a weight-streaming lower bound as the duration of one stage. A and B are disjoint sets of requests, not successive tokens of a single sequence that could run without dependencies. At 0–15.1 ms, node 0 processes token t for A. At 15.1–30.2 ms, node 1 processes A while node 0 processes B. Only after A finishes both stages can token t + 1 for A begin. In a real engine, sampling, transfers, and scheduling add time, and the two stages may not balance. The ideal pipeline can emit one microbatch result every 15.1 ms after filling, although each microbatch spends 30.2 ms on its path. Dividing 810 GB by 16 × 3.35 TB/s estimates one stage or an ideal throughput interval, not single-microbatch latency. Meta measures improved throughput and increased latency with microbatching, so the ideal drawing is not a measured speedup. Prompt chunk pipelining can also change schedules and must be modeled explicitly. Storage units are decimal: 1 GB = 10^9 bytes. The baseline uses the canonical architecture with 8 KV heads, BF16 weights, and BF16 KV unless explicitly changed. P = 405 billion is rounded; equal weight partitioning is a planning approximation. The baseline excludes prefix sharing, CPU offload, and a speculative draft model. No GPU measurements were run.

- [Llama 3 paper §6](https://arxiv.org/html/2407.21783v3#S6)
- [Week 3: PP microbatches](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-11)
- [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)

## 25. Decode: reuse weights, but read each request’s KV

**Section: Case study · Llama 3.1 405B on 64 H100s**

For TP8 × PP2, one microbatch traverses both stages, so the total path uses an effective 8-way bandwidth denominator in this additive stage model. Each stage has half the weight and KV volume. At B = 32 and S = 8,192, KV occupies 135.291469824 GB across the model, total traffic is 945.291469824 GB, and the ideal stage-summed HBM time is 35.2721 ms. Leading linear work is 2PB = 25.92 TFLOPs; dividing by 8 × 989 TFLOP/s gives 3.276 ms. These are separate lower-bound components, not terms to simply add: compute and memory can overlap. The model assumes each weight and each unique KV entry is read once with efficient reuse within each GQA group; actual traffic can be higher. Decode attention adds approximately 4 × 126 × 16,384 × B × S FLOPs, about 8.4% of 2PB at S = 8,192, plus non-matmul work. At these small batches, the HBM bound dominates the linear compute bound. This does not imply that all decode workloads are bandwidth-bound. No output-token/s estimate is made without a schedule for independent PP microbatches. Storage units are decimal: 1 GB = 10^9 bytes. The baseline uses the canonical architecture with 8 KV heads, BF16 weights, and BF16 KV unless explicitly changed. P = 405 billion is rounded; equal weight partitioning is a planning approximation. The baseline excludes prefix sharing, CPU offload, and a speculative draft model. No GPU measurements were run.

- [Applied inference chapter](https://liltom-eth.github.io/scaling-book-pytorch/chapters/applied-inference.html)
- [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)
- [Week 3: communication costs](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-19)

## 26. Prefill reuses weights across thousands of tokens

**Section: Case study · Llama 3.1 405B on 64 H100s**

P = 405B and T = 8,192 give 2PT = 6.63552 PFLOPs for the leading dense forward work. Do not use the training approximation 6PT. For one unchunked prompt sent through TP8 × PP2, each stage does half the work on 8 GPUs: 6.63552e15 / (2 × 8 × 989e12) = 0.41933 s; the two stages sum to 0.83867 s. Using all 16 GPU peaks at once would assume the same prompt occupies both sequential stages simultaneously. Independent prompts, or a supported pipelined chunk schedule, can improve steady-state utilization; state those assumptions separately. The weight-only streaming floor is 30.2 ms. Under a weight-dominated linear model, BF16 arithmetic intensity is about 8,192 FLOP/byte, far above 989 TFLOP/s divided by 3.35 TB/s ≈ 295 FLOP/byte. Full prefill also includes causal attention, whose cost grows quadratically with sequence length; 2PT alone is not an exact model-wide count. Some embedding/output effects and kernel tiling also make the parameter-count approximation imperfect. Storage units are decimal: 1 GB = 10^9 bytes. The baseline uses the canonical architecture with 8 KV heads, BF16 weights, and BF16 KV unless explicitly changed. P = 405 billion is rounded; equal weight partitioning is a planning approximation. The baseline excludes prefix sharing, CPU offload, and a speculative draft model. No GPU measurements were run.

- [Applied inference chapter](https://liltom-eth.github.io/scaling-book-pytorch/chapters/applied-inference.html)
- [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)
- [Week 3: PP scheduling](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-11)

## 27. Long contexts make KV memory a limiting factor

**Section: Case study · Llama 3.1 405B on 64 H100s**

The canonical maximum context is 131,072 tokens, including generated output; the example is a near-limit cache snapshot, not a 131,072-token prompt with unlimited additional output. Unique BF16 KV = 516,096 × 131,072 = 67.645734912 GB per request. At B = 4, TP8 × PP2 divides the total KV by 16: 16.911433728 GB per GPU. Plain TP16 with duplicated heads divides it only by 8: 33.822867456 GB per GPU. Add approximately 50.625 GB of weights for TP8 × PP2 or 51.153 GB for TP16, including duplicated K/V projections, and the hypothetical 8 GB reserve. In vLLM 0.19.1, decode context parallelism for GQA reuses the TP group: TP16 + DCP2 divides cached positions between ranks that would otherwise duplicate a KV head. The 2× reduction is a logical KV-storage calculation, not a tested 405B deployment benchmark. It requires a compatible model, version, and attention backend. The implementation interleaves cached positions as the history grows; the diagram shows disjoint shares, not contiguous halves. The distributed attention path exchanges queries/results and merges softmax statistics, so saved memory does not establish a latency improvement. Week 3 CP provides the conceptual sequence partition; this versioned DCP implementation nests within TP rather than adding GPUs. Long-prefill context parallelism is a separate scheduling/communication choice. Storage units are decimal: 1 GB = 10^9 bytes. The baseline uses the canonical architecture with 8 KV heads, BF16 weights, and BF16 KV unless explicitly changed. P = 405 billion is rounded; equal weight partitioning is a planning approximation. The baseline excludes prefix sharing, CPU offload, and a speculative draft model. No GPU measurements were run.

- [vLLM context parallelism](https://docs.vllm.ai/en/v0.19.1/serving/context_parallel_deployment/)
- [Meta model configuration](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)
- [Week 3: context parallelism](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-12)

## 28. FP8 can fit a full replica on one node

**Section: Case study · Llama 3.1 405B on 64 H100s**

Weight precision changes here, while KV stays BF16. The actual canonical architecture has approximately 405.853 billion parameters. Meta quantizes the three FFN matrices in the 124 middle layers: Q = 124 × 3 × 16384 × 53248 = 324.538 billion parameters. Attention, endpoint layers, embeddings and output weights remain BF16 in the cited loader. Stored weight bytes are Q + 2(P − Q) ≈ 487.168 GB, before scales; TP8 row scales bring the estimate to roughly 487.286 GB. Round to 61 GB per GPU for the capacity example. A single 8,192-token request uses 4.228 GB of BF16 KV across the TP8 replica, or 0.5285 GB per GPU. With the illustrative 8 GB reserve, the rounded budget fits about 20 such requests per replica. Ignoring scales narrowly suggests 21, which is too optimistic. Four BF16 replicas have about 320 slots at this context snapshot; eight mixed-FP8 replicas have about 160. This is not evidence that FP8 reduces throughput: latency, pipeline overhead, batching and kernel speeds also change. Keeping four replicas and using the saved weight memory for KV would be another valid design. NVIDIA NIM documents a separately optimized FP8 profile on eight H100 SXM GPUs; its recipe must not be equated with Meta’s. These analytical memory estimates do not establish runtime feasibility for every buffer shape or backend. Measure peak allocation and task quality with the actual checkpoint. Storage units are decimal: 1 GB = 10^9 bytes. The baseline uses the canonical architecture with 8 KV heads, BF16 weights, and BF16 KV unless explicitly changed. P = 405 billion is rounded; equal weight partitioning is a planning approximation. The baseline excludes prefix sharing, CPU offload, and a speculative draft model. No GPU measurements were run.

- [Meta FP8 implementation](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/quantization/loader.py#L53-L96)
- [NVIDIA 405B FP8 support](https://docs.nvidia.com/nim/large-language-models/1.13.0/supported-models.html#llama-3-1-405b-instruct)

## 29. Turn the layout into a serving configuration

**Section: Case study · Llama 3.1 405B on 64 H100s**

This is a configuration example based on vLLM 0.19.1 documentation, not a deployment executed in this session. Prerequisites are approved access to the model checkpoint, identical supported container/CUDA/NCCL/software versions on both servers, checkpoint availability, a two-node Ray cluster with 16 visible GPUs, and a configured inter-node network. Restrict one replica to exactly its pair of nodes, via isolated clusters or explicit resource placement; four unconstrained launches into one shared cluster can give unintended placement. Confirm each 8-rank TP group stays inside a server and PP links the two groups. Repeat across node pairs 0–1, 2–3, 4–5, and 6–7. The setting max-model-len = 16384 includes prompt plus generated tokens. The setting max-num-seqs = 32 is an initial concurrency cap rather than proof that every 32-request, 16k-context workload fits with all buffers. The earlier capacity example was a snapshot at 8,192 cached positions; recalibrate for allowed growth. Runtime GPU memory utilization and KV allocation settings are not equated to the teaching reserve of 8 GB. Prefix caching, speculative decoding, and quantization should be introduced only after this baseline is measured so their effects can be isolated. Storage units are decimal: 1 GB = 10^9 bytes. The baseline uses the canonical architecture with 8 KV heads, BF16 weights, and BF16 KV unless explicitly changed. P = 405 billion is rounded; equal weight partitioning is a planning approximation. The baseline excludes prefix sharing, CPU offload, and a speculative draft model. No GPU measurements were run.

- [vLLM distributed serving](https://docs.vllm.ai/en/v0.19.1/serving/parallelism_scaling/)
- [Week 3: combined layout](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-17)

## 30. Which layout would we actually choose?

**Section: Case study · Llama 3.1 405B on 64 H100s**

This final slide returns to the original 64-GPU decision. The baseline is defensible from topology and Meta’s deployment, but is not proven optimal on an unspecified cluster. First compare 16-GPU replicas with fixed model precision, contexts, arrival traces, and cache state. For lower single-request ITL, TP16 may remove the PP serial-stage penalty, but only if collective overhead does not erase the gain. For long-context GQA, test TP16 + DCP2 and inspect attention communication. For FP8, revalidate quality, actual checkpoint bytes, kernel choices, and independent replica load distribution. At cluster scale, compare aggregate output tokens/s while holding explicit p95 TTFT and p95 ITL objectives, rather than comparing raw throughput alone. A peak-bandwidth bound is not a service-level promise. Use a profiler to separate compute, HBM traffic, exposed collectives, and pipeline gaps. This synthesizes Week 3: TP splits operators, PP splits layers, CP splits context, and replicas split independent requests. The 405B model is dense, so expert parallelism is not applicable. Training-oriented FSDP/ZeRO optimizer sharding is not the default for this persistent-weight inference case. Storage units are decimal: 1 GB = 10^9 bytes. The baseline uses the canonical architecture with 8 KV heads, BF16 weights, and BF16 KV unless explicitly changed. P = 405 billion is rounded; equal weight partitioning is a planning approximation. The baseline excludes prefix sharing, CPU offload, and a speculative draft model. No GPU measurements were run.

- [Applied inference chapter](https://liltom-eth.github.io/scaling-book-pytorch/chapters/applied-inference.html)
- [Llama 3 paper §6](https://arxiv.org/html/2407.21783v3#S6)
- [Week 3: topology](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-18)
- [vLLM distributed serving](https://docs.vllm.ai/en/v0.19.1/serving/parallelism_scaling/)
