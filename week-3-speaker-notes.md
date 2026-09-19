# Week 3 speaker notes

Scaling LLMs across GPUs. Expanded coverage of data, tensor, pipeline, context, and expert parallelism, with ZeRO/FSDP state sharding. No fixed presentation duration.

45 slides. Use the Slides menu to navigate by section. The standalone HTML includes all diagrams and works offline. External reference links require internet access.

All numerical examples and diagrams are teaching examples unless explicitly stated otherwise. Questions are authored exercises, not attributed company interview reports.

## 1. Scaling LLMs across GPUs

**Section: Introduction**

This expanded Week 3 has no fixed presentation time. Read the sections in order or use the Slides menu to navigate by topic. The prerequisite is Weeks 1 and 2: GPU memory, GEMM, prefill/decode, KV cache, and online softmax. All diagrams and numerical examples are authored teaching schematics unless explicitly labeled otherwise. No example is a measured production benchmark or an attributed company interview. Sources are linked on individual slides and in these notes. No software setup is required to view the standalone HTML.


## 2. Capacity, latency, and throughput are different goals

**Section: Introduction**

Capacity, per-request latency, and service throughput lead to different deployment choices. Reuse Week 2 definitions of TTFT and ITL, including queueing when discussing client-observed measurements. More memory can support larger batches, but extra GPUs do not automatically lower latency. Performance and memory must be checked per GPU rather than only summing the cluster. We will keep these objectives visible while introducing each partition.

- [vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [Scaling Book: inference](https://jax-ml.github.io/scaling-book/inference/)

## 3. GPUs have separate memory and communicate in groups

**Section: Introduction**

A node is a physical server; a rank is a process identifier within a distributed group. This deck uses one process per GPU, but the terms are not definitions of one another. Each device owns its allocations and HBM; a model tensor is not automatically spread across all memory just because four devices are visible. A communication group specifies which ranks participate in an operation. On many accelerator servers intra-node GPU links are faster than the inter-node path, but inspect the real topology: PCIe-only nodes and different networks exist. This diagram is schematic and claims no universal link bandwidth.

- [CS336 Lecture 8](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)
- [vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

## 4. Serving replicas divide requests across model instances

**Section: Serving replicas**

Use a dense, read-only model. Both replicas start from the same trained checkpoint. The routing arrows assign whole requests, not different tokens of one request. A replica owns the KV cache of its assigned requests; the two replica caches do not automatically form one larger shared cache. More replicas can increase aggregate throughput if there is demand and the router balances it, but do not intrinsically shorten one request’s computation. A model instance may be a TP or PP group rather than a single GPU. Some MoE deployments couple data-parallel workers through expert parallelism; that is a separate configuration discussed later.

- [vLLM: data-parallel deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)

## 5. A two-layer MLP is our tensor-parallel example

**Section: Tensor parallelism**

We use a dense feedforward block with two linear maps and an elementwise GELU. The exact shapes are teaching dimensions, not a real model. X has two rows: this could be two prompt tokens or two decode requests. W1 expands width 4 to 8, H has eight intermediate features, and W2 contracts it back to 4. Bias terms are omitted to focus on partitions. Modern gated MLPs have another projection but use the same compatible intermediate-feature partition idea. Each row still belongs to its original token/request.

- [Megatron-LM, §3](https://arxiv.org/abs/1909.08053)
- [PyTorch tensor parallelism](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)

## 6. Column partition produces different output features

**Section: Tensor parallelism**

Write W1 as the horizontal concatenation [W1a W1b], where each shard is 4x4. X is already replicated across the two ranks, so each independently computes a 2x4 output. GELU is elementwise, so it may act on each output shard independently. Concatenating would recover the full H, but physical all-gather is unnecessary if the following layer can consume the partition. Column refers to W as stored in the displayed mathematical convention XW; PyTorch nn.Linear stores its weight transposed, so implementation dimension numbers differ. This deck consistently uses XW.

- [Megatron-LM, §3](https://arxiv.org/abs/1909.08053)
- [Berkeley Lecture 4](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture4.pdf)

## 7. Row partition produces partial sums

**Section: Tensor parallelism**

W2 is partitioned vertically into W2a and W2b, each 4x4. The first shard pairs with Ha and the second with Hb. The two products both have the complete output shape 2x4, but each covers only half of the reduction dimension. Matrix multiplication sums over all eight intermediate features, so the correct output is their elementwise sum. This is fundamentally different from the previous output-feature partition: concatenation there, addition here. No nonlinear function can generally be applied independently to these partial sums before reducing if the function is intended to act on their total.

- [Megatron-LM, §3](https://arxiv.org/abs/1909.08053)
- [NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

## 8. AllReduce makes the sum available on both GPUs

**Section: Tensor parallelism**

This small numeric example illustrates the semantics of AllReduce with the sum operation, independent of algorithm. Both ranks enter a matching collective in a matching order; they contribute arrays with the same shape. The output is [2+7,5+1]=[9,6] on both. NCCL and frameworks choose actual transfer algorithms. Training gradient averaging may use a sum plus division or equivalent averaging semantics; TP partial sums require addition, not an average. A collective must be called consistently by all ranks in its group to avoid hanging or incorrect behavior.

- [NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

## 9. The paired MLP avoids an intermediate gather

**Section: Tensor parallelism**

Advance the three frames. W1a and W1b own distinct intermediate columns. W2a and W2b own matching intermediate rows. Therefore each rank consumes exactly the H shard it already produced; no H all-gather is needed between the two projections. The output sum restores a replicated hidden state for the next block. This is the classic paired MLP from Megatron-LM. The diagram concerns the forward pass; backward has conjugate communication operations and is not claimed communication-free. Tensor and sequence parallel variants change activation layouts at the boundaries, introduced later.

- [Megatron-LM, §3](https://arxiv.org/abs/1909.08053)
- [PyTorch tensor parallelism](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)

## 10. Attention can partition heads across GPUs

**Section: Tensor parallelism**

For ordinary MHA, independently computed attention heads naturally map to TP ranks. Here four Q heads have four corresponding K/V heads; each GPU holds two heads and their cached history. The output projection mixes head results, so row-partitioning its input feature dimension yields partial sums that are reduced. Along with the MLP reduction, this gives two forward AllReduces per Transformer block in the classic replicated-boundary design. Models with GQA/MQA have fewer KV heads, so their cache sharding depends on implementation and TP degree; for example TP beyond the number of KV heads can replicate KV heads. This slide does not promise perfect 1/TP savings for all model tensors.

- [Megatron-LM, §3](https://arxiv.org/abs/1909.08053)
- [PyTorch tensor parallelism](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)

## 11. Small messages still pay communication latency

**Section: Communication costs**

This is a latency-plus-bandwidth teaching model, not a measured GPU specification or full collective formula. Decimal units: 10000/100e9 seconds=0.1 microseconds, 10e6/100e9=100 microseconds. The fixed 5 microseconds is assumed. For collectives, algorithm, rank count, routing, launch overhead and overlap determine the effective cost, so do not simply apply one-link numbers to an AllReduce. In small-batch decode many layers can expose small collectives with relatively little compute to amortize them. Prefill usually has more token rows and larger messages/work, changing the tradeoff.

- [Berkeley Lecture 4](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture4.pdf)
- [Scaling Book: inference](https://jax-ml.github.io/scaling-book/inference/)

## 12. Communication can consume the time saved by sharding

**Section: Communication costs**

The chart shows illustrative sequences, not data and not predictions for a particular GPU. Some local kernels shrink with more TP; actual speed depends on GEMM shape and utilization, so even local compute may scale poorly. Communication can increase with rank count and topology. Exposed communication is the part not hidden behind independent work. The next operation may depend on the complete reduced result, placing that communication directly on the critical path. The right performance model follows dependencies across kernels and transfers; adding all times double-counts overlap, while taking a global max can unrealistically assume all communication overlaps.

- [Megatron-LM, §3](https://arxiv.org/abs/1909.08053)
- [Berkeley Lecture 4](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture4.pdf)

## 13. Pipeline parallelism splits the layers

**Section: Pipeline parallelism**

The model has four layers only for illustration. The first stage computes layers 1–2 and sends their output hidden states to the second stage, which computes layers 3–4. The weights remain at their owning stage across iterations. At inference each stage also retains KV for its layers; it does not need to transfer the complete KV cache at every stage boundary. The communication payload at a boundary is typically an activation tensor, with sampling/control information handled by the serving implementation. A microbatch here contains independent examples or requests split out of a larger batch.

- [CS336 Lecture 8](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)
- [vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

## 14. Independent microbatches fill the pipeline

**Section: Pipeline parallelism**

Read each row as one GPU and each column as one equal-duration stage interval. A enters stage0 in slot1 and stage1 in slot2. While stage1 processes A, stage0 processes B. Four microbatches on two stages require five slots. Eight occupied device-slots out of ten gives 80% utilization in this idealized model. More generally m equally sized microbatches on p equally fast stages use m/(m+p−1) of total stage-time slots, with no transfer cost. Unequal layers, communication and runtime overhead lower utilization. This formula is not a general training pipeline efficiency estimate and does not guarantee the same efficiency in online autoregressive decoding.

- [CS336 Lecture 8](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)
- [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html)

## 15. One decode request still has a token dependency

**Section: Pipeline parallelism**

This diagram makes the autoregressive dependency explicit. A forward pass on the last sampled token must reach the final layer and sampler before the next token is known. Merely partitioning layers does not turn later tokens of a single sequence into independent microbatches. Multiple independent requests or suitable scheduling can keep stages busier. PP increases model capacity and can improve throughput under appropriate workloads, but must be measured for latency. Compare with Week2 PD disaggregation: prefill and decode instances each implement the full model, possibly themselves using TP or PP; the phase boundary transfers KV. PP and PD can coexist, but they split different things.

- [Scaling Book: inference](https://jax-ml.github.io/scaling-book/inference/)
- [vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

## 16. Data-parallel training must keep the model copies consistent

**Section: Data parallelism**

DDP means DistributedDataParallel. Each worker processes different examples with an identical model and optimizer state. Backward produces a local gradient; synchronizing gradients makes the following optimizer step consistent across workers. The diagram assumes equal numbers of equally weighted training targets and local mean losses. In language modeling, different valid-token counts can require weighted normalization for the intended global token-mean objective. DDP can overlap gradient communication with backward through gradient buckets; this diagram shows the logical dependency, not a serialized implementation. The input batch must be partitioned by the data pipeline.

- [PyTorch: DDP design](https://docs.pytorch.org/docs/stable/notes/ddp.html)

## 17. The optimizer update explains why DDP averages gradients

**Section: Data parallelism**

All numbers are an authored arithmetic example. The model is deliberately a two-parameter vector so the operation is visible. Local gradients [2,4] and [6,8] are supplied, not derived from a particular network. Sum and division are shown separately for clarity; a backend may combine scaling with communication. Both workers must start from equal weights and compatible optimizer state. With Adam rather than SGD, matching gradients and matching moment states similarly lead to matching updates. Unequal local loss denominators change the correct weighting; equal worker averaging is the intended objective only under the stated assumptions.

- [PyTorch: DDP design](https://docs.pytorch.org/docs/stable/notes/ddp.html)
- [NCCL: collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

## 18. ZeRO progressively shards the state that DDP replicates

**Section: Training-state sharding**

ZeRO stages progressively partition optimizer state, gradients, and parameters across the data-parallel group. Each strip represents the same parameter-index ranges, not layers. The stage-3 diagram is a persistent ownership view: it does not mean a local matrix multiplication can execute using an arbitrary quarter of its required weights. Workers reconstruct parameters when needed. This differs from tensor parallelism, which explicitly partitions the layer computation. ZeRO does not divide the local-example activation memory by the data-parallel degree. Fully sharded data parallel implementations follow a similar state-sharding principle.

- [ZeRO paper, §3](https://arxiv.org/abs/1910.02054)

## 19. Sharding the same 8B model: 128 → 56 → 44 → 32 GB per GPU

**Section: Training-state sharding**

The numbers are derived from the same illustrative state layout used in Week 1. For P=8 billion parameters, 2P bytes is 16 GB, 12P is 96 GB, and 16P is 128 GB. Dividing only optimizer state gives 16+16+24=56 GB; additionally dividing gradients gives 16+4+24=44 GB; dividing all persistent state gives 4+4+24=32 GB. This is not an exact prediction for PyTorch FSDP2: master-parameter representation, gradient precision, accumulation, buffer reuse, and allocator behavior vary. Activations scale with the local workload. Gathering and prefetching introduce temporary allocations, so a 32 GB device is not sufficient merely because this state budget equals 32 GB.

- [Ultra-Scale: ZeRO memory](https://nanotron-ultrascale-playbook.static.hf.space/index.html#zero-redundancy-optimizer-zero)

## 20. Gather weights for computation; scatter gradients for the update

**Section: Training-state sharding**

AllGather copies disjoint parameter slices into a full tensor on every participating GPU; it does not sum those slices. ReduceScatter sums matching gradient entries across workers and leaves different output slices on different workers. In this authored example, the summed vector is [6,8,10,12], with sum shards [6,8] and [10,12]. Division by two produces the displayed mean shards. The optimizer on GPU 0 needs only the first gradient shard because it owns the first parameter and optimizer-state shard; GPU 1 updates the second shard. Scaling can be implemented within or around the collective.

- [NCCL: collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [PyTorch: FSDP2 execution](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-fsdp2-works)

## 21. FSDP saves memory by gathering weights when needed

**Section: Training-state sharding**

This is a logical dependency trace for one layer, not a proportional timing chart. The example reshards weights after forward, so backward must gather them again. Real schedules can prefetch the next layer while the current layer computes, and can keep some parameters unsharded depending on configuration. Other layers remain sharded while this layer executes. Temporary full weights, full local gradients before reduction, and communication workspaces contribute to peak memory. Data-parallel workers still process distinct local data. A separate tensor-parallel group can partition the layer computation inside each data-parallel model instance.

- [PyTorch: FSDP2 execution](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-fsdp2-works)
- [Ultra-Scale: ZeRO memory](https://nanotron-ultrascale-playbook.static.hf.space/index.html#zero-redundancy-optimizer-zero)

## 22. Context parallelism splits one long sequence

**Section: Context parallelism**

Start with the workload, not the acronym. DP processes different examples; CP assigns parts of one example to different devices. The eight-position drawing is a small stand-in for a very long sequence. Weights are replicated within this pure CP group, while the positions whose activations are owned by each device differ. In a hybrid TP/PP/CP layout, CP peers hold the same TP/PP parameter slice rather than necessarily the entire model. LayerNorm and token-wise MLP computation can operate on owned token rows. Attention cannot treat those token shards as independent examples. During training, gradients for replicated parameters must also be combined across CP peers; CP is not independent unsynchronized training. Insu Jang’s supplied article gives an accessible roadmap; Megatron’s current documentation specifies the implementation scope.
Sources: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html
Background reading: https://insujang.github.io/2024-09-20/introducing-context-parallelism/

- [Megatron CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [Insu Jang: CP overview](https://insujang.github.io/2024-09-20/introducing-context-parallelism/)

## 23. Attention must still cross the token shards

**Section: Context parallelism**

Here each numbered cell stands for both the key and value belonging to that original token position, for one attention head. The query q6 resides on GPU 1. The correct causal key set includes positions 1 through 6, including q6’s own position. Positions 7 and 8 must not contribute even if their KV data happen to be on the same GPU. Position ownership is an implementation detail; causal visibility follows original positions. The complete softmax normalization spans the union of allowed keys. Attention restricted to the GPU-local chunk is a different sparse-attention computation, not exact context parallelism. Ring exchange and Ulysses are two ways to satisfy this dependency.
Sources: https://arxiv.org/abs/2310.01889
https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html

- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [Megatron CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)

## 24. Ring attention keeps Q local and circulates KV

**Section: Context parallelism**

Start at local computation, then press Next step twice. The first frame is the diagonal pair of causal attention blocks. In the second, GPU 0 receives only future keys and can skip their computation; GPU 1 receives past keys and adds a second contribution. A two-GPU ring has one remote shard per device. General P-GPU rings process local KV plus P−1 remote shards, with causal optimizations able to skip fully masked work. This is not a claim that both GPUs perform equal work in the toy. The query shard and its running per-query statistics stay local while KV communication uses temporary buffers. Original owned KV can be retained or recomputed according to the implementation and training policy. Incoming KV need not accumulate into a full-sequence resident tensor. The final output stays token-partitioned. Real ring implementations may overlap communication with attention, but useful overlap is conditional on compute per block, transfer time, and scheduling.
Source: https://arxiv.org/abs/2310.01889

- [Ring Attention, §3](https://arxiv.org/abs/2310.01889)

## 25. Merge summaries, not local softmax outputs

**Section: Context parallelism**

The two sets partition the visible keys of one query, not two independent queries. Define scores including the usual head-dimension scaling. m is a stability reference; ℓ is the normalizer in that reference; u is the unnormalized weighted sum of value vectors. Each set has initially used a different maximum, so its ℓ and u must be multiplied by exp(old maximum − new maximum) before addition. Both the numerator and denominator must be rescaled. A plain arithmetic average of O_A and O_B is generally wrong because the two sets have different total probability mass. The merged output equals dense attention in exact arithmetic, with floating-point-order differences in practice. Fully masked sets contribute zero and should be skipped rather than evaluating undefined −infinity differences. This algebra is an original derivation of the online-softmax reduction used in memory-efficient attention and ring implementations.
Sources: https://arxiv.org/abs/2205.14135
https://arxiv.org/abs/2310.01889

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Ring Attention](https://arxiv.org/abs/2310.01889)

## 26. Balance causal work across the GPUs

**Section: Context parallelism**

This is an original eight-position, four-GPU arithmetic example. Contiguous pairs of query positions have valid-key counts 1+2=3, 3+4=7, 5+6=11, 7+8=15. Assigning mirrored pairs gives 1+8=2+7=3+6=4+5=9. There are still 8×9/2=36 allowed attention interactions. The example illustrates zigzag-style balanced chunk assignment: with larger sequences one pairs early and late chunks, not necessarily individual tokens. This is related to, but not identical to, Striped Attention’s uniform interleaving of tokens. The source paper establishes the causal load-imbalance motivation; Megatron exposes contiguous-to-zigzag layout conversion. Preserve original absolute/rotary positions and sequence boundaries. Communication kernels must also know those positions so that future tokens are masked even when storage order is permuted. Equal pair counts do not guarantee equal wall time; block geometry matters.
Sources: https://arxiv.org/abs/2311.09431
https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.context_parallel.layout.html

- [Striped Attention](https://arxiv.org/abs/2311.09431)
- [Megatron zigzag layout](https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.context_parallel.layout.html)

## 27. Ulysses trades token shards for head shards

**Section: Context parallelism**

Read this original diagram left to right. Initially each device has four token rows and all four heads for those rows. GPU 0 keeps its head-0/1 slices and sends head-2/3 slices to GPU 1, while GPU 1 sends its head-0/1 slices to GPU 0 and keeps head-2/3. After the All-to-All, each device has eight token rows but only two heads: total per-device Q/K/V element count is unchanged. Each local head can then run a normal full-context attention kernel, including FlashAttention. A second All-to-All on attention outputs restores four token rows and all four heads to each original owner. This is activation-layout redistribution, not weight tensor parallelism: pure Ulysses keeps projection and MLP weights replicated. For the toy, Q/K/V all have four heads (MHA). GQA/MQA and combinations with TP require care about available KV-head partitions, replication, or specialized schemes. Do not claim arbitrary CP degree is possible or that Ulysses always outperforms ring attention.
Source: https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md
Additional walkthrough: https://insujang.github.io/2024-09-20/introducing-context-parallelism/

- [DeepSpeed Ulysses](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md)
- [Insu Jang: Ulysses](https://insujang.github.io/2024-09-20/introducing-context-parallelism/)

## 28. Context parallelism and Megatron SP differ

**Section: Context parallelism**

This is a terminology clarification after teaching the concrete mechanisms. In Megatron’s TP-associated SP, selected activations outside tensor-parallel linear regions are sequence-partitioned. A typical forward region gathers token rows before TP computation and returns a sequence shard with ReduceScatter instead of leaving an AllReduce-replicated activation. CP retains token ownership across the network and introduces distributed attention to satisfy remote-key dependencies. Do not generalize the label SP to every paper: Ulysses is explicitly called sequence parallelism by DeepSpeed, and older literature also uses SP for long-sequence attention distribution. CP and TP-associated SP can coexist. The ownership and exchange diagrams are more reliable than an acronym alone.
Sources: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html
https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md

- [Megatron CP vs SP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [DeepSpeed Ulysses](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md)

## 29. Decode parallelism shards the cached history

**Section: Context parallelism**

Return to Week 2’s distinction: prefill supplies many new query rows, while ordinary autoregressive decode supplies one per active request. For the diagram, q is a query whose visible history is positions 1–8; the current token’s own K/V may be included among these eight depending on where the step is drawn. Each shard computes a partial attention summary, and the global output follows the same max-rescaled merge as the earlier slide. A production implementation may exchange output and log-sum-exp rather than literal m,ℓ,u triples; these encode equivalent normalizer information. The drawing explains the dependency, not a prescribed NCCL sequence. vLLM separates prefill and decode context parallelism and can use DCP inside an existing TP group to reduce KV duplication; therefore DCP need not multiply the total GPU count. Explain the runtime’s particular rank layout before writing a GPU-count product. Sharding stored KV does not automatically speed up weight projections or the MLP, and communication may exceed the saved KV-read time at short contexts.
Sources: https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/
https://vllm.ai/blog/2026-08-07-decode-context-parallelism

- [vLLM context parallelism](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)
- [vLLM DCP design](https://vllm.ai/blog/2026-08-07-decode-context-parallelism)

## 30. What context parallelism saves

**Section: Context parallelism**

This closes the section by connecting the mechanism to resource accounting. For fixed global work and a balanced pure-CP partition, the primary owned token activations scale approximately as S/P per rank; peak memory also includes attention statistics, communication/double buffers, backward saved tensors, parameters, and optimizer state. Do not promise total allocated HBM falls exactly by P. Dense causal attention still has S(S+1)/2 valid query-key pairs per head across all ranks; distributing them does not make the mathematical workload linear. FlashAttention has already removed the need to keep an S×S score/probability tensor in HBM, so CP should not be motivated only by splitting that materialized tensor. Pure CP replicates weights; combine FSDP/ZeRO, TP or PP for parameter/state capacity. Long blocks can provide enough attention computation to overlap ring transfer; shorter blocks, weak links or imbalance may expose communication. Show memory, kernel time and collective time in measurement before assuming additional CP improves latency.
Sources: https://arxiv.org/abs/2310.01889
https://arxiv.org/abs/2205.14135
https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html

- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [FlashAttention](https://arxiv.org/abs/2205.14135)

## 31. MoE: choose a few expert MLPs for each token

**Section: Expert parallelism**

An expert is a learned feed-forward network with its own weight matrices. The router reads a token hidden vector and selects a small subset of experts. The selected experts each map that hidden vector to an output vector of the same width, which is weighted and summed. This slide uses a four-expert, top-2 teaching model and weights normalized over the selected experts. It does not prescribe a universal gate normalization: models can use different scoring, scaling, shared-expert, and routing designs. The attention sublayer is omitted here, not removed from the Transformer. Sparse activation means few expert MLPs run per token; it does not mean the unselected experts have no weights or need no storage. Total parameters matter for storage while active parameters help describe arithmetic per token. The specific values and original schematic are illustrative, not a measurement.

- [Mixtral §2](https://arxiv.org/abs/2401.04088)

## 32. Expert parallelism places different experts on different GPUs

**Section: Expert parallelism**

Expert ownership is the new partition axis. In pure two-way EP, GPU 0 owns E0 and E1 and GPU 1 owns E2 and E3. A selected expert can run locally on its owner because its complete matrices are present. In expert TP, each GPU owns a shard of every expert’s matrices; computing a selected expert requires the tensor-parallel collaboration introduced earlier. A real system can combine expert TP with EP, so these are illustrative alternatives rather than mutually exclusive deployment modes. Unselected weights remain resident in this example; there is no weight offload or on-demand model loading. Placement is repeated independently for each MoE layer.

- [vLLM: expert parallelism](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- [DeepSpeed MoE](https://www.deepspeed.ai/tutorials/mixture-of-experts/)

## 33. Follow four tokens through a two-GPU MoE layer

**Section: Expert parallelism**

Use exactly this routing table on the next slide. A and B start on GPU 0, C and D start on GPU 1. E0 and E1 live on GPU 0, E2 and E3 live on GPU 1. Token A therefore runs E0 locally and sends its hidden vector to E2 remotely. Repeat the reasoning for each row. Expert batches are E0:[A,C], E1:[B,D], E2:[A,C], E3:[B,D]. There are eight token-expert assignments but only four distinct input tokens. With this placement every token has one local expert and one remote expert. Router weights are supplied, sum to one within each row, and belong to the token, not the GPU. All weights, placement, and token identifiers are original teaching examples. A–D need not be consecutive positions in a shared sequence.

- [Megatron: token dispatch](https://docs.nvidia.com/megatron-core/developer-guide/0.19.0/apidocs/core/core.transformer.moe.token_dispatcher.html)

## 34. Dispatch → expert computation → return and combine

**Section: Expert parallelism**

Interactive sequence using the previous routing table. Step 1: GPU 0 sends A and B to GPU 1 while GPU 1 sends C and D to GPU 0. Local routes do not require crossing the network. More generally each source packs separate activation subsets for each destination, and counts can differ; this is the all-to-all communication pattern. It is not an AllReduce, because the destinations need different token subsets rather than a common sum. Step 2: each expert processes its assigned tokens, normally grouped to make its MLP matrix multiplications more efficient. Step 3: expert outputs return to the original token owners and are unpermuted into token order. Token A has E0(A)=[2,4] and E2(A)=[10,0], chosen illustrative expert outputs. Its given gate weights produce [0.75*2+0.25*10,0.75*4+0.25*0]=[4,3]. The first component is 4, not 12: combining is a router-weighted vector sum. The other tokens are combined with their own weights. The explicit sum illustrates semantics; implementations may fuse weighting with expert kernels or communication. The Megatron all-to-all dispatcher is one concrete design; all-gather/reduce-scatter and specialized fused backends also exist. No model weights or KV cache are migrated in this example.

- [Megatron: token dispatch](https://docs.nvidia.com/megatron-core/developer-guide/0.19.0/apidocs/core/core.transformer.moe.token_dispatcher.html)

## 35. Routing imbalance can leave some GPUs waiting

**Section: Expert parallelism**

Expert count is a capacity allocation, while routed token count is a workload allocation. The balanced case is the previous table: each expert has two token assignments, giving four per GPU. In the skewed example A and B both select E0/E1, C selects E0/E2, D selects E1/E3. The counts are [3,3,1,1], hence GPU totals [6,2], with eight assignments in both examples. Assuming equally costly experts makes the imbalance visible, but runtime is not simply proportional to counts: GEMM shapes, batching, overlap, kernels, and network costs matter. vLLM monitors load and can adjust expert placement and use redundant replicas. Replication uses extra weight memory and placement changes have cost. Training can use load-balancing objectives; inference cannot simply reroute every token to an arbitrary different expert without changing the model result. Capacity limits are an implementation/model policy: dropping, buffering, padding, or dropless dispatch must be specified, not silently assumed. Very small token groups also give small expert GEMMs and may expose communication latency even when load is balanced.

- [vLLM: expert parallelism](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)

## 36. Different layers can use different parallelism strategies

**Section: Expert parallelism**

This connects EP to the earlier distinction between request replicas and cooperating GPUs. In this particular deployment, attention weights are replicated across two workers, each worker attends over only its own requests, and each maintains independent request KV storage. The expert weights are distributed across both GPUs, so every MoE layer can exchange token activations between those workers. The entire models are therefore not independent dense serving replicas: even a worker with no local requests may need to participate in the coordinated expert computation. The data-parallel label describes attention ownership in this example; it does not imply gradient synchronization during inference. Router and other small shared weights can be replicated; they are omitted visually. TP or CP can further shard attention and KV, but neither is enabled in this schematic. This is supported by vLLM’s DP-attention plus EP-expert deployment description, rather than a universal claim about every MoE inference runtime.

- [vLLM: expert parallelism](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- [vLLM: data parallelism](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)

## 37. Parallelism choices change different tensor dimensions

**Section: Putting it together**

This is a recap after the worked examples, not a replacement for them. Serving replicas are a form of data-parallel inference but do not perform training gradient synchronization. ZeRO/FSDP refer to state sharding along data-parallel groups, while TP changes how layer arithmetic is divided. CP and EP have implementation-specific communication paths explained in their sections. These dimensions can be combined, but group membership must be defined explicitly and the total GPU product depends on how the groups overlap.

- [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html)
- [CS336 Lecture 8](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

## 38. A deployment combines several communication groups

**Section: Putting it together**

Two replicas process independent request batches. Each replica has two pipeline stages and each stage uses two TP ranks: ranks0/1 and2/3 in replica0, ranks4/5 and6/7 in replica1. TP reductions happen only within a stage’s pair; stage-boundary activations transfer between compatible ranks in consecutive stages. There is no training gradient synchronization between these serving replicas. The simple product 2x2x2=8 applies to these explicitly independent axes. Do not blindly multiply every named parallel degree in a real MoE setup: EP groups can be organized from an existing data-parallel axis, and framework constraints matter.

- [vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html)

## 39. Four GPUs offer several serving layouts

**Section: Putting it together**

The same hypothetical 8B dense model has 16 GB of BF16 weights. Four independent replicas store 64 GB of aggregate weights; two TP2 instances store 32 GB; one TP4 instance stores16GB assuming ideal sharding of all counted weights. Real implementations replicate some tensors and add buffers. Although replication can improve throughput by serving independent requests, it consumes aggregate memory that could otherwise serve KV. The KV distribution itself depends on model and parallel scheme. Compare fixed model/precision, input-output lengths, offered arrival rate and latency targets. Avoid comparing one instance’s tokens/sec with an entire cluster’s rate. No layout is declared universally best.

- [vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [Scaling Book: inference](https://jax-ml.github.io/scaling-book/inference/)

## 40. Question 1: Complete a tensor-parallel MLP

**Section: Questions**

Authored interview-style exercise grounded in the Megatron-LM tensor partition, not a reported company question. Assume F is divisible by2, biases omitted, the same X is available on both ranks, and elementwise GELU. Ask the audience to label shapes before naming a collective. GPU0 computes Ua=Ha@W2a and GPU1 Ub=Hb@W2b, both [M,D]. Their sum isY. Keeping Ha/Hb partitioned eliminates an intermediate gather. If the first projection splits the reduction/input dimension, the partial X shards times W1 shards must be summed before GELU because GELU(a+b) is generally not GELU(a)+GELU(b). The next slide gives the answer.

- [Megatron-LM, §3](https://arxiv.org/abs/1909.08053)

## 41. Solution 1: Local products, then a sum AllReduce

**Section: Questions**

The input is replicated and each local H contains different output features, so GELU works locally. The second projection contracts the feature dimension: contributions add. An all-gather of Y_partial would concatenate redundant-shaped partial results rather than complete them. Output bias must not be added identically on both ranks before a sum, as that would add it twice. First-layer bias can be column-sharded with W1. Correctness tests should use numerical tolerances because floating-point reduction order changes. The pure-PyTorch logical-rank take-home later allows testing this on one device.

- [Megatron-LM, §3](https://arxiv.org/abs/1909.08053)
- [PyTorch tensor parallelism](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)

## 42. Question 2: Why does TP = 4 barely improve decode?

**Section: Questions**

This is an authored diagnostic question using explicitly hypothetical times, not benchmark evidence. Same per-instance batch8 and context are used to diagnose TP scaling. Local work improves by6ms while exposed communication adds4ms, leaving a2ms end-to-end gain. Ask why communication might rise: rank count, topology, synchronization, contention, collective implementation. Ask whether a throughput conclusion can be drawn under the same total arrival process. Two TP2 instances could collectively process twice as many requests at this particular per-instance batch, but that is a saturated upper-style comparison and does not establish online SLO capacity. TP4 at batch16 has not been measured. The solution distinguishes these comparisons.

- [Scaling Book: inference](https://jax-ml.github.io/scaling-book/inference/)
- [vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

## 43. Solution 2: Separate local speedup from serving capacity

**Section: Questions**

The given arithmetic identifies exposed communication as consuming most of the local benefit, but not its underlying reason. Trace matching collectives on all ranks and inspect skew, launch and network paths. Compare nodes and intra-node layouts before blaming a specific fabric. For the given fixed-batch numbers two saturated TP2 groups would yield2*8/0.020=800 tokens/s, while oneTP4 group atB8 gives8/0.018≈444; those configurations admit different total active requests and are not a complete online comparison. TP4 atB16 or another workload could behave differently. The correct service test fixes the request distribution and offered arrivals, increases load, and compares throughput within the desired TTFT/ITL percentiles and capacity.

- [Scaling Book: inference](https://jax-ml.github.io/scaling-book/inference/)
- [vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

## 44. Take-home: Simulate two tensor-parallel ranks

**Section: Take-home**

The exercise extends Question1 into executable code. First use one process and torch tensors to simulate logical ranks, making the data distribution explicit without needing a multi-GPU notebook. Use casesM=2,D=4,F=8;M=7,D=6,F=10;M=1,D=8,F=12 and a nonzero bias. Then, optionally, implement each local path under torch.distributed across two GPUs. A simulation timing is not a distributed performance measurement. The following slide supplies a complete concise correctness reference and links to the official TP tutorial.

- [PyTorch tensor parallelism](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)

## 45. Take-home solution: PyTorch reference

**Section: Take-home**

Complete the visible code by initializing X[M,D],W1[D,F],W2[F,D],b1[F],b2[D] with matching floating dtype and device. For example set a deterministic seed and use torch.randn with float64 on CPU for a precise correctness test. In the real distributed version, each rank owns one local W1/b1/W2 shard, the sum becomes all_reduce(SUM), and each rank adds b2 once after the reduction. The function shown uses temporary tensors on a single device and intentionally does not claim distributed speedup. This references the official PyTorch TP tutorial rather than a proprietary interview solution.

- [PyTorch tensor parallelism](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)
