# Week 3 visual overview — speaker notes

A visual introduction to data, tensor, pipeline, context, and expert parallelism, plus ZeRO/FSDP. The [50-slide reference](https://zhiweixx.github.io/llm-systems-study-group/week-3/slides.html) retains the detailed derivations and exercises.

21 slides. Use the Slides menu to navigate by section. The standalone HTML includes all diagrams and works offline. External reference links require internet access.

Diagrams and toy examples show tensor ownership and communication. Sizes and timings are schematic unless stated otherwise.

## 1. Parallelism and sharding

**Section: Introduction**

This is the 21-slide visual overview of Week 3. The original 50-slide lecture remains available as the detailed reference. We will distinguish three questions throughout: what is copied on multiple GPUs, what is partitioned across them, and what communication is needed to recover the intended computation. Boxes and colors indicate ownership or computation, not measured physical sizes or performance. The audience should know a Transformer forward pass and the basic GPU memory and inference concepts from Weeks 1 and 2. Data parallelism, tensor parallelism, pipeline parallelism, context parallelism, expert parallelism, and training-state sharding solve different parts of the scaling problem.


## 2. Separate memories, connected GPUs

**Section: Introduction**

A node is a server. This teaching setup uses one process per GPU, but a rank is a process identifier in a distributed group rather than the hardware itself. Every GPU has its own memory allocations. A replica is a complete copy of the object being replicated; a shard is a subset of it. The two blue rows are identical copies, while the blue and teal pieces are different portions that together form one tensor. These are schematic tensor diagrams, not capacity bars. The links inside a server often provide more bandwidth and lower latency than links between servers, but this depends on the actual hardware. The later placement slide uses that assumption explicitly. A group may span nodes, and there is no universal prohibition against tensor, context, or expert parallelism crossing a server boundary.

- [vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

## 3. Data-parallel training (DDP)

**Section: Data parallelism**

Read the diagram from the local examples at the top to the matching updated weights at the bottom. GPU 0 receives examples A and B, while GPU 1 receives C and D. Each worker has a complete copy of the same weights and runs its own forward and backward computation. Their gradients differ because their examples differ. The average gradient is then available to both workers, so identical initial parameters and optimizer state produce identical updates. An AllReduce with sum followed by division by two implements this average; the diagram shows its semantics, not a required implementation order. Equal local example weighting and local mean losses are assumed. Different valid-token counts in language modeling may require weighting by those counts to represent the desired global token-mean loss. DDP does not partition the model weights across workers. Production implementations can overlap gradient communication with backward through gradient buckets. The boxes show ownership and dependencies rather than elapsed time.

- [PyTorch DDP](https://docs.pytorch.org/docs/stable/notes/ddp.html)

## 4. Data parallelism for serving

**Section: Serving replicas**

Both GPUs load the same trained dense-model checkpoint. The dispatcher assigns whole requests A and B to GPU 0 and C and D to GPU 1. Each worker then performs prefill and decode for its own requests, maintaining their request-specific KV caches. These are independent model instances: inference does not compute training gradients or synchronize those gradients after a token. Additional replicas can increase aggregate service capacity when there is sufficient traffic and balanced routing, but they do not divide a single request’s layer computation across the two GPUs. Each replica stores a full weight copy, so replication uses more aggregate weight memory. In a larger deployment, one model instance could itself be a tensor- or pipeline-parallel group. The figure deliberately assumes ordinary dense inference. MoE deployments can instead replicate attention while distributing experts across workers, which creates communication between otherwise separate request groups. That expert-parallel arrangement appears later in the overview.

- [vLLM: data-parallel serving](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)

## 5. ZeRO: less replicated training state

**Section: Training-state sharding**

ZeRO means Zero Redundancy Optimizer. Each horizontal group of four small blocks represents four disjoint parameter-index ranges. Read a row across all four GPUs. In DDP, every GPU retains all weight, gradient, and optimizer ranges. In ZeRO-1, each GPU retains only its assigned optimizer range. ZeRO-2 also distributes the gradient ranges. ZeRO-3 distributes the parameter ranges as well. Pale filled blocks are local allocations; white outlines represent ranges held by other workers. Every sharded state collectively remains complete across the group. The blocks have equal schematic size to show ownership, not bytes: actual optimizer state can occupy much more memory than low-precision parameters. The view describes persistent state between computations, not the instantaneous memory peak. Gathering parameters, holding activations, and communicating gradients can need additional allocations. The examples processed by each data-parallel worker still differ. FSDP on the next slide shows how a worker can execute a layer when that layer’s weights are persistently distributed.

- [ZeRO paper](https://arxiv.org/abs/1910.02054)

## 6. FSDP gathers one layer at a time

**Section: Training-state sharding**

The two colored blocks are parameter shards W0 and W1 of one layer, not two entire model layers. GPU 0 owns W0 and GPU 1 owns W1. AllGather reconstructs this layer’s full weights on both workers without adding the values together. Each then runs the layer on different local examples, as in data-parallel training. In the depicted configuration, workers release the full gathered copy after forward and retain only their owned shard. Backward therefore gathers this layer again, computes local gradients, and uses ReduceScatter to combine gradient contributions while retaining a different gradient shard on each worker. Each optimizer updates its corresponding parameter shard and optimizer-state shard. This logical trace omits other layers and communication overlap. Frameworks can prefetch another layer or retain selected full tensors, trading extra live memory for less exposed communication. The diagram explains why sharded persistent storage reduces memory but does not equal peak memory: live activations and temporary gathered tensors still occupy space.

- [PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-fsdp2-works)
- [NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

## 7. TP: split the output features

**Section: Tensor parallelism**

The input X has two token rows and four features. The first MLP matrix W1 has shape [4,8]. Split its eight output columns into W1a and W1b, each [4,4]. Both GPUs have the same X. Each produces four of the eight intermediate features, and the elementwise GELU can be applied locally to each shard. Conceptually H is the concatenation of Ha and Hb; no physical concatenation is needed if the next operation consumes the existing partitions. In this diagram the blue and teal cells denote different feature ownership, while pale inputs are replicated. The matrix convention is XW; PyTorch Linear stores its weight transposed. Biases and gated-MLP variants are omitted to isolate the partition.

- [Megatron-LM](https://arxiv.org/abs/1909.08053)

## 8. TP: sum the contributions to the output

**Section: Tensor parallelism**

W2 has shape [8,4]. Partition its eight input-feature rows into W2a and W2b, each [4,4], matching the intermediate features retained by the corresponding GPU. HaW2a and HbW2b both have the full output shape [2,4], but each contains only part of the sum over eight input features. Therefore Y is their elementwise sum, not their concatenation. AllReduce with sum makes that result available on every participating GPU. The first MLP projection did not require gathering the intermediate tensor; the second restores the replicated output by summation. This is the forward path only. The backward pass also has communication. The boxes describe tensor ownership, not actual kernel boundaries.

- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

## 9. TP can also split attention heads

**Section: Tensor parallelism**

For ordinary multi-head attention in this example, four query heads have four corresponding KV heads. Each GPU computes two heads and stores the KV history for those heads. Both still process the same token positions. The output projection mixes the head outputs; partitioning its input dimension produces partial sums that need to be added. This is the same structure as the row-partitioned second MLP projection. Grouped-query or multi-query attention has fewer KV heads, so not every TP degree produces an equal nonreplicated KV partition. Some implementations replicate KV heads when the TP degree is too large. No general 1/TP KV saving is implied. TP reduces each rank’s weight/computation share but introduces repeated communication within the model path.

- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

## 10. PP: split the model by layers

**Section: Pipeline parallelism**

This toy model has six layers and two pipeline stages. The first stage owns layers 1–3 and sends hidden states to the second stage for layers 4–6. The weights remain with their stage. During autoregressive inference, each stage also retains the KV cache for its own layers; the entire KV history is not transferred at every stage boundary. Training adds the reverse flow of activation gradients. A stage can comprise more than one GPU if combined with TP, introduced later. A microbatch is a subset of examples processed through this full path. Pipeline parallelism divides model layers; prefill–decode disaggregation instead separates two execution phases of inference, so they should not be conflated.

- [vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [Megatron: pipeline schedules](https://arxiv.org/abs/2104.04473)

## 11. PP: overlap work with microbatches

**Section: Pipeline parallelism**

Read each horizontal row as one GPU and columns as equal stage-time intervals. With one microbatch A, GPU1 waits for GPU0; then GPU0 waits while GPU1 completes A. Two of four GPU-time slots contain work. With four independent microbatches A–D, GPU0 can process B while GPU1 handles A. Eight of ten slots contain work. This is a forward-only toy with equal stage speeds and no communication cost, not a measured utilization. In this ideal model the active share is m/(m+p−1). Training schedules change the timeline and memory requirements. Microbatch count and microbatch size are different: at fixed global batch and DP degree, smaller microbatches increase their count but may produce less efficient GEMMs. Megatron, PyTorch pipelining, and DeepSpeed support pipeline microbatches. One autoregressive request cannot treat its unknown future tokens as independent microbatches; inference needs independent requests or batches to overlap stages.

- [Megatron: pipeline schedules](https://arxiv.org/abs/2104.04473)

## 12. Context parallelism divides one long sequence

**Section: Context and expert parallelism**

Begin with the workload: a single long sequence can create too much activation or KV storage for one device. CP divides its token positions among cooperating GPUs; these are not independent requests. The eight-position schematic represents a much longer sequence. In pure CP the model weights are replicated. In combined TP/PP/CP, CP peers instead hold the same TP/PP parameter slice. Local token-wise operations can run on owned rows, but attention requires remote key/value information. DP divides independent training examples or requests rather than cutting a single sequence. No batch-size or parameter-memory speedup is implied by this toy. Figure is an original teaching diagram.

- [Megatron CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [Ring Attention](https://arxiv.org/abs/2310.01889)

## 13. Ring attention brings remote keys to each query

**Section: Context and expert parallelism**

This two-frame schematic uses the same ownership as the preceding slide. Frame 1: GPU 0 queries positions 1–4 against its local keys, preserving causal visibility; GPU 1 queries positions 5–8 against its local keys. Each query has its own weighted sum and softmax normalizer. Frame 2: the KV shards are exchanged while Q and accumulated statistics remain on their owners. GPU 0 can skip the received future-key block. GPU 1 adds the past-key block. The two contributions use one global normalizer, with stable maximum-rescaling as taught in Week 2; independently normalized outputs must not be averaged. Causal masking is based on original positions, not physical storage order. With more GPUs the ring repeats exchanges until every query has processed every permitted key block. Real implementations may skip masked computation and overlap transfer with computation; the figure does not promise perfect balance or hidden communication. No full score matrix is materialized. The two-GPU exchange arrows depict KV movement, not movement of model weights.

- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [Megatron CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)

## 14. CP and Megatron SP shard different operations

**Section: Context and expert parallelism**

Read each row as the same simplified Transformer fragment, not as a full layer specification. The original model may have pre-norm, residual paths, and multiple normalization operations; those details are omitted to focus on activation ownership. G0 and G1 mean GPU 0 and GPU 1. Upper row: pure CP retains token ownership across attention and token-wise operators; attention exchanges remote KV to satisfy dependencies. Lower row: Megatron’s TP-associated SP uses token partitions for token-wise operations such as normalization and dropout. Inside TP attention and linear regions, GPUs process all token positions but different heads or matrix shards. The two “head shard” labels denote different head subsets, and the weight shards also differ. A typical forward transition from TP computation to the SP norm region uses ReduceScatter, then AllGather before the next TP region. This is not a claim that every activation in TP is full-width, nor that all methods named sequence parallelism behave this way. DeepSpeed Ulysses uses the same name for another mechanism. CP and TP-associated SP can coexist; the separate rows isolate their roles.

- [Megatron CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)

## 15. Route tokens to experts, then return the outputs

**Section: Context and expert parallelism**

An MoE expert is a complete learned MLP, not an attention head or a GPU. This overview chooses top-1 routing so each token has one destination and one returned vector; a top-k model duplicates token work across selected experts and combines their outputs with router weights. Here both original tokens A and B are owned by GPU 0. GPU 0 also holds experts E0/E1, while GPU 1 holds E2/E3. The router chooses E0 for B and E2 for A. B’s expert execution is local. A’s activation crosses to GPU 1, is processed by E2, and its resulting vector returns to GPU 0. Both outputs are restored to the original token order. E1 and E3 receive no tokens in this step but their weights remain allocated. Dispatch and return can be implemented with all-to-all style exchanges; physical network traffic is needed only for remote routes. The paths out of the expert boxes are return paths to the original worker, not a transfer of expert weights or KV histories. Router weights are omitted in this top-1 ownership illustration.

- [MoE architecture](https://arxiv.org/abs/2401.04088)
- [Megatron token dispatch](https://docs.nvidia.com/megatron-core/developer-guide/0.19.0/apidocs/core/core.transformer.moe.token_dispatcher.html)

## 16. Uneven routing leaves some GPUs waiting

**Section: Context and expert parallelism**

Continue the top-1, four-expert placement from the previous slide, but with four token activations. Assign A/B/C to E0 and D to E2; E1 and E3 are idle. GPU 0 receives three token-expert assignments while GPU 1 receives one. The bars are a schematic of equal token work, not a measured timeline or an assertion that production kernels execute tokens one by one. Real expert MLPs batch token rows into GEMMs, and runtime depends on their shapes and efficiency. The waiting region illustrates a synchronized dispatch/compute/return schedule whose completion depends on the busiest rank. Useful overlap and asynchronous implementations may change the visible wait but do not remove the unequal work. Communication can dominate even with balanced routes when per-expert batches are small. Changing placement or adding replicas of a hot expert can reduce imbalance at the cost of memory and routing complexity. Arbitrarily choosing a different trained expert just to balance work changes model semantics and is not a generally valid inference optimization. Diagnose expert token counts and communication time rather than assuming evenly sized experts imply evenly used GPUs.

- [vLLM expert parallelism](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)

## 17. Combine groups to build one GPU layout

**Section: Choosing a layout**

This diagram combines three dimensions without introducing another parallelism algorithm. Each row is one logical model replica spread across four GPUs. Within that row, the first two GPUs own the early layers and the second pair own the later layers. Within each pair, tensor parallelism partitions the weights and computation inside those layers, with the required tensor collectives. Hidden states flow from pipeline stage zero to stage one for the same inputs. The second replica uses an identical model partition but different inputs. For ordinary replicated training DP, corresponding weight shards synchronize gradients: for example, GPU zero with GPU four, rather than reducing all eight GPUs as though they held the same weights. Independent serving replicas instead accept separate requests and do not perform training gradient synchronization. The diagram shows logical groups, not physical nodes; deciding where those groups should run is the next step. Communication arrows describe dependencies, not a measured communication schedule.

- [Megatron-LM](https://arxiv.org/abs/2104.04473)
- [vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

## 18. Choose placement from the communication pattern

**Section: Choosing a layout**

The example places a two-way TP group inside each node and pipeline boundaries between nodes. This is a common starting configuration when local GPU links are substantially faster than the inter-node network. TP frequently communicates within a Transformer layer, so exposed collective latency and bandwidth can directly delay the next operation. PP sends hidden states between stages, and corresponding activation gradients during training. Ordinary training DP replicates the model partition and synchronizes gradients; some communication can overlap backward work. Sharded DP such as FSDP also gathers parameters, so it deserves a separate communication analysis. CP and EP are not limited to a node: their attention exchanges or token routing can use inter-node networks when the implementation and topology support it. This schematic does not require that a physical server boundary equal a fast-interconnect boundary. Benchmark the actual network, message sizes, overlap, and workload instead of treating the acronym as a placement rule.

- [vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html)

## 19. Communication can offset the benefit of more GPUs

**Section: Choosing a layout**

The bars are a qualitative illustration, not benchmark measurements or a prediction for a particular model. The blue section represents local computation on the critical path. The warm section represents communication and waiting that remain exposed after any overlap. Dividing a layer among more GPUs can reduce local work, but the reduction need not be proportional: smaller GEMMs may run less efficiently, and communication startup does not shrink with tensor size. A larger group or a slower network boundary can increase communication cost. The drawn result shows only a small end-to-end improvement despite visibly less local computation. Other workloads can show strong scaling, no improvement, or a regression. Compare the same model, precision, batch, context length, and objective, and inspect exposed communication rather than summing every communication operation regardless of overlap. A configuration that improves single-request latency may still reduce throughput for a fixed total GPU budget by using fewer independent replicas.

- [Scaling Book: inference](https://jax-ml.github.io/scaling-book/inference/)
- [vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

## 20. Choose a degree that fits the model

**Section: Choosing a layout**

A parallel degree counts the GPUs or ranks in that communication group. These simple examples separate tensor divisibility from the belief that a degree must always be even or a power of two. Eight query heads can be partitioned into four equal groups of two. Trying the same equal-head scheme on three ranks leaves two heads unassigned. Other model dimensions and KV layouts impose further requirements. On the right, nine positions can be partitioned into three contiguous groups of three, so the basic CP partition itself does not require an even number of ranks. This is a generic equal-token partition, not a claim that a specific Megatron configuration accepts sequence length nine. For example, Megatron’s balanced causal layout commonly uses 2*CP chunks and therefore imposes a different divisibility check. Head-redistribution algorithms may have head-count restrictions too. Backend support, layout, GPU count, and kernel constraints must all be checked. The full reference lecture gives the implementation-specific examples.

- [Megatron: TP constraints](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py)
- [Megatron: CP constraints](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py)

## 21. Six methods: tensor ownership and communication

**Section: Choosing a layout**

Use this final table as a compact map rather than six separate prescriptions. DP separates inputs while replicating the logical model; the gradient entry refers to synchronized training, not independent inference replicas. TP divides work inside layers, producing activation shards or partial sums that require the appropriate collective. PP divides layer ownership and sends hidden states forward and activation gradients backward. CP divides one sequence and exchanges attention information; the exact payload can be KV chunks, head-transposed tensors, or partial attention results depending on the algorithm and phase. EP divides expert ownership and moves selected token activations to experts before returning and combining outputs. ZeRO and FSDP divide training state: optimizer state, gradients, and possibly parameters depending on the stage and implementation. Their collectives therefore differ. These axes can be combined, but their group sizes, memory savings, and communication costs must be evaluated in the resulting layout. The small drawings recall the partition patterns; they are not exact tensor shapes.

- [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html)
- [Megatron-LM](https://arxiv.org/abs/2104.04473)
