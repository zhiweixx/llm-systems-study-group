"""Week 3 foundations, tensor/pipeline parallelism, and synthesis."""
from .common import *
MEG=('Megatron-LM, §3','https://arxiv.org/abs/1909.08053')
TPDOC=('PyTorch tensor parallelism','https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html')
NCCL=('NCCL collectives','https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html')
CS=('CS336 Lecture 8','https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf')
BERK=('Berkeley Lecture 4','https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture4.pdf')
VLLM=('vLLM parallelism','https://docs.vllm.ai/en/stable/serving/parallelism_scaling/')
SCALE=('Scaling Book: inference','https://jax-ml.github.io/scaling-book/inference/')
PLAY=('Ultra-Scale Playbook','https://nanotron-ultrascale-playbook.static.hf.space/index.html')

def foundations():
    s=[]
    s.append(slide('Scaling LLMs across GPUs',
        text(75,206,'Week 3',29,color=MUTED)+
        lines(75,309,['Where do the tensors live,', 'and what must cross the network?'],45,weight=700)+
        line(75,433,1525,433)+
        text(75,500,'Replicas and training state',30,700,color=BLUE)+text(715,500,'Data parallelism, ZeRO, and FSDP',29)+
        line(75,540,1525,540)+text(75,600,'One model across devices',30,700,color=BLUE)+text(715,600,'Tensor, pipeline, context, and expert parallelism',29)+
        text(75,782,'Worked tensor diagrams, communication traces, and questions with solutions',27,color=MUTED),
        'This expanded Week 3 has no fixed presentation time. Read the sections in order or use the Slides menu to navigate by topic. The prerequisite is Weeks 1 and 2: GPU memory, GEMM, prefill/decode, KV cache, and online softmax. All diagrams and numerical examples are authored teaching schematics unless explicitly labeled otherwise. No example is a measured production benchmark or an attributed company interview. Sources are linked on individual slides and in these notes. No software setup is required to view the standalone HTML.',section='Introduction'))
    s.append(slide('Capacity, latency, and throughput are different goals',
        table(75,207,1450,['Goal','Concrete problem','What to measure'],[
            ['Capacity','Weights or active KV do not fit','Peak memory on every GPU'],
            ['Latency','One request generates too slowly','TTFT and time between tokens'],
            ['Throughput','Many requests wait in a queue','Output tokens/s under latency targets'],
        ],[.19,.45,.36],row_h=107,size=28)+
        text(75,686,'A scheme can improve capacity while adding communication and increasing latency.',29)+
        takeaway('Choose the workload and objective before choosing the GPU layout.'),
        'Capacity, per-request latency, and service throughput lead to different deployment choices. Reuse Week 2 definitions of TTFT and ITL, including queueing when discussing client-observed measurements. More memory can support larger batches, but extra GPUs do not automatically lower latency. Performance and memory must be checked per GPU rather than only summing the cluster. We will keep these objectives visible while introducing each partition.',(VLLM,SCALE),section='Introduction'))
    b=text(75,191,'Node = one server. Rank = one process in a communication group.',29)
    for n,x in enumerate([75,845]):
        b+=rect(x,252,680,350,'white')+text(x+25,298,f'Node {n}',32,700,color=BLUE)
        for g,gx in enumerate([x+30,x+360]):
            b+=label_box(gx,346,288,67,f'GPU {2*n+g} / rank {2*n+g}',size=27)+label_box(gx,444,288,86,'Own HBM',size=28)
        b+=arrow(x+323,380,x+355,380)+text(x+340,575,'Local GPU interconnect',25,anchor='middle')
    b+=arrow(766,437,834,437)+text(800,642,'Network between nodes',28,anchor='middle')
    b+=takeaway('Remote data needs an explicit transfer or communication operation.', 'Teaching setup: one worker process per GPU. Actual interconnects and bandwidth vary.')
    s.append(slide('GPUs have separate memory and communicate in groups',b,
        'A node is a physical server; a rank is a process identifier within a distributed group. This deck uses one process per GPU, but the terms are not definitions of one another. Each device owns its allocations and HBM; a model tensor is not automatically spread across all memory just because four devices are visible. A communication group specifies which ranks participate in an operation. On many accelerator servers intra-node GPU links are faster than the inter-node path, but inspect the real topology: PCIe-only nodes and different networks exist. This diagram is schematic and claims no universal link bandwidth.',(CS,VLLM),section='Introduction'))
    return s

def tensor_slides():
    s=[]
    s.append(slide('A two-layer MLP is our tensor-parallel example',
        text(75,190,'Two token rows, hidden width 4, intermediate width 8. Biases omitted.',29)+
        label_box(75,289,190,100,'X [2, 4]',size=31)+arrow(275,338,337,338)+
        label_box(350,289,245,100,'W₁ [4, 8]',size=31)+arrow(605,338,667,338)+
        label_box(680,289,245,100,'GELU',size=31)+arrow(935,338,997,338)+
        label_box(1010,289,245,100,'W₂ [8, 4]',size=31)+arrow(1265,338,1317,338)+
        label_box(1330,289,195,100,'Y [2, 4]',size=29)+
        text(75,502,'H = GELU(XW₁) has shape [2, 8]',37,700,color=BLUE)+
        text(75,566,'Y = HW₂ has shape [2, 4]',37,700,color=BLUE)+
        text(75,670,'Each token row uses the same learned weights. GELU acts on each scalar separately.',28)+
        takeaway('Tensor parallelism divides the work inside these matrix multiplications.'),
        'We use a dense feedforward block with two linear maps and an elementwise GELU. The exact shapes are teaching dimensions, not a real model. X has two rows: this could be two prompt tokens or two decode requests. W1 expands width 4 to 8, H has eight intermediate features, and W2 contracts it back to 4. Bias terms are omitted to focus on partitions. Modern gated MLPs have another projection but use the same compatible intermediate-feature partition idea. Each row still belongs to its original token/request.',(MEG,TPDOC),section='Tensor parallelism'))
    b=text(75,192,'Column partition: each GPU owns different output features of W₁.',30)
    for g,x in enumerate([75,845]):
        b+=text(x,271,f'GPU {g}',32,700,color=BLUE)+label_box(x,310,245,90,'Full X [2, 4]',size=28)+text(x+275,364,'×',36)+label_box(x+320,310,345,90,f'W₁{["a","b"][g]} [4, 4]',size=30)
        b+=arrow(x+333,411,x+333,467)+label_box(x+140,482,390,83,f'H{["a","b"][g]} = GELU(XW₁{["a","b"][g]})',size=29)
        b+=text(x+333,620,'[2, 4] intermediate features',27,anchor='middle')
    b+=takeaway('H = [Hₐ  Hᵦ], but we can keep these two pieces on separate GPUs.', 'The inputs are replicated here. The first weight matrix and intermediate features are partitioned.')
    s.append(slide('Column partition produces different output features',b,
        'Write W1 as the horizontal concatenation [W1a W1b], where each shard is 4x4. X is already replicated across the two ranks, so each independently computes a 2x4 output. GELU is elementwise, so it may act on each output shard independently. Concatenating would recover the full H, but physical all-gather is unnecessary if the following layer can consume the partition. Column refers to W as stored in the displayed mathematical convention XW; PyTorch nn.Linear stores its weight transposed, so implementation dimension numbers differ. This deck consistently uses XW.',(MEG,BERK),section='Tensor parallelism'))
    b=text(75,192,'Row partition: split W₂ along the same intermediate features as H.',30)
    for g,x in enumerate([75,845]):
        a=['a','b'][g];u=['U₀','U₁'][g]
        b+=text(x,268,f'GPU {g}',32,700,color=BLUE)+label_box(x,311,245,86,f'H{["ₐ","ᵦ"][g]} [2, 4]',size=30)+text(x+275,364,'×',36)+label_box(x+320,311,345,86,f'W₂{a} [4, 4]',size=30)
        b+=arrow(x+333,410,x+333,462)+label_box(x+100,477,470,87,f'{u}: partial sum [2, 4]',size=29)
    b+=text(800,654,'Y = HₐW₂a + HᵦW₂b = U₀ + U₁',35,700,color=BLUE,anchor='middle')
    b+=takeaway('Each GPU contributes to every output feature. Add the partial results.', 'A partial sum already has the final output shape; concatenating it would give the wrong result.')
    s.append(slide('Row partition produces partial sums',b,
        'W2 is partitioned vertically into W2a and W2b, each 4x4. The first shard pairs with Ha and the second with Hb. The two products both have the complete output shape 2x4, but each covers only half of the reduction dimension. Matrix multiplication sums over all eight intermediate features, so the correct output is their elementwise sum. This is fundamentally different from the previous output-feature partition: concatenation there, addition here. No nonlinear function can generally be applied independently to these partial sums before reducing if the function is intended to act on their total.',(MEG,NCCL),section='Tensor parallelism'))
    s.append(slide('AllReduce makes the sum available on both GPUs',
        text(75,192,'Separate toy example: two-element partial outputs, before and after a sum AllReduce.',29)+
        table(75,258,1450,['Rank','Before: different contributions','After: the same complete result'],[
            ['GPU 0','[2, 5]','[9, 6]'],
            ['GPU 1','[7, 1]','[9, 6]'],
        ],[.16,.42,.42],row_h=104,size=31)+
        text(75,641,'AllReduce(sum): elementwise reduction + delivery of the result to every rank.',28,700)+
        takeaway('The collective exchanges tensor data among the participating GPUs.', 'A sum AllReduce does not concatenate arrays or automatically divide by the number of GPUs.'),
        'This small numeric example illustrates the semantics of AllReduce with the sum operation, independent of algorithm. Both ranks enter a matching collective in a matching order; they contribute arrays with the same shape. The output is [2+7,5+1]=[9,6] on both. NCCL and frameworks choose actual transfer algorithms. Training gradient averaging may use a sum plus division or equivalent averaging semantics; TP partial sums require addition, not an average. A collective must be called consistently by all ranks in its group to avoid hanging or incorrect behavior.',(NCCL,),section='Tensor parallelism'))
    scenes=[]
    for phase in range(3):
        b=text(75,201,['1. Replicated input; partitioned weights','2. Each GPU computes its local MLP path','3. Sum the partial outputs on both GPUs'][phase],31,700,color=BLUE)
        for g,x in enumerate([75,845]):
            b+=text(x,280,f'GPU {g}',30,700)+label_box(x,317,665,66,'X [2, 4]',size=29)
            b+=label_box(x,411,665,66,f'GELU(XW₁{["a","b"][g]}) × W₂{["a","b"][g]}',size=31)
            if phase>=1:b+=arrow(x+333,484,x+333,519)+label_box(x,533,665,65,('U₀ [2, 4]' if g==0 else 'U₁ [2, 4]') if phase==1 else 'Y = U₀ + U₁ [2, 4]',size=29)
        if phase==2:b+=arrow(750,565,837,565)+text(800,635,'AllReduce(sum)',27,700,color=TEAL,anchor='middle')
        scenes.append(b)
    s.append(slide('The paired MLP avoids an intermediate gather',steps('tp-mlp',scenes)+
        takeaway('Column-partition W₁, keep H local, row-partition W₂, then AllReduce.', 'This forward example starts and ends with replicated hidden states.'),
        'Advance the three frames. W1a and W1b own distinct intermediate columns. W2a and W2b own matching intermediate rows. Therefore each rank consumes exactly the H shard it already produced; no H all-gather is needed between the two projections. The output sum restores a replicated hidden state for the next block. This is the classic paired MLP from Megatron-LM. The diagram concerns the forward pass; backward has conjugate communication operations and is not claimed communication-free. Tensor and sequence parallel variants change activation layouts at the boundaries, introduced later.',(MEG,TPDOC),section='Tensor parallelism'))
    s.append(slide('Attention can partition heads across GPUs',
        text(75,190,'Toy multi-head attention: 4 query heads and 4 KV heads; TP = 2.',30)+
        text(75,231,'Both GPUs process the same tokens, but each owns 2 heads.',28,color=MUTED)+
        label_box(75,303,665,70,'GPU 0: heads 0 and 1',size=30)+label_box(845,303,665,70,'GPU 1: heads 2 and 3',size=30)+
        label_box(75,411,665,91,'Local Q/K/V + KV cache for heads 0–1',size=27)+
        label_box(845,411,665,91,'Local Q/K/V + KV cache for heads 2–3',size=27)+
        arrow(405,508,405,552)+arrow(1175,508,1175,552)+
        label_box(75,566,665,72,'Local attention × output-weight shard',size=27)+label_box(845,566,665,72,'Local attention × output-weight shard',size=27)+
        text(800,703,'Sum the output-projection contributions with AllReduce',29,700,color=BLUE,anchor='middle')+
        takeaway('The KV cache can be partitioned by head in this configuration.', 'Fewer KV heads or incompatible TP degrees can require KV replication; do not assume memory always divides by TP.'),
        'For ordinary MHA, independently computed attention heads naturally map to TP ranks. Here four Q heads have four corresponding K/V heads; each GPU holds two heads and their cached history. The output projection mixes head results, so row-partitioning its input feature dimension yields partial sums that are reduced. Along with the MLP reduction, this gives two forward AllReduces per Transformer block in the classic replicated-boundary design. Models with GQA/MQA have fewer KV heads, so their cache sharding depends on implementation and TP degree; for example TP beyond the number of KV heads can replicate KV heads. This slide does not promise perfect 1/TP savings for all model tensors.',(MEG,TPDOC),section='Tensor parallelism'))
    s.append(slide('Small messages still pay communication latency',
        text(75,190,'A simple message model: time ≈ startup latency + bytes / effective bandwidth',30,700)+
        table(75,251,1450,['Illustrative transfer','Startup','Payload time','Total'],[
            ['10 KB at 100 GB/s','5 μs','0.1 μs','5.1 μs'],
            ['10 MB at 100 GB/s','5 μs','100 μs','105 μs'],
        ],[.43,.19,.19,.19],row_h=98,size=28)+
        text(75,597,'Small-batch decode can hit many short collectives on the layer-by-layer path.',29)+
        text(75,653,'More GPUs reduce local work, but do not remove synchronization or startup costs.',29)+
        takeaway('Bandwidth matters for large transfers; startup matters for small ones.', 'Hypothetical numbers for one transfer. A collective can involve multiple rounds and different effective costs.'),
        'This is a latency-plus-bandwidth teaching model, not a measured GPU specification or full collective formula. Decimal units: 10000/100e9 seconds=0.1 microseconds, 10e6/100e9=100 microseconds. The fixed 5 microseconds is assumed. For collectives, algorithm, rank count, routing, launch overhead and overlap determine the effective cost, so do not simply apply one-link numbers to an AllReduce. In small-batch decode many layers can expose small collectives with relatively little compute to amortize them. Prefill usually has more token rows and larger messages/work, changing the tradeoff.',(BERK,SCALE),section='Communication costs'))
    b=text(75,190,'Schematic critical path for one forward segment. Horizontal length represents time.',28)
    for y,title,segments in [(283,'TP = 2',[(530,280,'Local work',PALE),(810,160,'Exchange','#e6eeeb'),(970,280,'Local work',PALE)]),(453,'TP = 4',[(530,170,'Local work',PALE),(700,260,'Exchange','#e6eeeb'),(960,170,'Local work',PALE)])]:
        b+=text(75,y+49,title,33,700,color=BLUE)
        for x,w,label,c in segments:b+=label_box(x,y,w,80,label,c,25)
    b+=text(75,630,'End-to-end time includes local kernels, exposed communication, and other overhead.',28)
    b+=takeaway('Measure the critical path rather than assuming time scales as 1 / GPU count.', 'Compute and communication can overlap when dependencies permit. These bars are illustrative, not benchmarks.')
    s.append(slide('Communication can consume the time saved by sharding',b,
        'The chart shows illustrative sequences, not data and not predictions for a particular GPU. Some local kernels shrink with more TP; actual speed depends on GEMM shape and utilization, so even local compute may scale poorly. Communication can increase with rank count and topology. Exposed communication is the part not hidden behind independent work. The next operation may depend on the complete reduced result, placing that communication directly on the critical path. The right performance model follows dependencies across kernels and transfers; adding all times double-counts overlap, while taking a global max can unrealistically assume all communication overlaps.',(MEG,BERK),section='Communication costs'))
    return s

def pipeline_slides():
    s=[]
    b=text(75,190,'A 4-layer model on 2 GPUs. A stage owns complete layers and their weights.',29)
    b+=label_box(75,276,650,84,'GPU 0: layers 1–2',size=32)+label_box(875,276,650,84,'GPU 1: layers 3–4',size=32)
    b+=label_box(75,414,650,77,'Hidden states for this microbatch',size=29)+arrow(733,453,866,453)+label_box(875,414,650,77,'Continue the same forward pass',size=28)
    b+=text(75,581,'A microbatch is a subset of the batch processed through the pipeline.',29)+text(75,638,'For inference, each stage also keeps the KV cache for its own layers.',29)
    b+=takeaway('Pipeline parallelism transfers activations between layer groups.', 'Tensor parallelism splits operations within a layer. Pipeline parallelism splits the model by layers.')
    s.append(slide('Pipeline parallelism splits the layers',b,
        'The model has four layers only for illustration. The first stage computes layers 1–2 and sends their output hidden states to the second stage, which computes layers 3–4. The weights remain at their owning stage across iterations. At inference each stage also retains KV for its layers; it does not need to transfer the complete KV cache at every stage boundary. The communication payload at a boundary is typically an activation tensor, with sampling/control information handled by the serving implementation. A microbatch here contains independent examples or requests split out of a larger batch.',(CS,VLLM),section='Pipeline parallelism'))
    b=text(75,190,'Two equally fast stages, four independent microbatches. Transfers omitted in this toy.',28)
    b+=text(75,272,'GPU 0',30,700)+text(75,386,'GPU 1',30,700)
    for slot in range(5):
        x=280+slot*236
        b+=text(x+108,235,f'Slot {slot+1}',25,anchor='middle',color=MUTED)
        for g,y in enumerate([267,381]):
            mb=slot-g
            b+=label_box(x,y,216,82,chr(65+mb) if 0<=mb<4 else 'Idle', PALE if 0<=mb<4 else '#f5f5f5',30)
    b+=text(75,550,'Each microbatch still crosses both stages: A finishes at the end of slot 2.',29)
    b+=text(75,609,'Overlap lets different stages work on different microbatches at the same time.',29)
    b+=text(75,666,'With equal stages: 4 / (4 + 2 − 1) = 80% of GPU-time slots contain work.',29,700,color=BLUE)
    b+=takeaway('Pipeline bubbles are idle slots while the pipeline fills or drains.', 'This is a forward-only illustration. Training adds backward passes and different schedules.')
    s.append(slide('Independent microbatches fill the pipeline',b,
        'Read each row as one GPU and each column as one equal-duration stage interval. A enters stage0 in slot1 and stage1 in slot2. While stage1 processes A, stage0 processes B. Four microbatches on two stages require five slots. Eight occupied device-slots out of ten gives 80% utilization in this idealized model. More generally m equally sized microbatches on p equally fast stages use m/(m+p−1) of total stage-time slots, with no transfer cost. Unequal layers, communication and runtime overhead lower utilization. This formula is not a general training pipeline efficiency estimate and does not guarantee the same efficiency in online autoregressive decoding.',(CS,PLAY),section='Pipeline parallelism'))
    s.append(slide('One decode request still has a token dependency',
        text(75,191,'Request A must sample token t before it can process token t in the next pass.',29)+
        label_box(75,287,375,92,'Layers 1–2',size=31)+arrow(465,333,543,333)+label_box(560,287,375,92,'Layers 3–4',size=31)+arrow(950,333,1028,333)+label_box(1045,287,480,92,'Sample next token',size=31)+
        line(1285,396,1285,447,BLUE,2)+line(1285,447,262,447,BLUE,2)+arrow(262,447,262,494)+label_box(75,507,375,88,'Next decode pass',size=29)+
        text(560,544,'Later tokens from this request cannot fill',29)+text(560,592,'the earlier stage before sampling finishes.',29)+
        takeaway('Multiple requests can provide overlap. One request alone cannot fill every stage.', 'PD disaggregation instead separates phases: each pool runs all model layers for its assigned phase.'),
        'This diagram makes the autoregressive dependency explicit. A forward pass on the last sampled token must reach the final layer and sampler before the next token is known. Merely partitioning layers does not turn later tokens of a single sequence into independent microbatches. Multiple independent requests or suitable scheduling can keep stages busier. PP increases model capacity and can improve throughput under appropriate workloads, but must be measured for latency. Compare with Week2 PD disaggregation: prefill and decode instances each implement the full model, possibly themselves using TP or PP; the phase boundary transfers KV. PP and PD can coexist, but they split different things.',(SCALE,VLLM),section='Pipeline parallelism'))
    return s

def synthesis():
    s=[]
    s.append(slide('Parallelism choices change different tensor dimensions',
        table(75,190,1450,['Scheme','Partition / replica','Main communication'],[
            ['Data parallelism','Different examples, model replicas','Training gradients'],
            ['ZeRO / FSDP','Training state across DP ranks','Parameters and gradients'],
            ['Tensor parallelism','Features inside layer operations','Activation contributions'],
            ['Pipeline parallelism','Groups of model layers','Stage-boundary activations'],
            ['Context parallelism','Positions of the same sequence','Remote K/V or head exchange'],
            ['Expert parallelism','Different expert networks','Routed token activations'],
        ],[.25,.41,.34],row_h=77,size=27)+
        takeaway('Trace a tensor through the computation to see why communication is required.'),
        'This is a recap after the worked examples, not a replacement for them. Serving replicas are a form of data-parallel inference but do not perform training gradient synchronization. ZeRO/FSDP refer to state sharding along data-parallel groups, while TP changes how layer arithmetic is divided. CP and EP have implementation-specific communication paths explained in their sections. These dimensions can be combined, but group membership must be defined explicitly and the total GPU product depends on how the groups overlap.',(PLAY,CS),section='Putting it together'))
    b=text(75,190,'One dense model instance uses TP = 2 and PP = 2. Two replicas serve different requests.',28)
    for n,x in enumerate([75,845]):
        b+=text(x,260,f'Replica {n}: its own request batch',30,700,color=BLUE)
        for p,y in enumerate([301,478]):
            b+=text(x,y+23,f'Stage {p}',27,700)
            for t,off in enumerate([155,415]):
                rank=n*4+p*2+t
                b+=label_box(x+off,y-8,240,78,f'GPU {rank}',size=30)+text(x+off+120,y+112,f'TP shard {t}',26,anchor='middle')
            b+=arrow(x+402,y+33,x+408,y+33)
        b+=arrow(x+396,414,x+396,454)
    b+=text(800,704,'2 replicas × 2 pipeline stages × 2 TP ranks = 8 GPUs',32,700,color=BLUE,anchor='middle')
    b+=takeaway('A parallelism group is the set of ranks cooperating on one operation.', 'This example has no CP or EP. Additional dimensions require explicit groups and compatible layouts.')
    s.append(slide('A deployment combines several communication groups',b,
        'Two replicas process independent request batches. Each replica has two pipeline stages and each stage uses two TP ranks: ranks0/1 and2/3 in replica0, ranks4/5 and6/7 in replica1. TP reductions happen only within a stage’s pair; stage-boundary activations transfer between compatible ranks in consecutive stages. There is no training gradient synchronization between these serving replicas. The simple product 2x2x2=8 applies to these explicitly independent axes. Do not blindly multiply every named parallel degree in a real MoE setup: EP groups can be organized from an existing data-parallel axis, and framework constraints matter.',(VLLM,PLAY),section='Putting it together'))
    s.append(slide('Four GPUs offer several serving layouts',
        text(75,190,'Assume the BF16 8B model fits on one GPU with the required per-request KV budget.',28)+
        table(75,244,1450,['Layout','Weights across the 4 GPUs','What must be measured'],[
            ['4 single-GPU replicas','4 copies: 64 GB total','Batch efficiency and request balance'],
            ['2 instances, each TP = 2','2 copies: 32 GB total','Latency gain vs communication'],
            ['1 instance, TP = 4','1 copy: 16 GB total','KV capacity vs collective cost'],
        ],[.29,.31,.40],row_h=104,size=27)+
        text(75,657,'Weight totals assume ideal BF16 sharding. KV, temporary tensors, and replicated parts are extra.',26)+
        takeaway('Compare capacity and latency under the same offered workload and GPU budget.', 'A low-concurrency latency test and a saturated-throughput test answer different questions.'),
        'The same hypothetical 8B dense model has 16 GB of BF16 weights. Four independent replicas store 64 GB of aggregate weights; two TP2 instances store 32 GB; one TP4 instance stores16GB assuming ideal sharding of all counted weights. Real implementations replicate some tensors and add buffers. Although replication can improve throughput by serving independent requests, it consumes aggregate memory that could otherwise serve KV. The KV distribution itself depends on model and parallel scheme. Compare fixed model/precision, input-output lengths, offered arrival rate and latency targets. Avoid comparing one instance’s tokens/sec with an entire cluster’s rate. No layout is declared universally best.',(VLLM,SCALE),section='Putting it together'))
    s.append(slide('Question 1: Complete a tensor-parallel MLP',
        text(75,192,'Y = GELU(XW₁)W₂. Split the intermediate width equally across two GPUs.',29)+
        code(75,239,1450,280,'# Same X on both GPUs. W1 is [D, F]; W2 is [F, D].\nW1a, W1b = W1.chunk(2, dim=1)\nW2a, W2b = W2.chunk(2, dim=0)\nHa = gelu(X @ W1a)    # GPU 0\nHb = gelu(X @ W1b)    # GPU 1\n# Complete the local work and the communication.',27)+
        lines(75,568,['1. What does each GPU compute next? Is the final operation sum or concatenation?', '2. Can we avoid gathering H between the two linear layers?', '3. If W₁ were split along D instead, could GELU run before the partial sums combine?'],28,gap=55)+
        takeaway('Explain both numerical correctness and the required tensor transfers.'),
        'Authored interview-style exercise grounded in the Megatron-LM tensor partition, not a reported company question. Assume F is divisible by2, biases omitted, the same X is available on both ranks, and elementwise GELU. Ask the audience to label shapes before naming a collective. GPU0 computes Ua=Ha@W2a and GPU1 Ub=Hb@W2b, both [M,D]. Their sum isY. Keeping Ha/Hb partitioned eliminates an intermediate gather. If the first projection splits the reduction/input dimension, the partial X shards times W1 shards must be summed before GELU because GELU(a+b) is generally not GELU(a)+GELU(b). The next slide gives the answer.',(MEG,),section='Questions'))
    s.append(slide('Solution 1: Local products, then a sum AllReduce',
        code(75,195,1450,255,'# On each rank r, with matching W1r columns and W2r rows:\nH_local = gelu(X @ W1_local)\nY_partial = H_local @ W2_local\ndist.all_reduce(Y_partial, op=dist.ReduceOp.SUM)\n# Both ranks now hold Y.',29)+
        text(75,475,'Why it works',31,700,color=BLUE)+
        text(75,528,'GELU([XW₁a, XW₁b]) = [Hₐ, Hᵦ]',33)+
        text(75,584,'[Hₐ, Hᵦ] × [W₂a; W₂b] = HₐW₂a + HᵦW₂b',33)+
        text(75,681,'GELU(a + b) ≠ GELU(a) + GELU(b) in general: reduce before that nonlinearity.',28)+
        takeaway('Partition the two projections along the same intermediate features.', 'With an output bias, add it once after the sum, or partition the bias contribution consistently.'),
        'The input is replicated and each local H contains different output features, so GELU works locally. The second projection contracts the feature dimension: contributions add. An all-gather of Y_partial would concatenate redundant-shaped partial results rather than complete them. Output bias must not be added identically on both ranks before a sum, as that would add it twice. First-layer bias can be column-sharded with W1. Correctness tests should use numerical tolerances because floating-point reduction order changes. The pure-PyTorch logical-rank take-home later allows testing this on one device.',(MEG,TPDOC),section='Questions'))
    s.append(slide('Question 2: Why does TP = 4 barely improve decode?',
        text(75,190,'Hypothetical profile: same model, 8 requests per instance, same context length.',29)+
        table(75,242,1450,['Per decode iteration','TP = 2','TP = 4'],[
            ['Local kernels on the critical path','14 ms','8 ms'],
            ['Exposed collective communication','4 ms','8 ms'],
            ['Other work','2 ms','2 ms'],
            ['Total iteration time','20 ms','18 ms'],
        ],[.60,.20,.20],row_h=78,size=28)+
        text(75,678,'With 4 GPUs available, compare two TP=2 instances against one TP=4 instance.',28)+
        takeaway('What explains the latency result, and what experiment decides the serving layout?', 'Assume these components do not overlap. These are invented diagnostic inputs, not measured results.'),
        'This is an authored diagnostic question using explicitly hypothetical times, not benchmark evidence. Same per-instance batch8 and context are used to diagnose TP scaling. Local work improves by6ms while exposed communication adds4ms, leaving a2ms end-to-end gain. Ask why communication might rise: rank count, topology, synchronization, contention, collective implementation. Ask whether a throughput conclusion can be drawn under the same total arrival process. Two TP2 instances could collectively process twice as many requests at this particular per-instance batch, but that is a saturated upper-style comparison and does not establish online SLO capacity. TP4 at batch16 has not been measured. The solution distinguishes these comparisons.',(SCALE,VLLM),section='Questions'))
    s.append(slide('Solution 2: Separate local speedup from serving capacity',
        lines(75,197,['Local work saves 6 ms; communication adds 4 ms.', 'The net reduction is 2 ms per decode iteration.'],32,gap=50,weight=700,color=BLUE)+
        table(75,326,1450,['Hypothesis','Evidence to collect'],[
            ['A slower link enters the TP group','GPU topology and collective bandwidth/latency'],
            ['More synchronization dominates small work','Trace exposed collectives, not just their total duration'],
            ['Replicas can handle more independent traffic','Same arrival process; aggregate tokens/s and p95 latency'],
            ['TP=4 supports a more useful batch or KV budget','Sweep batch/context and measure peak memory'],
        ],[.43,.57],row_h=80,size=26)+
        takeaway('Benchmark both layouts under the workload that the service must support.', 'Keep per-instance batch, total active requests, and total GPU count distinct.'),
        'The given arithmetic identifies exposed communication as consuming most of the local benefit, but not its underlying reason. Trace matching collectives on all ranks and inspect skew, launch and network paths. Compare nodes and intra-node layouts before blaming a specific fabric. For the given fixed-batch numbers two saturated TP2 groups would yield2*8/0.020=800 tokens/s, while oneTP4 group atB8 gives8/0.018≈444; those configurations admit different total active requests and are not a complete online comparison. TP4 atB16 or another workload could behave differently. The correct service test fixes the request distribution and offered arrivals, increases load, and compares throughput within the desired TTFT/ITL percentiles and capacity.',(SCALE,VLLM),section='Questions'))
    s.append(slide('Take-home: Simulate two tensor-parallel ranks',
        text(75,193,'Use ordinary PyTorch tensors on a CPU or one GPU. No distributed setup is required.',29)+
        lines(75,276,['Implement Y = GELU(XW₁)W₂ and the two logical-rank version.', 'Test several shapes and compare the full output with torch.testing.assert_close.', 'Keep Hₐ and Hᵦ separate; combine only the two final partial outputs.', 'Add biases and show how to avoid adding the output bias twice.'],30,gap=75)+
        text(75,653,'Extension: run the local computations on 2 GPUs and replace the final sum with AllReduce.',28)+
        takeaway('A correctness simulation teaches the partition. A GPU run measures communication.', 'Record dtype, shapes, hardware, warmup, synchronization, and tolerances if you benchmark.'),
        'The exercise extends Question1 into executable code. First use one process and torch tensors to simulate logical ranks, making the data distribution explicit without needing a multi-GPU notebook. Use casesM=2,D=4,F=8;M=7,D=6,F=10;M=1,D=8,F=12 and a nonzero bias. Then, optionally, implement each local path under torch.distributed across two GPUs. A simulation timing is not a distributed performance measurement. The following slide supplies a complete concise correctness reference and links to the official TP tutorial.',(TPDOC,),section='Take-home'))
    s.append(slide('Take-home solution: PyTorch reference',
        code(75,181,1450,498,'import torch\nfrom torch.nn.functional import gelu\n\ndef tp_mlp(X, W1, W2, b1, b2):\n    W1a, W1b = W1.chunk(2, dim=1)\n    W2a, W2b = W2.chunk(2, dim=0)\n    b1a, b1b = b1.chunk(2)\n    Ua = gelu(X @ W1a + b1a) @ W2a\n    Ub = gelu(X @ W1b + b1b) @ W2b\n    return Ua + Ub + b2\n\nreference = gelu(X @ W1 + b1) @ W2 + b2\ntorch.testing.assert_close(tp_mlp(X, W1, W2, b1, b2), reference)',25)+
        '<a href="https://github.com/zhiweixx/llm-systems-study-group/blob/main/week-3-tp-exercise.py" target="_blank" rel="noopener">'+text(75,711,'Runnable exercise + correctness checks',27,600,color=BLUE,attrs='text-decoration="underline"')+'</a>'+f'<a href="{TPDOC[1]}" target="_blank" rel="noopener">'+text(840,711,'Official PyTorch TP tutorial',27,600,color=BLUE,attrs='text-decoration="underline"')+'</a>'+
        takeaway('The mathematical partition stays the same when the sum becomes a collective.', 'Initialize compatible tensors and choose tolerances for the dtype. F must be divisible by 2 here.'),
        'Complete the visible code by initializing X[M,D],W1[D,F],W2[F,D],b1[F],b2[D] with matching floating dtype and device. For example set a deterministic seed and use torch.randn with float64 on CPU for a precise correctness test. In the real distributed version, each rank owns one local W1/b1/W2 shard, the sum becomes all_reduce(SUM), and each rank adds b2 once after the reduction. The function shown uses temporary tensors on a single device and intentionally does not claim distributed speedup. This references the official PyTorch TP tutorial rather than a proprietary interview solution.',(TPDOC,),section='Take-home'))
    return s
