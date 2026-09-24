"""Diagram-led foundations and tensor/pipeline examples for the short deck."""
from .common import *

MEG = ('Megatron-LM', 'https://arxiv.org/abs/1909.08053')
PIPE = ('Megatron: pipeline schedules', 'https://arxiv.org/abs/2104.04473')
VLLM = ('vLLM parallelism', 'https://docs.vllm.ai/en/stable/serving/parallelism_scaling/')
NCCL = ('NCCL collectives', 'https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html')


def grid(x, y, rows, cols, color=PALE, cell=30):
    return ''.join(rect(x+c*cell,y+r*cell,cell,cell,color,'white') for r in range(rows) for c in range(cols))


def foundations():
    b = text(75, 242, 'Scaling LLMs across GPUs', 50, 700, color=BLUE)
    b += text(75, 310, 'Week 3 · Visual overview', 30, color=MUTED)
    for i in range(4):
        x=75+i*380
        b += rect(x, 420, 310, 176, 'white')
        b += text(x+155, 474, f'GPU {i}', 32, 700, anchor='middle')
        b += grid(x+59, 509, 2, 6, PALE, 32)
        if i<3: b += arrow(x+322, 499, x+368, 499)
    b += text(75, 697, 'What is copied?     What is split?     What must move?', 35, 700)
    b += takeaway('Follow the tensors to understand the parallelism.')
    s=[slide('Parallelism and sharding',b,
        'This is the 21-slide visual overview of Week 3. The original 50-slide lecture remains available as the detailed reference. '
        'We will distinguish three questions throughout: what is copied on multiple GPUs, what is partitioned across them, and '
        'what communication is needed to recover the intended computation. Boxes and colors indicate ownership or computation, '
        'not measured physical sizes or performance. The audience should know a Transformer forward pass and the basic GPU memory '
        'and inference concepts from Weeks 1 and 2. Data parallelism, tensor parallelism, pipeline parallelism, context parallelism, '
        'expert parallelism, and training-state sharding solve different parts of the scaling problem.',section='Introduction')]
    b=text(75, 192, 'Each GPU owns its memory. Remote tensors need communication.', 31)
    for n,x in enumerate([75,845]):
        b+=rect(x,256,680,273,'white')+text(x+24,302,f'Node {n} = one server',30,700,color=BLUE)
        for g,gx in enumerate([x+24,x+376]):
            b+=label_box(gx,339,280,65,f'GPU {2*n+g}',size=30)
            b+=label_box(gx,415,280,65,'Own HBM',fill='white',size=28)
        b+=arrow(x+315,370,x+365,370)
    b+=arrow(766,393,834,393)
    b+=text(800,567,'Within-node links and between-node links can have different speeds.',27,anchor='middle')
    for x,name,cells in [(75,'Replica: a complete copy',4),(845,'Shard: part of a tensor',2)]:
        b+=text(x,632,name,30,700,color=BLUE)
        b+=grid(x,661,1,cells,BLUE,44)+grid(x+243,661,1,cells,TEAL if cells==2 else BLUE,44)
        if cells==2:b+=text(x+117,691,'+',30)
        else:b+=text(x+198,691,'=',30)
    b+=takeaway('A communication group specifies which GPUs work together.')
    s.append(slide('Separate memories, connected GPUs',b,
        'A node is a server. This teaching setup uses one process per GPU, but a rank is a process identifier in a distributed '
        'group rather than the hardware itself. Every GPU has its own memory allocations. A replica is a complete copy of the '
        'object being replicated; a shard is a subset of it. The two blue rows are identical copies, while the blue and teal '
        'pieces are different portions that together form one tensor. These are schematic tensor diagrams, not capacity bars. '
        'The links inside a server often provide more bandwidth and lower latency than links between servers, but this depends '
        'on the actual hardware. The later placement slide uses that assumption explicitly. A group may span nodes, and there '
        'is no universal prohibition against tensor, context, or expert parallelism crossing a server boundary.',(VLLM,),section='Introduction'))
    return s


def tensor_pipeline():
    slides=[]
    b=text(75,193,'A two-layer MLP: H = GELU(XW₁), Y = HW₂.   Shapes: X [2, 4], W₁ [4, 8], W₂ [8, 4].',32)
    for g,x in enumerate([75,835]):
        color=[BLUE,TEAL][g]; part=['a','b'][g]; h=['Hₐ','Hᵦ'][g]
        b+=rect(x,265,690,337,'white')+text(x+24,309,f'GPU {g}',31,700,color=color)
        b+=text(x+30,366,'X [2, 4]',27,700)
        b+=text(x+223,366,f'W₁{part} [4, 4]',27,700)
        b+=text(x+477,366,f'{h} [2, 4]',27,700)
        b+=grid(x+30,420,2,4,PALE,33)+text(x+181,463,'×',34)
        b+=grid(x+239,386,4,4,color,33)
        b+=arrow(x+394,451,x+465,451)+text(x+427,410,'GELU',24,anchor='middle')
        b+=grid(x+493,420,2,4,color,33)
        b+=text(x+30,559,'Same input',27)+text(x+223,559,'Different columns',27)
    b+=text(800,675,'H = [ Hₐ | Hᵦ ]     ·     8 features split across 2 GPUs',34,700,anchor='middle')
    b+=takeaway('Tensor parallelism splits a computation inside a layer.')
    slides.append(slide('TP: split the output features',b,
        'The input X has two token rows and four features. The first MLP matrix W1 has shape [4,8]. Split its eight output '
        'columns into W1a and W1b, each [4,4]. Both GPUs have the same X. Each produces four of the eight intermediate '
        'features, and the elementwise GELU can be applied locally to each shard. Conceptually H is the concatenation of Ha '
        'and Hb; no physical concatenation is needed if the next operation consumes the existing partitions. In this diagram '
        'the blue and teal cells denote different feature ownership, while pale inputs are replicated. The matrix convention '
        'is XW; PyTorch Linear stores its weight transposed. Biases and gated-MLP variants are omitted to isolate the partition.',
        (MEG,),section='Tensor parallelism'))

    b=text(75,193,'Keep H split. Partition W₂ along the same intermediate features.',31)
    for g,x in enumerate([75,835]):
        color=[BLUE,TEAL][g];part=['a','b'][g];h=['Hₐ','Hᵦ'][g]
        b+=rect(x,261,690,314,'white')+text(x+24,305,f'GPU {g}',31,700,color=color)
        b+=text(x+30,360,f'{h} [2, 4]',27,700)+text(x+223,360,f'W₂{part} [4, 4]',27,700)
        b+=text(x+480,360,f'U{g} [2, 4]',27,700)
        b+=grid(x+30,407,2,4,color,33)+text(x+181,451,'×',34)
        b+=grid(x+239,374,4,4,color,33)+arrow(x+395,440,x+465,440)
        b+=grid(x+493,407,2,4,PALE,33)+text(x+455,540,'Partial sum',28,700)
    b+=arrow(420,587,420,649)+arrow(1180,587,1180,649)
    b+=rect(326,650,948,69,PALE,'none')+text(800,694,'AllReduce(sum): Y = U0 + U1 on both GPUs',32,700,color=BLUE,anchor='middle')
    b+=takeaway('Each GPU computes part of every output. Add the partial results.')
    slides.append(slide('TP: sum the contributions to the output',b,
        'W2 has shape [8,4]. Partition its eight input-feature rows into W2a and W2b, each [4,4], matching the intermediate '
        'features retained by the corresponding GPU. HaW2a and HbW2b both have the full output shape [2,4], but each contains '
        'only part of the sum over eight input features. Therefore Y is their elementwise sum, not their concatenation. '
        'AllReduce with sum makes that result available on every participating GPU. The first MLP projection did not require '
        'gathering the intermediate tensor; the second restores the replicated output by summation. This is the forward '
        'path only. The backward pass also has communication. The boxes describe tensor ownership, not actual kernel boundaries.',
        (MEG,NCCL),section='Tensor parallelism'))

    b=text(75,190,'Example: 4 query heads, 4 KV heads, TP = 2.',31)
    b+=label_box(466,237,668,64,'Same token positions on both GPUs',size=29)
    b+=arrow(630,310,420,357)+arrow(970,310,1180,357)
    for g,x in enumerate([75,835]):
        color=[BLUE,TEAL][g]
        b+=rect(x,366,690,291,'white')+text(x+25,410,f'GPU {g}',31,700,color=color)
        for h in range(2):
            xx=x+25+330*h
            b+=label_box(xx,443,308,73,f'Head {2*g+h}',fill=PALE,size=29)
            b+=text(xx+154,557,f'KV for head {2*g+h}',27,anchor='middle')
        b+=text(x+345,619,'Local attention → output projection',28,anchor='middle')
    b+=text(800,710,'Sum the output-projection contributions with AllReduce',30,700,color=BLUE,anchor='middle')
    b+=takeaway('TP partitions features or heads while processing the same tokens.',
                 'KV-cache savings depend on the model’s KV-head layout and implementation.')
    slides.append(slide('TP can also split attention heads',b,
        'For ordinary multi-head attention in this example, four query heads have four corresponding KV heads. Each GPU '
        'computes two heads and stores the KV history for those heads. Both still process the same token positions. The output '
        'projection mixes the head outputs; partitioning its input dimension produces partial sums that need to be added. '
        'This is the same structure as the row-partitioned second MLP projection. Grouped-query or multi-query attention '
        'has fewer KV heads, so not every TP degree produces an equal nonreplicated KV partition. Some implementations '
        'replicate KV heads when the TP degree is too large. No general 1/TP KV saving is implied. TP reduces each rank’s '
        'weight/computation share but introduces repeated communication within the model path.',(MEG,VLLM),section='Tensor parallelism'))

    b=text(75,190,'Pipeline parallelism assigns different layers to different GPUs.',31)
    for stage,x in enumerate([75,875]):
        b+=rect(x,278,650,324,'white')+text(x+25,325,f'GPU {stage} / stage {stage}',32,700,color=BLUE)
        for layer in range(3):b+=label_box(x+25+200*layer,365,183,100,f'Layer {stage*3+layer+1}',size=29)
        b+=text(x+325,552,'Weights and per-layer KV stay here',28,anchor='middle')
    b+=arrow(735,415,863,415)
    b+=text(800,245,'Hidden states →',30,700,color=BLUE,anchor='middle')
    b+=line(1200,608,1200,664)+line(1200,664,400,664)+arrow(400,664,400,608)
    b+=text(800,713,'← Activation gradients during training',29,anchor='middle')
    b+=takeaway('The same microbatch passes through every stage.',
                 'A microbatch is a smaller group of examples taken from the batch.')
    slides.append(slide('PP: split the model by layers',b,
        'This toy model has six layers and two pipeline stages. The first stage owns layers 1–3 and sends hidden states '
        'to the second stage for layers 4–6. The weights remain with their stage. During autoregressive inference, each '
        'stage also retains the KV cache for its own layers; the entire KV history is not transferred at every stage '
        'boundary. Training adds the reverse flow of activation gradients. A stage can comprise more than one GPU if '
        'combined with TP, introduced later. A microbatch is a subset of examples processed through this full path. '
        'Pipeline parallelism divides model layers; prefill–decode disaggregation instead separates two execution phases '
        'of inference, so they should not be conflated.',(VLLM,PIPE),section='Pipeline parallelism'))

    b=text(75,190,'Toy schedule: equal stages, no transfer cost. Letters are independent microbatches.',29)
    colors=[PALE,'#e5efec','#dce7ef','#f0e8df']
    for y,m in [(294,1),(510,4)]:
        b+=lines(75,y+32,[str(m),'microbatch'+('es' if m>1 else '')],28,gap=42,weight=700,color=BLUE)
        for g in range(2):
            yy=y+g*80
            b+=text(342,yy+41,f'GPU {g}',27,anchor='end')
            for slot in range(m+1):
                mb=slot-g
                b+=label_box(382+slot*164,yy,146,65,chr(65+mb) if 0<=mb<m else 'Idle',colors[mb] if 0<=mb<m else '#f1f1f1',29)
        b+=text(1260,y+49,'50%' if m==1 else '80%',40,700,color=BLUE)
        b+=text(1260,y+91,'active GPU-time',25)
    b+=text(75,238,'Time →',27,color=MUTED)
    b+=takeaway('More microbatches reduce pipeline bubbles.',
                 'At a fixed batch size, smaller microbatches give more chunks but may make GEMMs less efficient.')
    slides.append(slide('PP: overlap work with microbatches',b,
        'Read each horizontal row as one GPU and columns as equal stage-time intervals. With one microbatch A, GPU1 '
        'waits for GPU0; then GPU0 waits while GPU1 completes A. Two of four GPU-time slots contain work. With four '
        'independent microbatches A–D, GPU0 can process B while GPU1 handles A. Eight of ten slots contain work. '
        'This is a forward-only toy with equal stage speeds and no communication cost, not a measured utilization. '
        'In this ideal model the active share is m/(m+p−1). Training schedules change the timeline and memory requirements. '
        'Microbatch count and microbatch size are different: at fixed global batch and DP degree, smaller microbatches '
        'increase their count but may produce less efficient GEMMs. Megatron, PyTorch pipelining, and DeepSpeed support '
        'pipeline microbatches. One autoregressive request cannot treat its unknown future tokens as independent '
        'microbatches; inference needs independent requests or batches to overlap stages.',(PIPE,),section='Pipeline parallelism'))
    return slides
