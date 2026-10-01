"""A worked 64-H100 serving case. All estimates are derived, not measured."""
from .common import *

BOOK=('Applied inference chapter','https://liltom-eth.github.io/scaling-book-pytorch/chapters/applied-inference.html')
META=('Llama 3 paper §6','https://arxiv.org/html/2407.21783v3#S6')
CONFIG=('Meta model configuration','https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py')
H100=('H100 specifications','https://www.nvidia.com/en-us/data-center/h100/')
DGX=('DGX H100 system guide','https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html')
VLLM=('vLLM distributed serving','https://docs.vllm.ai/en/v0.19.1/serving/parallelism_scaling/')
QKV=('vLLM Q/K/V sharding','https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/layers/linear.py#L1009')
DCP=('vLLM context parallelism','https://docs.vllm.ai/en/v0.19.1/serving/context_parallel_deployment/')
NIM=('NVIDIA 405B FP8 support','https://docs.nvidia.com/nim/large-language-models/1.13.0/supported-models.html#llama-3-1-405b-instruct')
FP8_CODE=('Meta FP8 implementation','https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/quantization/loader.py#L53-L96')
def week3(n,label):return ('Week 3: '+label,f'https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-{n}')
SECTION='Case study · Llama 3.1 405B on 64 H100s'
ASSUMPTIONS='Storage units are decimal: 1 GB = 10^9 bytes. The baseline uses the canonical architecture with 8 KV heads, BF16 weights, and BF16 KV unless explicitly changed. P = 405 billion is rounded; equal weight partitioning is a planning approximation. The baseline excludes prefix sharing, CPU offload, and a speculative draft model. No GPU measurements were run.'

def page(title,body,notes,sources):
    return slide(title,body,notes+' '+ASSUMPTIONS,sources,SECTION)

def node(x,y,w,h,name,subtitle='8 × H100',fill=PALE):
    out=rect(x,y,w,h,'white')+text(x+16,y+31,name,24,700,color=BLUE)
    gap=7; gw=(w-32-7*gap)/8
    for i in range(8):out+=rect(x+16+i*(gw+gap),y+48,gw,25,fill)
    return out+text(x+16,y+h-16,subtitle,23)

def problem():
    body=text(75,193,'Assumption: 8 servers × 8 H100 SXM 80 GB = 64 GPUs.',34,700)
    body+=text(75,242,'Serve Llama 3.1 405B: first establish a BF16 baseline, then compare alternatives.',29)
    for i in range(8):
        body+=node(75+(i%4)*370,290+(i//4)*146,340,119,f'Node {i}')
    body+=line(75,570,1525,570)
    body+=lines(75,613,['Within a node: NVLink / NVSwitch', 'Across nodes: the cluster network'],28,gap=43)
    body+=lines(820,613,['Per GPU: 3.35 TB/s HBM bandwidth', 'Dense BF16 peak: 989 TFLOP/s'],28,gap=43)
    body+=text(75,707,'Worked workload: 8,192 cached tokens per request; compare different decode batch sizes.',27)
    body+=takeaway('How should we divide 64 GPUs between one model and independent replicas?', 'We will distinguish memory capacity, one-request latency, and total serving throughput.')
    return page('Serve a 405B model on eight H100 nodes',body,
      'This is a new deployment case, separate from the four-GPU 8B example on slide 13. Here a node explicitly means an eight-GPU server: eight nodes contain 64 GPUs. The hardware is a DGX-like H100 SXM 80 GB system with an intra-node NVSwitch fabric and an inter-node RDMA-capable network whose actual bandwidth and latency must be measured. The H100 BF16 peak is the dense peak, not the doubled sparse peak. The linked applied-inference chapter supplies the progression from capacity to latency and deployment, but its 70B numerical examples are not reused. All 405B values are derived independently. Batch B later means the number of sequences in one decode microbatch on one replica; sequence length S means the number of currently cached positions. S grows during generation.',[BOOK,DGX,H100])

def fit():
    body=text(75,198,'Weights alone: 405 × 10⁹ parameters × 2 bytes ≈ 810 GB.',38,700,color=BLUE)
    x,w=75,1410; scale=w/1280
    body+=text(75,280,'One node: 640 GB',29,700)+text(830,280,'Two nodes: 1,280 GB',29,700)
    body+=rect(x,310,810*scale,70,PALE)+text(x+810*scale/2,354,'BF16 weights: 810 GB',29,700,anchor='middle')
    body+=rect(x+810*scale,310,470*scale,70,'white')+text(x+1045*scale,354,'KV + runtime',28,anchor='middle')
    body+=line(x+640*scale,295,x+640*scale,414,WARM,3,'7 5')
    body+=text(x+640*scale,448,'One-node capacity ends here',25,color=WARM,anchor='middle')
    body+=table(75,485,1450,['Layout','Weight storage per GPU','Fits before KV?'],[
      ['TP8 · one node','810 / 8 ≈ 101.3 GB','No: each GPU has 80 GB'],
      ['TP8 × PP2 · two nodes','810 / 16 ≈ 50.6 GB','Yes: budget KV and buffers next']
    ],ratios=[.34,.34,.32],row_h=69,size=28)
    body+=takeaway('One BF16 replica needs more than one eight-GPU node.', 'At least 11 GPUs by bytes alone; topology and sharding constraints motivate a 16-GPU layout.')
    return page('First constraint: one BF16 copy exceeds one node',body,
      'The bar compares total storage, not a physically pooled memory address space. Each GPU must hold its assigned tensors. ceil(810 / 80) = 11 is only a necessary byte-capacity condition; it is not a suggested tensor-parallel degree. TP degrees must suit the model heads and matrix partitions, while runtime allocations and KV also need space. A practical two-node option is TP8 × PP2: each node owns 63 of the 126 Transformer layers and shards its layers over eight GPUs. The embedding and output matrices at the two endpoints, plus other unsharded tensors, can make the true rank allocations unequal. Section 6.1 of the Meta paper reports this two-machine BF16 inference design.',[CONFIG,META,week3(17,'combine groups')])

def placement():
    body=text(75,195,'Four independent replicas; each replica spans two servers.',34,700)
    body+=text(75,242,'TP8 splits each layer inside a server. PP2 assigns 63 layers to each server.',29)
    for r in range(4):
        y=280+r*104
        body+=text(75,y+47,f'Replica {r}',28,700,color=BLUE)
        body+=label_box(270,y,510,78,f'Node {2*r}  ·  layers 1–63  ·  TP8',size=27)
        body+=arrow(780,y+39,915,y+39)
        body+=label_box(915,y,510,78,f'Node {2*r+1}  ·  layers 64–126  ·  TP8',size=27)
    body+=text(803,710,'PP sends hidden states between stages',24,color=MUTED)
    body+=takeaway('4 replicas × 2 pipeline stages × 8 tensor shards = 64 GPUs.', 'Route each request to one replica. Replicas process independent requests; no gradient synchronization.')
    return page('A starting layout: four replicas of TP8 × PP2',body,
      'Read one horizontal row first: it is one complete 405B model. Node 0 stores the first 63 layers and node 1 stores the last 63. Within either node, eight GPUs cooperate on every local layer. Duplicate this layout to create four replicas using the 64-GPU budget. These are inference replicas, the serving analogue of data parallelism from Week 3, rather than DDP training jobs. KV belongs to the chosen replica. A session can be routed back to its replica for cache reuse, subject to load and cache residency. The PP arrow represents a logical hidden-state transfer; actual wire volume depends on whether the backend transfers replicated activations or gathers/scatters a sharded boundary. One request does not span all four replicas.',[META,week3(4,'serving replicas'),week3(10,'PP')])

def tp_layer():
    body=text(75,193,'Inside one layer: 128 query heads, 8 KV heads; head dimension = 128.',31)
    for r in range(8):
        x=75+r*184
        body+=rect(x,253,164,174,PALE)+text(x+82,291,f'GPU {r}',26,700,color=BLUE,anchor='middle')
        body+=text(x+82,337,'16 Q heads',24,anchor='middle')+text(x+82,377,f'KV head {r}',24,anchor='middle')
        body+=line(x+82,427,x+82,459,BLUE,2)
    body+=line(157,459,1445,459,BLUE,2)+arrow(800,459,800,485)
    body+=label_box(410,485,780,62,'Output projection → sum with AllReduce',size=29)
    body+=text(75,599,'The FFN uses paired column / row sharding too:',30,700,color=BLUE)
    body+=text(75,643,'53,248 intermediate features → 6,656 per GPU → sum partial outputs.',30)
    body+=text(75,696,'Standard TP layout: about two AllReduces per layer, or 252 over the full model.',28)
    body+=takeaway('TP8 matches the eight KV heads: every GPU owns one head’s cache.', 'The layer weights are sharded persistently; only intermediate results are communicated.')
    return page('Apply Week 3 tensor parallelism to this model',body,
      'The diagram depicts one layer on its assigned PP stage. GQA associates 16 query heads with each of eight KV heads. TP8 assigns one complete KV head and its 16 associated query heads to each rank. The row-parallel attention output projection sums contributions from all ranks. A conventional paired column-parallel/row-parallel SwiGLU FFN similarly has a reduction after the down projection: 53,248 / 8 = 6,656 intermediate features per rank. Both gate and up projections follow that split. Thus the standard replicated-activation TP forward layout has two AllReduces per layer. Sequence-parallel and fused implementations may replace or combine collectives; the diagram is a reasoning baseline, not a profiler trace. Each PP node performs its own 126 layer reductions; 252 lie along a full forward path.',[CONFIG,week3(9,'attention TP'),week3(8,'TP reduction')])

def capacity():
    body=text(75,192,'TP8 × PP2; B = 32 requests, S = 8,192 cached tokens; BF16 KV.',31,700)
    body+=text(75,256,'KV / GPU = 2 × 63 layers × B × S × 1 KV head × 128 × 2 bytes',31)
    body+=text(75,304,'= 8.46 GB at B = 32; 0.264 GB for each additional 8k request.',31,700,color=BLUE)
    # Values in decimal GB, widths proportional to per-rank storage.
    vals=[(50.625,'Weights','50.6 GB',PALE),(8.455716864,'KV','8.46 GB','#d7e8e3'),(8,'Reserve','8 GB','#eeeeee'),(12.919283136,'Remaining','12.9 GB','white')]
    x=75
    for val,a,b,fill in vals:
        ww=1450*val/80
        body+=rect(x,360,ww,64,fill)
        body+=text(x+ww/2,460,a,24,700,anchor='middle')+text(x+ww/2,493,b,25,anchor='middle')
        x+=ww
    body+=line(75,533,1525,533)+text(75,579,'Capacity estimate with an assumed 8 GB runtime reserve per GPU:',29)
    body+=text(75,635,'B ≤ floor((80 − 50.625 − 8) / 0.26424) = 80 requests / replica',32,700,color=BLUE)
    body+=text(75,689,'Four replicas: about 320 requests at this context length, before additional allocation overhead.',26)
    body+=takeaway('Capacity depends on total cached tokens, not just the number of requests.', 'The 8 GB reserve is a teaching assumption; actual buffers, KV blocks and peak allocations must be measured.')
    return page('Budget KV memory on each GPU',body,
      'KV bytes for the whole model per cached token are 2 × 126 × 8 × 128 × 2 = 516,096. One 8,192-token request uses 4.227858432 GB of unique KV across the replica. TP8 head sharding and PP2 layer sharding divide this by 16: 0.264241152 GB per GPU per request. At B = 32, this is 8.455716864 GB per GPU. The assumed rank budget for KV is 80 − 50.625 − 8 = 21.375 GB, so floor(21.375 / 0.264241152) = 80. Four replicas would fit approximately 320 such requests under this simplified capacity model. This is not a throughput claim, a max-num-seqs recommendation, or a promise of an ITL target. Context grows during generation. At 32,768 cached tokens, the same calculation gives 20 requests per replica. Runtime workspace may grow with the batch and prefill token budget; the 8 GB reserve is not a guarantee. No prefix sharing is credited.',[CONFIG,week3(9,'KV ownership')])

def tp16():
    body=text(75,194,'Alternative: TP16 across two nodes, with no pipeline stages.',32,700)
    body+=table(75,234,1450,['Same 16 GPUs / replica','TP8 × PP2','TP16 · ordinary head sharding'],[
      ['Weight memory / GPU','≈ 50.6 GB','≈ 51.2 GB'],
      ['Where TP reductions run','Inside each node','Across both nodes'],
      ['Layers held on each GPU','63','126'],
      ['KV at B32, S8,192 / GPU','8.46 GB','16.91 GB']
    ],ratios=[.35,.27,.38],row_h=64,size=27)
    body+=text(75,615,'Why twice the KV? There are 16 ranks but only 8 distinct KV heads.',30,700,color=BLUE)
    body+=lines(75,660,['TP16 assigns 8 query heads per rank; two ranks can need the same KV head.', 'Both store that head’s history and K/V projection weights.'],28,gap=40)
    body+=takeaway('The same GPU count can give different communication and KV costs.', 'Compare shorter layer execution against network collectives and KV replication.')
    return page('Why not simply use TP16?',body,
      'Compare one replica using exactly the same two nodes. In plain head-sharded TP16, each rank handles 8 of the 128 query heads. Each group of 16 query heads shares one KV head, so two ranks need the same KV history. Without context sharding or another specialized attention scheme, each KV head has two copies across the 16 ranks. Each rank now holds one head across 126 layers, versus 63 layers in TP8 × PP2, so per-rank KV doubles. The exact checkpoint and implementation matter: Meta also distributed an MP16 configuration with 16 KV heads, reflecting duplication; this case starts from the canonical architecture with 8 KV heads. The duplicated K/V projections add about 0.528 GB per rank beyond 810 / 16, so TP16 weights occupy approximately 51.2 GB per rank; this duplication remains with TP16 + DCP2. All 252 conventional TP reductions now span both nodes. Wider TP can lower local layer time, but exposed network cost can negate it. Do not treat the 900 GB/s aggregate bidirectional NVLink specification as inter-node or usable AllReduce bandwidth. Context parallelism later gives a way to remove the cache duplication.',[CONFIG,QKV,week3(18,'placement')])

def pipeline():
    body=text(75,193,'Follow one decode microbatch through PP2. Each stage owns half the model.',31)
    body+=text(75,251,'Weight-read floor per stage: (810 GB / 16) / (3.35 TB/s) ≈ 15.1 ms.',32,700,color=BLUE)
    body+=text(75,310,'Idealized schedule: equal stage times; transfer, KV traffic and other costs omitted.',26,color=MUTED)
    x0=290; ww=278
    for j,t in enumerate(['0','15.1','30.2','45.3','60.4 ms']):body+=text(x0+j*ww,359,t,24,anchor='middle')
    for r,name in enumerate(['Node 0','Node 1']):body+=text(75,423+r*91,name,28,700)
    slots=[(0,0,'A: token t',PALE),(0,1,'B: token t','#d7e8e3'),(0,2,'A: token t+1',PALE),(0,3,'B: token t+1','#d7e8e3'),(1,1,'A: token t',PALE),(1,2,'B: token t','#d7e8e3'),(1,3,'A: token t+1',PALE)]
    for r,c,label,fill in slots:body+=label_box(x0+c*ww,380+r*91,ww-8,63,label,fill,25)
    body+=line(75,580,1525,580)
    body+=lines(75,627,['One microbatch: two serial stages → at least 30.2 ms for the weight reads.', 'A and B are independent microbatches: their stages can overlap.', 'A’s next token cannot start until A’s previous token finishes and is sampled.'],29,gap=43)
    body+=takeaway('PP adds memory capacity and can improve throughput; it does not split one token’s time in half.', 'The 15.1 ms stage interval differs from the 30.2 ms request path. Actual times are longer.')
    return page('Pipeline throughput is different from token latency',body,
      'This teaching timeline uses a weight-streaming lower bound as the duration of one stage. A and B are disjoint sets of requests, not successive tokens of a single sequence that could run without dependencies. At 0–15.1 ms, node 0 processes token t for A. At 15.1–30.2 ms, node 1 processes A while node 0 processes B. Only after A finishes both stages can token t + 1 for A begin. In a real engine, sampling, transfers, and scheduling add time, and the two stages may not balance. The ideal pipeline can emit one microbatch result every 15.1 ms after filling, although each microbatch spends 30.2 ms on its path. Dividing 810 GB by 16 × 3.35 TB/s estimates one stage or an ideal throughput interval, not single-microbatch latency. Meta measures improved throughput and increased latency with microbatching, so the ideal drawing is not a measured speedup. Prompt chunk pipelining can also change schedules and must be modeled explicitly.',[META,week3(11,'PP microbatches'),H100])

def decode():
    body=text(75,193,'One decode microbatch: B sequences, each with S = 8,192 cached tokens.',31)
    body+=text(75,253,'Linear-layer work ≈ 2 × 405B × B FLOPs',34,700,color=BLUE)
    body+=text(75,303,'HBM traffic ≈ 810 GB of weights + B × 4.228 GB of KV',34,700,color=BLUE)
    body+=text(75,352,'One weight scan is shared across the batch; each sequence brings its own KV history.',28)
    body+=table(75,392,1450,['B','Weight + KV traffic','HBM-time floor','Linear-compute floor'],[
      ['1','814 GB','30.4 ms','0.10 ms'],
      ['8','844 GB','31.5 ms','0.82 ms'],
      ['32','945 GB','35.3 ms','3.28 ms']
    ],ratios=[.09,.31,.30,.30],row_h=63,size=28)
    body+=text(75,699,'Floors sum the two PP stages: traffic / (8 × 3.35 TB/s); linear FLOPs / (8 × 989 TFLOP/s).',26)
    body+=takeaway('Batching amortizes the weight scan; growing KV traffic limits the gain.', 'Analytic streaming floors, not predicted TPOT. Add attention work, communication and imperfect efficiency.')
    return page('Decode: reuse weights, but read each request’s KV',body,
      'For TP8 × PP2, one microbatch traverses both stages, so the total path uses an effective 8-way bandwidth denominator in this additive stage model. Each stage has half the weight and KV volume. At B = 32 and S = 8,192, KV occupies 135.291469824 GB across the model, total traffic is 945.291469824 GB, and the ideal stage-summed HBM time is 35.2721 ms. Leading linear work is 2PB = 25.92 TFLOPs; dividing by 8 × 989 TFLOP/s gives 3.276 ms. These are separate lower-bound components, not terms to simply add: compute and memory can overlap. The model assumes each weight and each unique KV entry is read once with efficient reuse within each GQA group; actual traffic can be higher. Decode attention adds approximately 4 × 126 × 16,384 × B × S FLOPs, about 8.4% of 2PB at S = 8,192, plus non-matmul work. At these small batches, the HBM bound dominates the linear compute bound. This does not imply that all decode workloads are bandwidth-bound. No output-token/s estimate is made without a schedule for independent PP microbatches.',[BOOK,H100,week3(19,'communication costs')])

def prefill():
    body=text(75,193,'Now prefill one new 8,192-token prompt, with no cached prefix.',32,700)
    body+=text(75,251,'A linear layer sees 8,192 token rows at once, rather than one row per request.',29)
    body+=text(75,316,'Forward linear work ≈ 2 × 405 × 10⁹ × 8,192 = 6.64 × 10¹⁵ FLOPs',33,700,color=BLUE)
    body+=label_box(75,382,670,96,'Stage 0: half the work on 8 GPUs',size=30)
    body+=arrow(750,430,838,430)+label_box(855,382,670,96,'Stage 1: half the work on 8 GPUs',size=30)
    body+=text(410,529,'≈ 0.419 s at dense BF16 peak',28,anchor='middle')
    body+=text(1190,529,'≈ 0.419 s at dense BF16 peak',28,anchor='middle')
    body+=line(75,573,1525,573)
    body+=text(75,620,'Unchunked prompt path: ≈ 0.839 s of linear compute at peak.',34,700,color=BLUE)
    body+=text(75,671,'Compare the weight-streaming floor: ≈ 0.030 s. Reusing weights shifts the balance.',28)
    body+=takeaway('Prefill offers enough token rows to keep large matrix multiplications busy.', 'Attention, collectives, real kernel efficiency and queueing add cost; this is not a TTFT prediction.')
    return page('Prefill reuses weights across thousands of tokens',body,
      'P = 405B and T = 8,192 give 2PT = 6.63552 PFLOPs for the leading dense forward work. Do not use the training approximation 6PT. For one unchunked prompt sent through TP8 × PP2, each stage does half the work on 8 GPUs: 6.63552e15 / (2 × 8 × 989e12) = 0.41933 s; the two stages sum to 0.83867 s. Using all 16 GPU peaks at once would assume the same prompt occupies both sequential stages simultaneously. Independent prompts, or a supported pipelined chunk schedule, can improve steady-state utilization; state those assumptions separately. The weight-only streaming floor is 30.2 ms. Under a weight-dominated linear model, BF16 arithmetic intensity is about 8,192 FLOP/byte, far above 989 TFLOP/s divided by 3.35 TB/s ≈ 295 FLOP/byte. Full prefill also includes causal attention, whose cost grows quadratically with sequence length; 2PT alone is not an exact model-wide count. Some embedding/output effects and kernel tiling also make the parameter-count approximation imperfect.',[BOOK,H100,week3(11,'PP scheduling')])

def context():
    body=text(75,191,'At S = 131,072, one sequence has 67.65 GB of unique BF16 KV.',32,700)
    body+=text(75,240,'Consider B = 4 long requests per replica. Keep the same 16 GPUs.',29)
    body+=table(75,280,1450,['Layout','KV / GPU','Weights + KV + 8 GB reserve'],[
      ['TP8 × PP2','16.91 GB','≈ 75.5 GB'],
      ['TP16 · duplicated KV','33.82 GB','≈ 93.0 GB → exceeds 80 GB'],
      ['TP16 + DCP2','16.91 GB','≈ 76.1 GB']
    ],ratios=[.38,.20,.42],row_h=67,size=28)
    body+=text(75,593,'DCP2: split each duplicated head’s cached positions across its two ranks.',30,700,color=BLUE)
    body+=label_box(75,628,640,57,'Rank A: half of the cached positions',size=27)
    body+=label_box(750,628,640,57,'Rank B: the remaining positions',size=27)
    body+=text(75,721,'Distributed attention must exchange query information and combine partial softmax results.',27)
    body+=takeaway('Context parallelism trades extra attention communication for less KV per GPU.', 'In this vLLM DCP layout, TP ranks are reused: TP16 + DCP2 still uses 16 GPUs, not 32.')
    return page('Long contexts make KV memory a limiting factor',body,
      'The canonical maximum context is 131,072 tokens, including generated output; the example is a near-limit cache snapshot, not a 131,072-token prompt with unlimited additional output. Unique BF16 KV = 516,096 × 131,072 = 67.645734912 GB per request. At B = 4, TP8 × PP2 divides the total KV by 16: 16.911433728 GB per GPU. Plain TP16 with duplicated heads divides it only by 8: 33.822867456 GB per GPU. Add approximately 50.625 GB of weights for TP8 × PP2 or 51.153 GB for TP16, including duplicated K/V projections, and the hypothetical 8 GB reserve. In vLLM 0.19.1, decode context parallelism for GQA reuses the TP group: TP16 + DCP2 divides cached positions between ranks that would otherwise duplicate a KV head. The 2× reduction is a logical KV-storage calculation, not a tested 405B deployment benchmark. It requires a compatible model, version, and attention backend. The implementation interleaves cached positions as the history grows; the diagram shows disjoint shares, not contiguous halves. The distributed attention path exchanges queries/results and merges softmax statistics, so saved memory does not establish a latency improvement. Week 3 CP provides the conceptual sequence partition; this versioned DCP implementation nests within TP rather than adding GPUs. Long-prefill context parallelism is a separate scheduling/communication choice.',[DCP,CONFIG,week3(12,'context parallelism')])

def fp8():
    body=text(75,193,'Meta’s mixed FP8 recipe: roughly 487 GB of weights; BF16 KV remains separate.',30,700)
    body+=text(75,240,'Attention and the first / last Transformer layers stay in higher precision.',29)
    body+=table(75,282,1450,['Deployment','Weights / GPU','Replicas','8k requests / replica'],[
      ['BF16 · TP8 × PP2','≈ 50.6 GB','4','≈ 80'],
      ['Mixed FP8 · TP8','≈ 61 GB','8','≈ 20']
    ],ratios=[.35,.23,.15,.27],row_h=77,size=27)
    body+=text(75,554,'Capacity estimates: same 80 GB GPUs and assumed 8 GB runtime reserve.',28)
    body+=text(75,612,'FP8 TP8: floor((80 − 61 − 8) / 0.5285) ≈ 20 requests per replica.',31,700,color=BLUE)
    body+=text(75,667,'Doubling replicas spends memory on more model copies: total KV capacity can fall.',29)
    body+=takeaway('FP8 can remove the cross-node pipeline, but concurrency still needs a memory budget.', 'Validate the chosen precision recipe and task quality. These counts are capacity estimates, not throughput.')
    return page('FP8 can fit a full replica on one node',body,
      'Weight precision changes here, while KV stays BF16. The actual canonical architecture has approximately 405.853 billion parameters. Meta quantizes the three FFN matrices in the 124 middle layers: Q = 124 × 3 × 16384 × 53248 = 324.538 billion parameters. Attention, endpoint layers, embeddings and output weights remain BF16 in the cited loader. Stored weight bytes are Q + 2(P − Q) ≈ 487.168 GB, before scales; TP8 row scales bring the estimate to roughly 487.286 GB. Round to 61 GB per GPU for the capacity example. A single 8,192-token request uses 4.228 GB of BF16 KV across the TP8 replica, or 0.5285 GB per GPU. With the illustrative 8 GB reserve, the rounded budget fits about 20 such requests per replica. Ignoring scales narrowly suggests 21, which is too optimistic. Four BF16 replicas have about 320 slots at this context snapshot; eight mixed-FP8 replicas have about 160. This is not evidence that FP8 reduces throughput: latency, pipeline overhead, batching and kernel speeds also change. Keeping four replicas and using the saved weight memory for KV would be another valid design. NVIDIA NIM documents a separately optimized FP8 profile on eight H100 SXM GPUs; its recipe must not be equated with Meta’s. These analytical memory estimates do not establish runtime feasibility for every buffer shape or backend. Measure peak allocation and task quality with the actual checkpoint.',[FP8_CODE,NIM])

def launch():
    body=text(75,193,'Make one two-node BF16 replica work before replicating it four times.',32,700)
    body+=text(75,241,'Example: a Ray cluster restricted to two nodes; eight H100s visible on each.',28)
    body+=code(75,282,1450,312,'vllm serve meta-llama/Llama-3.1-405B-Instruct \\\n  --dtype bfloat16 \\\n  --tensor-parallel-size 8 \\\n  --pipeline-parallel-size 2 \\\n  --distributed-executor-backend ray \\\n  --max-model-len 16384 \\\n  --max-num-seqs 32',29)
    body+=text(75,642,'Use four isolated two-node groups; route requests to their four serving endpoints.',30)
    body+=text(75,695,'Verify rank placement, checkpoint access, peak memory and inter-node transfers before load tests.',27)
    body+=takeaway('These flags express the baseline layout; they do not establish its achieved performance.', '16k total context leaves output room for an 8k prompt. The batch cap is an initial test setting.')
    return page('Turn the layout into a serving configuration',body,
      'This is a configuration example based on vLLM 0.19.1 documentation, not a deployment executed in this session. Prerequisites are approved access to the model checkpoint, identical supported container/CUDA/NCCL/software versions on both servers, checkpoint availability, a two-node Ray cluster with 16 visible GPUs, and a configured inter-node network. Restrict one replica to exactly its pair of nodes, via isolated clusters or explicit resource placement; four unconstrained launches into one shared cluster can give unintended placement. Confirm each 8-rank TP group stays inside a server and PP links the two groups. Repeat across node pairs 0–1, 2–3, 4–5, and 6–7. The setting max-model-len = 16384 includes prompt plus generated tokens. The setting max-num-seqs = 32 is an initial concurrency cap rather than proof that every 32-request, 16k-context workload fits with all buffers. The earlier capacity example was a snapshot at 8,192 cached positions; recalibrate for allowed growth. Runtime GPU memory utilization and KV allocation settings are not equated to the teaching reserve of 8 GB. Prefix caching, speculative decoding, and quantization should be introduced only after this baseline is measured so their effects can be isolated.',[VLLM,week3(17,'combined layout')])

def choose():
    body=text(75,194,'Start with BF16 TP8 × PP2 × 4 replicas; then test the bottleneck that matters.',30,700)
    body+=table(75,240,1450,['Observed constraint','Controlled comparison','Evidence for changing the layout'],[
      ['Single-request token latency','TP8 × PP2  vs.  TP16','Lower ITL after network costs'],
      ['Long-context memory','TP16  vs.  TP16 + DCP2','Less KV without excessive ITL'],
      ['Cluster serving throughput','4 BF16  vs.  8 FP8 replicas','More tokens/s at equal latency SLO'],
      ['Pipeline mostly idle','One  vs.  several microbatches','Better overlap at acceptable ITL']
    ],ratios=[.30,.32,.38],row_h=75,size=26)
    body+=text(75,671,'Use the same request trace, output lengths and cache state; record TTFT, ITL, memory and throughput.',26)
    body+=takeaway('A deployment is a choice of precision, shard placement and replica count for a workload.', 'Capacity calculations eliminate impossible choices; GPU and service traces choose among the feasible ones.')
    return page('Which layout would we actually choose?',body,
      'This final slide returns to the original 64-GPU decision. The baseline is defensible from topology and Meta’s deployment, but is not proven optimal on an unspecified cluster. First compare 16-GPU replicas with fixed model precision, contexts, arrival traces, and cache state. For lower single-request ITL, TP16 may remove the PP serial-stage penalty, but only if collective overhead does not erase the gain. For long-context GQA, test TP16 + DCP2 and inspect attention communication. For FP8, revalidate quality, actual checkpoint bytes, kernel choices, and independent replica load distribution. At cluster scale, compare aggregate output tokens/s while holding explicit p95 TTFT and p95 ITL objectives, rather than comparing raw throughput alone. A peak-bandwidth bound is not a service-level promise. Use a profiler to separate compute, HBM traffic, exposed collectives, and pipeline gaps. This synthesizes Week 3: TP splits operators, PP splits layers, CP splits context, and replicas split independent requests. The 405B model is dense, so expert parallelism is not applicable. Training-oriented FSDP/ZeRO optimizer sharding is not the default for this persistent-weight inference case.',[BOOK,META,week3(18,'topology'),VLLM])

def get_slides():
    return [problem(),fit(),placement(),tp_layer(),capacity(),tp16(),pipeline(),decode(),prefill(),context(),fp8(),launch(),choose()]
