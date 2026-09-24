"""Five diagram-led CP / EP overview slides; independent of the detailed deck."""
from .common import *

CP = ('Megatron CP', 'https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html')
RING = ('Ring Attention', 'https://arxiv.org/abs/2310.01889')
MIXTRAL = ('MoE architecture', 'https://arxiv.org/abs/2401.04088')
EP = ('vLLM expert parallelism', 'https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/')
DISPATCH = ('Megatron token dispatch', 'https://docs.nvidia.com/megatron-core/developer-guide/0.19.0/apidocs/core/core.transformer.moe.token_dispatcher.html')


def _bar(x, y, width, label, color=BLUE, height=62):
    return rect(x, y, width, height, '#edf3f7' if color == BLUE else '#eef5f3', color) + text(
        x + width/2, y + height/2 + 10, label, 30, 600, color, 'middle')


def _ring_scene(phase):
    s = text(75, 258, '1. Read local K/V' if phase == 0 else '2. Exchange K/V; preserve the causal mask', 31, 700, BLUE)
    for gpu, x in [(0,75),(1,845)]:
        s += rect(x,295,680,333)
        s += text(x+26,335,f'GPU {gpu}',31,700,BLUE)
        s += _bar(x+26,357,628,f'Q stays: positions {"1–4" if gpu == 0 else "5–8"}')
        kv = ('1–4' if gpu == 0 else '5–8') if phase == 0 else ('5–8' if gpu == 0 else '1–4')
        s += _bar(x+26,439,628,f'{"Local" if phase == 0 else "Remote"} K/V: {kv}',TEAL)
        if phase == 0:
            s += text(x+26,546,'Use visible keys; accumulate attention.',29)
            s += text(x+26,594,'Keep the weighted sum and normalizer.',29)
        else:
            s += text(x+26,546,'Future keys: masked.' if gpu == 0 else 'Past keys: merge their contribution.',29)
            s += text(x+26,594,f'Final outputs for positions {"1–4" if gpu == 0 else "5–8"}',29,600)
    if phase:
        s += arrow(755,458,845,458,TEAL) + arrow(845,484,755,484,TEAL)
    return s


def _ownership_card(x, y, first, second, width=370):
    s = rect(x,y,width,145)
    for j,(label,color) in enumerate([(first,BLUE),(second,TEAL)]):
        yy = y+16+66*j
        s += text(x+12,yy+33,f'G{j}',29,700,color)
        s += rect(x+63,yy,width-78,50,'#edf3f7' if j == 0 else '#eef5f3',color)
        s += text(x+63+(width-78)/2,yy+33,label,28,500,color,'middle')
    return s


def get_slides():
    slides = []
    s = text(75,207,'Context parallelism (CP) divides one long sequence across GPUs.',31)
    s += text(75,274,'One sequence: positions 1–8',31,700,BLUE)
    for i in range(8):
        color = BLUE if i < 4 else TEAL
        s += _bar(75+i*181.25,301,181.25,str(i+1),color,62)
    s += arrow(437,364,437,423) + arrow(1162,364,1162,423,TEAL)
    for gpu,x,color,positions in [(0,75,BLUE,'1–4'),(1,845,TEAL,'5–8')]:
        s += rect(x,425,680,245)
        s += text(x+25,469,f'GPU {gpu}',33,700,color)
        s += _bar(x+25,493,630,f'Activations for positions {positions}',color)
        s += label_box(x+25,578,630,63,'Same model weights',size=30)
    s += takeaway('DP divides different examples. CP divides the context of the same example.',
                   'Weights are replicated when CP is used alone; attention must still connect the token shards.')
    slides.append(slide('Context parallelism divides one long sequence',s,
        'Begin with the workload: a single long sequence can create too much activation or KV storage for one device. CP divides its token positions among cooperating GPUs; these are not independent requests. The eight-position schematic represents a much longer sequence. In pure CP the model weights are replicated. In combined TP/PP/CP, CP peers instead hold the same TP/PP parameter slice. Local token-wise operations can run on owned rows, but attention requires remote key/value information. DP divides independent training examples or requests rather than cutting a single sequence. No batch-size or parameter-memory speedup is implied by this toy. Figure is an original teaching diagram.',
        [CP,RING],section='Context and expert parallelism'))

    s = text(75,202,'Two shards of the same causal sequence. Queries stay; K/V blocks move.',30)
    s += steps('overview-ring',[_ring_scene(0),_ring_scene(1)])
    s += takeaway('Combine all allowed blocks with one global softmax weighting.',
                   'The result matches full-context attention; local softmax outputs cannot simply be averaged.')
    slides.append(slide('Ring attention brings remote keys to each query',s,
        'This two-frame schematic uses the same ownership as the preceding slide. Frame 1: GPU 0 queries positions 1–4 against its local keys, preserving causal visibility; GPU 1 queries positions 5–8 against its local keys. Each query has its own weighted sum and softmax normalizer. Frame 2: the KV shards are exchanged while Q and accumulated statistics remain on their owners. GPU 0 can skip the received future-key block. GPU 1 adds the past-key block. The two contributions use one global normalizer, with stable maximum-rescaling as taught in Week 2; independently normalized outputs must not be averaged. Causal masking is based on original positions, not physical storage order. With more GPUs the ring repeats exchanges until every query has processed every permitted key block. Real implementations may skip masked computation and overlap transfer with computation; the figure does not promise perfect balance or hidden communication. No full score matrix is materialized. The two-GPU exchange arrows depict KV movement, not movement of model weights.',
        [RING,CP],section='Context and expert parallelism'))

    s = text(75,202,'Megatron sequence parallelism (SP) complements tensor parallelism.',30)
    for x,label in [(260,'Attention'),(705,'Norm / dropout'),(1150,'MLP')]:
        s += text(x+185,261,label,31,700,BLUE,'middle')
    s += lines(75,335,['CP','alone'],30,43,weight=700,color=BLUE)
    for x in [260,705,1150]:
        s += _ownership_card(x,288,'Tokens 1–4','Tokens 5–8')
    s += arrow(630,359,705,359) + arrow(1075,359,1150,359)
    s += text(260,474,'CP attention exchanges K/V; token ownership persists.',28)
    s += lines(75,573,['TP','+ SP'],30,43,weight=700,color=BLUE)
    s += _ownership_card(260,521,'All tokens; heads 0–1','All tokens; heads 2–3')
    s += _ownership_card(705,521,'Tokens 1–4','Tokens 5–8')
    s += _ownership_card(1150,521,'All tokens; features a','All tokens; features b')
    s += arrow(630,593,705,593) + arrow(1075,593,1150,593)
    s += takeaway('CP distributes long-context attention. Megatron SP reduces replication around TP.',
                   'G0/G1 = GPUs. Simplified layer fragment; SP terminology varies.')
    slides.append(slide('CP and Megatron SP shard different operations',s,
        'Read each row as the same simplified Transformer fragment, not as a full layer specification. The original model may have pre-norm, residual paths, and multiple normalization operations; those details are omitted to focus on activation ownership. G0 and G1 mean GPU 0 and GPU 1. Upper row: pure CP retains token ownership across attention and token-wise operators; attention exchanges remote KV to satisfy dependencies. Lower row: Megatron’s TP-associated SP uses token partitions for token-wise operations such as normalization and dropout. Inside TP attention and linear regions, GPUs process all token positions but different heads or matrix shards. The two “head shard” labels denote different head subsets, and the weight shards also differ. A typical forward transition from TP computation to the SP norm region uses ReduceScatter, then AllGather before the next TP region. This is not a claim that every activation in TP is full-width, nor that all methods named sequence parallelism behave this way. DeepSpeed Ulysses uses the same name for another mechanism. CP and TP-associated SP can coexist; the separate rows isolate their roles.',
        [CP],section='Context and expert parallelism'))

    s = text(75,202,'MoE (mixture of experts): four MLPs on two GPUs; A/B start on GPU 0.',29)
    s += text(75,246,'Top-1 routing selects one expert per token. Expert weights stay on their owners.',29)
    s += _bar(75,332,180,'Token A',TEAL)
    s += _bar(75,467,180,'Token B',BLUE)
    s += label_box(350,378,225,104,'Router',size=32)
    s += arrow(255,363,350,405,TEAL) + arrow(255,498,350,455)
    for gpu,y,selected,other,color in [(0,287,'E0','E1',BLUE),(1,496,'E2','E3',TEAL)]:
        s += rect(735,y-17,390,177)
        s += text(754,y+17,f'GPU {gpu}',29,700,color)
        s += _bar(754,y+35,170,selected,color,59)
        s += label_box(943,y+35,162,59,other+' (idle)',fill='#f7f7f7',size=28)
    # Curves route into the left selected expert without crossing either expert box.
    s += arrow(575,405,754,351,BLUE) + arrow(575,455,754,560,TEAL)
    s += text(578,328,'B: local',28,600,BLUE)
    s += text(578,600,'A: remote',28,600,TEAL)
    s += text(1262,291,'Return to GPU 0',28,700)
    # Both return paths leave below the expert row, avoiding the idle expert boxes.
    for y,label,color in [(388,'B output',BLUE),(597,'A output',TEAL)]:
        s += line(839,y-7,839,y+26,color,2)
        s += arrow(839,y+26,1255,y+26,color)
        s += _bar(1255,y-4,250,label,color,60)
    s += takeaway('Expert parallelism places different expert MLPs on different GPUs.',
                   'Send token activations to the chosen expert; return the output to the original token owner.')
    slides.append(slide('Route tokens to experts, then return the outputs',s,
        'An MoE expert is a complete learned MLP, not an attention head or a GPU. This overview chooses top-1 routing so each token has one destination and one returned vector; a top-k model duplicates token work across selected experts and combines their outputs with router weights. Here both original tokens A and B are owned by GPU 0. GPU 0 also holds experts E0/E1, while GPU 1 holds E2/E3. The router chooses E0 for B and E2 for A. B’s expert execution is local. A’s activation crosses to GPU 1, is processed by E2, and its resulting vector returns to GPU 0. Both outputs are restored to the original token order. E1 and E3 receive no tokens in this step but their weights remain allocated. Dispatch and return can be implemented with all-to-all style exchanges; physical network traffic is needed only for remote routes. The paths out of the expert boxes are return paths to the original worker, not a transfer of expert weights or KV histories. Router weights are omitted in this top-1 ownership illustration.',
        [MIXTRAL,DISPATCH],section='Context and expert parallelism'))

    s = text(75,203,'Now A, B, C choose E0 on GPU 0; D chooses E2 on GPU 1.',31)
    s += text(75,260,'Same expert sizes; different token loads.',31,700,BLUE)
    start, unit = 375, 286
    for gpu,y,labels in [(0,328,['A','B','C']),(1,490,['D'])]:
        s += lines(75,y+39,[f'GPU {gpu}',f'Expert E{0 if gpu == 0 else 2}'],29,42,weight=700)
        s += label_box(233,y,118,86,'Route',fill='#f2f2f2',size=28)
        s += arrow(351,y+43,start,y+43)
        for j,label in enumerate(labels):
            s += _bar(start+j*unit,y,unit, label, BLUE if gpu == 0 else TEAL,86)
        if gpu == 1:
            s += rect(start+unit,y,2*unit,86,'white',LINE,attrs='stroke-dasharray="7 5"')
            s += text(start+2*unit,y+54,'Waiting for GPU 0',30,600,MUTED,'middle')
        s += arrow(start+3*unit,y+43,1260,y+43)
        s += label_box(1260,y,245,86,'Return outputs',fill='#f2f2f2',size=28)
    s += arrow(375,643,1505,643)
    s += text(936,690,'Time →',29,anchor='middle')
    s += takeaway('The busiest expert owner can delay the whole synchronized step.',
                   'Routing also costs communication. Schematic equal token costs; real experts batch tokens in GEMMs.')
    slides.append(slide('Uneven routing leaves some GPUs waiting',s,
        'Continue the top-1, four-expert placement from the previous slide, but with four token activations. Assign A/B/C to E0 and D to E2; E1 and E3 are idle. GPU 0 receives three token-expert assignments while GPU 1 receives one. The bars are a schematic of equal token work, not a measured timeline or an assertion that production kernels execute tokens one by one. Real expert MLPs batch token rows into GEMMs, and runtime depends on their shapes and efficiency. The waiting region illustrates a synchronized dispatch/compute/return schedule whose completion depends on the busiest rank. Useful overlap and asynchronous implementations may change the visible wait but do not remove the unequal work. Communication can dominate even with balanced routes when per-expert batches are small. Changing placement or adding replicas of a hot expert can reduce imbalance at the cost of memory and routing complexity. Arbitrarily choosing a different trained expert just to balance work changes model semantics and is not a generally valid inference optimization. Diagnose expert token counts and communication time rather than assuming evenly sized experts imply evenly used GPUs.',
        [EP],section='Context and expert parallelism'))
    return slides
