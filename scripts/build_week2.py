"""Build the self-contained Week 2 HTML presentation and companion notes.

Native SVG teaching diagrams retain the Week 1 design and support interactive
steps. Source figures are embedded. No external fonts, scripts, or network calls are required.
"""
from pathlib import Path
from html import escape as esc
import base64

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'site/slide-assets/week2'
BLUE, INK, MUTED, PALE, LINE = '#245675', '#172329', '#58656d', '#edf3f7', '#aec3d1'
TEAL, WARM = '#397b71', '#a36330'
SOURCES = {
    'inference': ('CS336 Lecture 10', 'https://cs336.stanford.edu/lectures/?trace=lecture_10'),
    'flash': ('CS336 Lecture 5, pp. 52–54', 'https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=52'),
    'kernels': ('CS336 Lecture 6', 'https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py'),
    'berkeley': ('Berkeley L18, PDF pp. 9–16', 'https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf#page=9'),
    'berkeley_chunk': ('Berkeley L18, PDF p. 18', 'https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf#page=18'),
    'berkeley_bench': ('Berkeley L18, PDF pp. 47–53', 'https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf#page=47'),
    'berkeley_pd': ('Berkeley L18, PDF pp. 39–41', 'https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf#page=39'),
    'normalizer': ('Online normalizer, 2018', 'https://arxiv.org/abs/1805.02867'),
    'orca': ('Orca, §4.1', 'https://www.usenix.org/system/files/osdi22-yu.pdf#page=5'),
    'distfig': ('DistServe, Fig. 2 (OSDI 2024)', 'https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=5'),
    'paper': ('FlashAttention paper', 'https://arxiv.org/abs/2205.14135'),
    'graphs': ('PyTorch CUDA Graphs', 'https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs'),
    'timing': ('PyTorch CUDA timing', 'https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution'),
    'matmul': ('NVIDIA matmul performance', 'https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-and-memory-bounds'),
    'h100': ('H100 SXM specifications', 'https://www.nvidia.com/en-sg/data-center/h100/'),
    'peak': ('NVIDIA dense compute peaks', 'https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput'),
    'paged': ('PagedAttention paper', 'https://arxiv.org/abs/2309.06180'),
    'vllm': ('vLLM: PagedAttention', 'https://vllm-project.github.io/2023/06/20/vllm.html'),
    'distserve': ('DistServe (OSDI 2024)', 'https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin'),
    'pd': ('vLLM: disaggregated prefill', 'https://docs.vllm.ai/en/latest/features/disagg_prefill/'),
    'triton_gemm': ('Triton: Matrix Multiplication', 'https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html'),
    'kv_size': ('Scaling Book: KV memory', 'https://jax-ml.github.io/scaling-book/inference/'),
}

def text(x, y, value, size=30, weight=400, color=INK, anchor='start', attrs=''):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}" {attrs}>{esc(str(value))}</text>'

def lines(x, y, values, size=30, gap=None, **kwargs):
    if isinstance(values, str): values = values.split('\n')
    return ''.join(text(x,y+i*(gap or size*1.4),v,size,**kwargs) for i,v in enumerate(values))

def rect(x,y,w,h,fill='white',stroke=LINE,attrs=''):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" {attrs}/>'

def line(x1,y1,x2,y2,color=LINE,width=1.5,dash=''):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'

def arrow(x1,y1,x2,y2,color=BLUE):
    return line(x1,y1,x2,y2,color,2)+f'<path d="M {x2-8} {y2-5} L {x2} {y2} L {x2-8} {y2+5}" fill="none" stroke="{color}" stroke-width="2"/>' if y1==y2 else f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{color}" stroke-width="2" fill="none" marker-end="url(#arrow)"/>'

def label_box(x,y,w,h,label,fill=PALE,size=28):
    return rect(x,y,w,h,fill)+text(x+w/2,y+h/2+size*.34,label,size,600,anchor='middle')

def takeaway(value, second=None):
    s=line(75,740,1525,740)+text(75,785,value,30,700,color=BLUE)
    if second: s+=text(75,820,second,23,color=MUTED)
    return s

def control(kind,x=75,y=667,extra=''):
    return f'''<foreignObject x="{x}" y="{y}" width="1450" height="64"><div xmlns="http://www.w3.org/1999/xhtml" class="interaction" data-deck-ignore-keys="true"><button type="button" data-step-prev="{kind}">Previous step</button><button type="button" data-step-next="{kind}">Next step</button><span data-step-label="{kind}" aria-live="polite"></span>{extra}</div></foreignObject>'''

def table(x,y,width,headers,rows,ratios=None,row_h=74,size=28):
    ratios=ratios or [1/len(headers)]*len(headers)
    xs=[x]; out=rect(x,y,width,row_h,PALE,'none')
    for r in ratios: xs.append(xs[-1]+width*r)
    for i,h in enumerate(headers): out+=text(xs[i]+14,y+46,h,size,700)
    out+=line(x,y+row_h,x+width,y+row_h)
    for j,row in enumerate(rows):
        yy=y+(j+1)*row_h
        for i,v in enumerate(row): out+=text(xs[i]+14,yy+46,v,size)
        out+=line(x,yy+row_h,x+width,yy+row_h,'#c7cdd1')
    return out

def matrix(x,y,name,rows=4,cols=4,cell=44,group='',values=None):
    out=text(x,y-17,name,27,700)
    for r in range(rows):
        for c in range(cols):
            out+=f'<g data-matrix="{group}" data-row="{r}" data-col="{c}">'+rect(x+c*cell,y+r*cell,cell,cell,'white')
            if values is not None: out+=text(x+(c+.5)*cell,y+(r+.67)*cell,values[r][c],23,anchor='middle')
            out+='</g>'
    return out

slides=[]
def add(title, minutes, body, notes, sources=()):
    slides.append(dict(title=title,minutes=minutes,body=body,notes=notes,sources=list(sources)))

# The cover content is filled after ordering and timing are resolved.
add('LLM inference performance',.5,'','')

# 2
add('A prompt produces the first token',2,
    text(75,198,'Autoregressive generation: each new token depends on the preceding tokens',30)+
    '<g id="generation-scene"></g>'+control('generation')+
    takeaway('Prefill predicts token 1. Each decode step predicts one more token.'),
    'Use Next step twice. The four words are illustrative token labels, not a tokenizer demonstration. Prefill processes all prompt positions with a causal mask and returns logits at the final prompt position. Sampling those logits yields the first generated token. Decode step 1 feeds that generated token, adds its K/V at each layer, and predicts token 2. Decode step 2 feeds token 2 and predicts token 3. The newest sampled token enters the KV cache only when it is subsequently processed. Cache blocks represent per-token K/V across layers, not complete hidden activations. Times are schematic.',('inference','berkeley'))

# 3
add('New token rows are only part of the workload',1,
    text(75,193,'M counts new hidden-state rows. B = requests, S = prompt tokens per request.',29)+
    table(75,235,1450,['At one Transformer layer','Prefill (no cached prefix)','One ordinary decode step'],[
        ['New rows through projections / MLP','M = B × S','M = B'],
        ['Example: B = 2, S = 4','8 new rows','2 new rows'],
        ['Weights needed','Layer weights','The same layer weights'],
        ['Attention uses','Prompt K/V, with a causal mask','Cached + current K/V'],
        ['KV cache update','Store B × S new positions','Append B new positions'],
    ],[.35,.325,.325],row_h=74,size=26)+
    text(75,713,'KV caching avoids recomputing old positions. Each new query still uses their K/V.',29,700)+
    takeaway('New token count does not measure total memory traffic.',
              'Memory traffic here means HBM reads/writes inside the GPU, not sending token IDs from the CPU.'),
    'M is a count of newly processed token positions, not bytes transferred. At one layer a dense projection or MLP acts on M hidden-state vectors, each with many scalar values. In an uncached prefill every prompt position must be processed to build its per-layer K/V and support later causal positions, even when only the last position produces the next-token prediction. For B=2 and S=4 this is eight positions. In ordinary autoregressive decode, each request feeds its most recently generated token, so two requests supply two new positions. Earlier positions do not need to pass through the projections and MLP again because their K/V has been retained. However each new query still attends to the relevant cached keys and values plus the current position. Weight reads also remain: model weights already reside in GPU HBM, but arithmetic needs to load weight tiles into on-chip storage. This is separate from CPU-to-GPU transfer of token IDs. New hidden-state traffic, weight traffic, historical KV reads, new KV writes, and other intermediates all contribute to physical memory traffic. The cache is not a promise that all historical KV fits on chip. Prefix-cache hits, speculative multi-token decoding, and ragged prompt batches are outside this simple shape comparison.',('inference','berkeley'))

# More token rows reuse the same dense-layer weights.
add('Many token rows reuse the same weights',2,
    text(75,195,'One dense layer: Y = XW. Each row of X is a token processed in this pass.',30)+
    text(75,269,'Decode, one request',30,700,color=BLUE)+
    label_box(75,315,255,60,'1 token row',size=29)+text(369,355,'×',37)+
    label_box(430,282,365,145,'Weights W',size=31)+arrow(823,355,925,355)+label_box(955,315,255,60,'1 output row',size=29)+
    lines(1260,321,['The weight matrix', 'is still read, for', 'just one new row.'],26)+
    line(75,465,1525,465)+text(75,513,'Prefill, one long prompt',30,700,color=BLUE)+
    ''.join(label_box(75,543+j*37,255,33,'Token rows' if j==1 else '',size=24) for j in range(4))+
    text(369,620,'×',37)+label_box(430,543,365,145,'The same W',size=31)+arrow(823,620,925,620)+
    ''.join(label_box(955,543+j*37,255,33,'Output rows' if j==1 else '',size=24) for j in range(4))+
    lines(1260,573,['Reuse a weight tile', 'across many rows', 'while it is on chip.'],26)+
    takeaway('Prefill performs more arithmetic per weight loaded from HBM.',
              'Small-batch decode can move fewer total bytes yet spend most of its time moving them.'),
    'The difference is matrix-vector versus matrix-matrix work, not different learned weights. With one decode request, a weight contributes to the output for one new token. With a prompt, the same weight contributes to many output rows. Tiled matrix multiplication exploits this reuse in on-chip storage. The entire weight matrix does not need to fit on chip, and real kernels can reload tiles. Only the ideal accounting on the next slides assumes one HBM read for each input. Causal masking restricts attention dependencies, but the tokenwise dense projections and MLP at each layer can still process all available prompt positions together.',('inference','matmul'))

add('Arithmetic intensity connects reuse to the bottleneck',2,
    text(75,194,'BF16 dense matmul: X[M, D] × W[D, F] = Y[M, F]',32,700)+
    table(75,240,1450,['Cost','Ideal accounting'],[
        ['Arithmetic','2MDF FLOPs'],
        ['HBM traffic: read X and W, write Y','2(MD + DF + MF) bytes'],
        ['Arithmetic intensity I','FLOPs / bytes = MDF / (MD + DF + MF)'],
    ],[.46,.54],row_h=78,size=28)+
    text(75,610,'If weight traffic dominates: I ≈ M FLOPs/byte',34,700,color=BLUE)+
    text(75,664,'Compute time ≥ FLOPs / compute peak. Transfer time ≥ bytes / HBM bandwidth.',28)+
    takeaway('Compare the two resource times for the same operation.',
              'The larger time is the ideal bottleneck. Launch overhead and extra traffic can add further limits.'),
    'M is the number of token rows, D the input dimension, and F the output dimension. There are MF output values, each formed from D multiply-add pairs. Counting one multiply and one add as two FLOPs gives 2MDF. BF16 uses two bytes, giving 2MD bytes for X, 2DF for W, and 2MF for Y in this idealized cold-HBM accounting. Divide to obtain intensity. When M is small relative to D and F, DF dominates the denominator, so I≈M. This approximation is not valid for arbitrarily large M. Compute time floor is FLOPs/C, and HBM time floor is bytes/BW. With overlap, the idealized time is their maximum; their crossover is I=C/BW. These are theoretical resource ceilings/lower bounds, not measured runtimes. Actual traffic, cache behavior, launch latency and hardware utilization matter.',('matmul','inference'))

def matmul_cost(m, d=8192, f=8192):
    flops=2*m*d*f
    byte_count=2*(m*d+d*f+m*f)
    return dict(flops=flops,bytes=byte_count,intensity=flops/byte_count,
                compute_us=flops/989e12*1e6,memory_us=byte_count/3.35e12*1e6)

def cost_bars():
    out=''
    for m,y,title in [(1,330,'Decode'),(2048,508,'Prefill')]:
        c=matmul_cost(m)
        out+=text(75,y,title,34,700,color=BLUE)+text(75,y+45,f'M = {m:,}',29)+text(75,y+88,f'I ≈ {c["intensity"]:,.0f} FLOPs/byte',26,color=MUTED)
        for offset,key,label,color in [(0,'compute_us','Compute floor',BLUE),(64,'memory_us','HBM floor',TEAL)]:
            yy=y+offset
            width=c[key]*2
            out+=text(335,yy,label,27)+rect(600,yy-25,width,31,color,'none')
            value=f'{c[key]:.2f}' if c[key]<1 else f'{c[key]:.1f}'
            out+=text(600+width+16,yy,value+' μs',27,700,color=color)
        out+=text(1245,y+56,'HBM dominates' if m==1 else 'Compute dominates',26,700,color=TEAL if m==1 else BLUE)
    return out

add('H100 example: same weights, different bottlenecks',2,
    text(75,190,'H100 SXM: 989 TFLOP/s dense BF16, 3.35 TB/s HBM. Crossover I ≈ 295 FLOPs/byte.',27)+
    text(75,239,'W = 8192 × 8192. BF16 inputs and output. Bars use the same time scale.',28)+
    cost_bars()+line(75,452,1525,452)+
    text(75,663,'Ideal HBM traffic: 134.25 MB for decode, 201.33 MB for prefill.',27,color=MUTED)+
    text(75,706,'Theoretical lower bounds, not timings. Attention, launches, and extra traffic are excluded.',26,color=MUTED)+
    takeaway('2,048× more arithmetic for only ~1.5× the ideal HBM traffic.'),
    'This is one dense linear layer, not the whole model. Use the exact intensity formula rather than I≈M at M=2048. W contains 67,108,864 values and 134,217,728 BF16 bytes. M=1 requires 134,217,728 FLOPs and 134,250,496 ideal HBM bytes: I=0.999756, compute floor 0.135711 μs, HBM floor 40.074775 μs. M=2048 requires 274,877,906,944 FLOPs and 201,326,592 bytes: I=1365.333, compute floor 277.935194 μs, HBM floor 60.097490 μs. Thus the dominant resource floor switches in this model. 989 TFLOP/s is dense BF16 throughput, not the higher sparsity-assisted headline. The skinny M=1 operation cannot attain that Tensor Core peak, so its 0.14 μs compute floor is emphatically not an achievable benchmark. Peak bandwidth is also a ceiling. The ratio 989/3.35≈295 is a FLOPs-per-byte threshold, not an exact batch-size threshold. Real tiles may reread data, and cached data can reduce HBM traffic.',('h100','peak','matmul'))

# 4
add('The KV cache saves work on earlier tokens',1.5,
    text(75,199,'At one layer, for the current token',31,700)+
    label_box(75,245,260,80,'Current hidden state',size=25)+arrow(335,285,400,285)+
    label_box(400,245,295,80,'New q, k, v')+arrow(695,285,785,285)+
    label_box(785,245,310,80,'Attention output')+arrow(1095,285,1160,285)+label_box(1160,245,345,80,'Rest of layer / next layer')+
    rect(400,400,695,163,'white')+text(425,440,'K/V cache in HBM',29,700,color=BLUE)+
    ''.join(label_box(425+i*120,466,108,60, f'Token {i+1}',size=24) for i in range(5))+
    arrow(550,325,550,400)+text(570,367,'Append new k, v',25,color=MUTED)+
    arrow(950,400,950,325)+
    lines(1145,418,['Read cached K/V', 'with the current q', 'to form attention'],27)+
    lines(75,620,['Earlier tokens keep their K/V at every layer.', 'Each request has its own cache. All requests use the same model weights.'],29)+
    takeaway('Caching avoids rebuilding prior K/V. Attention still uses those stored values.'),
    'For a fixed causal model, earlier token representations do not depend on newly generated future tokens, so their K/V can be retained. At each layer we compute the current token’s q, k, and v, append k and v to that layer’s cache, and attend over the relevant past positions plus the current position. Only the current hidden state proceeds through the rest of the network. We normally do not cache old queries for standard next-token decoding. Requests use common read-only model weights but different K/V because their token histories differ.',('inference',))

# 5
add('Question 1: Why did a larger decode batch stop helping?',1,
    lines(75,193,[
        'Same model, GPU, precision, and cached length S. All batches fit in HBM.',
        'KV caching is enabled. Time steady decoding, excluding prefill and queueing.',
    ],29,gap=45)+
    line(75,282,1525,282)+
    text(75,332,'A performance diagnosis scenario',29,700,color=BLUE)+
    text(75,394,'Doubling batch size B barely increases aggregate output tokens/s.',32,700)+
    text(75,450,'A teammate claims: “We hit the compute ceiling. Optimize the GEMMs.”',30)+
    line(75,491,1525,491)+
    text(75,552,'1. How do weight reads and historical KV reads scale with B?',30,700)+
    text(75,616,'2. Does the observation support the teammate’s conclusion?',30,700)+
    text(75,680,'3. Design a controlled experiment that separates the possible limits.',30,700,color=BLUE)+
    text(75,802,'Authored diagnostic scenario. The throughput observation is hypothetical.',25,color=MUTED),
    'Ask for a diagnosis with competing explanations, not a memorized prefill/decode label. This is an authored hypothetical case, not measured data or a reported company interview question. All batches fit without offloading; model, precision and implementation remain fixed. Sequence length is held fixed within a comparison. With ordinary one-token decode, a B-request batch produces B output tokens per step. Little throughput improvement after doubling B means the step duration has risen close to twofold. That can happen because per-request KV traffic scales with B, because arithmetic saturates compute throughput, or because per-request host/dispatch work grows with B. Fixed per-step launch overhead alone would usually allow throughput to grow with B rather than explain a plateau. The observation alone cannot distinguish these. An ideal weight-read budget is amortized over the batch, while unrelated requests generally have separate KV histories. Ask what evidence would disprove the proposed diagnosis before selecting an optimization.',('inference','matmul'))

# 6
add('Solution 1: A throughput plateau does not identify the limit',1,
    text(75,191,'W = weight bytes read per step. K(S) = KV bytes read per request.',28)+
    text(75,249,'Ideal reads per step ≈ W + B × K(S)',34,700,color=BLUE)+
    text(75,302,'Per output token ≈ W/B + K(S)',33,700)+
    text(75,351,'KV-dominated limit: doubling B doubles KV reads and output tokens; throughput can stay flat.',27)+
    table(75,386,1450,['Possible limit','Evidence needed before choosing a fix'],[
        ['Historical KV bandwidth','Attention dominates; HBM bandwidth near its achievable limit'],
        ['Dense-layer compute','Dense kernels dominate near achievable compute throughput'],
        ['Per-request host work','Host time grows with B; GPU idle gaps'],
    ],[.32,.68],row_h=72,size=27)+
    text(75,715,'Sweep B at fixed S, then S at fixed B. Profile attention and dense layers separately.',28,700)+
    takeaway('Faster GEMMs help only if their execution is a material part of the bottleneck.',
              'Ideal reads omit activations and other traffic, and assume one weight read per batch with no prefix reuse.'),
    'Let W be the bytes of weights ideally fetched once during one complete decode step, and K(S) the bytes of required K/V fetched for one request across layers. Both are traffic quantities under stated assumptions, not a measurement of reserved memory. Ideal weight-plus-KV reads are W+B K(S); divide by the B emitted tokens to get W/B+K(S). In a bandwidth-dominated model, step time is approximately (W+B K(S))/BW and aggregate output rate is approximately BW/(W/B+K(S)). If B K(S) dominates W, doubling B approximately doubles both step time and emitted tokens, so throughput approaches BW/K(S) even while bandwidth remains the limit. Compute can instead dominate, and the plateau alone proves neither case. Profile time attributed to dense layers versus attention, measured memory traffic and achieved bandwidth, relevant compute throughput, and host/GPU idle gaps. A large byte count alone does not prove bandwidth saturation. Distinguish bandwidth near an achievable ceiling from memory-latency or parallelism problems. Batch-independent launch overhead by itself would generally be amortized as B grows; per-request host/dispatch work that grows with B can instead limit aggregate throughput. Vary B at fixed S and then S at fixed B, keeping weights, GPU, dtype and timing boundaries fixed. Both attention arithmetic and KV traffic increase with S, so a length sweep alone is not proof of a bandwidth limit. Compare counters and kernel timings together. Real caches, rereads, allocations, ragged lengths, shared prefixes and extra intermediates can change the ideal traffic model. No universal speedup or single numeric batch threshold follows from this exercise.',('inference','matmul'))

# 7
add('Batching affects weight traffic and KV traffic differently',2,
    text(75,195,'Decode processes one new token per request.',30)+line(775,238,775,704)+
    text(75,280,'Dense layers: shared weights',32,700,color=BLUE)+
    label_box(230,323,335,75,'One weight matrix W',size=28)+
    ''.join(label_box(75+i*220,453,195,65,f'Request {i+1}',size=27) for i in range(3))+
    ''.join(arrow(395,399,172+i*220,449) for i in range(3))+
    text(75,584,'Arithmetic grows with B.',29)+text(75,631,'Reuse the same weight tile across B rows.',28)+
    text(75,689,'Weight-dominated BF16: I ≈ B',31,700,color=BLUE)+
    text(825,280,'Attention: independent K/V histories',31,700,color=BLUE)+
    ''.join(label_box(825+i*230,323,209,75,f'KV for request {i+1}',size=23) for i in range(3))+
    lines(825,464,['Per head: B requests, S history positions,', 'h values per key or value, BF16 storage.'],27)+
    text(825,575,'QKᵀ and PV: approximately 4BSh FLOPs',28)+
    text(825,626,'Read K and V: approximately 4BSh bytes',28)+
    text(825,689,'I ≈ 1 FLOP/byte. B cancels.',31,700,color=BLUE)+
    takeaway('Batching improves weight reuse; attention can remain bandwidth-limited.'),
    'For standard multi-head decode attention, consider one query head with its corresponding KV head per request. QK costs approximately 2BSh FLOPs and the weighted-value sum costs another 2BSh. K and V each contain BSh BF16 values, for 4BSh bytes in total. Ignoring the smaller query/output traffic and softmax arithmetic for a long history gives I≈1 FLOP/byte. Batch B appears in both work and KV bytes because different requests have independent histories. Batching does not create the same across-request reuse that exists for model weights. It can still improve attention throughput by improving occupancy, parallelism or effective bandwidth. Shared KV heads, prefix sharing, different precisions and cache residency can change this model; they are outside the current example. Do not generalize the cancellation into a claim that attention cannot be optimized.',('inference',))

# Paged KV allocation connects the per-request cache to batch capacity.
def kv_capacity_rows():
    out=''
    for row,(name,n) in enumerate([('A',6),('B',3),('C',5)]):
        yy=335+row*91
        for start,paged in [(75,False),(825,True)]:
            out+=text(start,yy+27,f'{name}: {n} tokens',26,700)
            count=((n+3)//4)*4 if paged else 12
            for i in range(count):
                # Space between physical blocks is conceptual, not a byte offset.
                xx=start+205+i*34+(i//4)*9 if paged else start+205+i*34
                out+=rect(xx,yy,32,39,BLUE if i<n else 'white')
            if paged:
                for b in range(count//4):
                    out+=rect(start+203+b*145,yy-3,140,45,'none',MUTED)
    return out

add('PagedAttention reduces wasted KV capacity',2,
    lines(75,190,['KV caches grow as tokens are processed.', 'A request’s final length is not known when it arrives.'],29,gap=43)+
    line(775,265,775,713)+
    text(75,295,'Reserve 12 slots per request',30,700,color=BLUE)+
    text(825,295,'Allocate blocks of 4 slots',30,700,color=BLUE)+
    kv_capacity_rows()+
    text(75,632,'36 reserved slots; 14 contain KV',29,700)+
    text(825,632,'20 allocated slots; 14 contain KV',29,700)+
    rect(75,676,24,24,BLUE)+text(112,697,'Stored KV',25)+
    rect(360,676,24,24,'white')+text(398,697,'Allocated but unused',25)+
    text(825,697,'Allocate another block only when needed.',25,color=MUTED)+
    takeaway('Same cached tokens, less reserved memory, room for more requests.',
              'Toy example: one slot represents one token’s KV. The model weights are unchanged.'),
    'Compare an illustrative strategy reserving space for up to 12 tokens per request against on-demand fixed-size KV blocks. Requests A, B, C currently have 6, 3, 5 cached tokens, respectively: 14 used slots. Fixed reservations allocate 36 slots and leave 22 unused. With 4-token blocks, the requests use 2, 1, 2 blocks: 20 allocated slots with 6 unused tail slots. The unused tail of a block can be filled as that request grows. PagedAttention also avoids requiring each request’s physical blocks to be contiguous, reducing external fragmentation; the next slide explains the lookup. The reservation baseline is a teaching example, not a claim that every contiguous allocator always reserves a fixed maximum. Actual blocks hold vector-valued K/V at model layers, and metadata has a small additional cost. Four tokens per block is a toy choice, not a recommended deployment setting. Paging does not compress K/V, move it automatically to CPU memory, or remove the reads needed for attention. More efficient capacity use permits more concurrent requests when KV capacity is the limiting resource; throughput gains depend on the workload and kernels.',('paged','vllm'))

add('PagedAttention: logical blocks and physical storage',2.5,
    text(75,181,'A block table maps token positions to KV blocks in GPU memory.',29)+
    text(75,214,'Independent example: A has 6 tokens; C has 8; B arrives later.',24,color=MUTED)+
    '<g id="paging-scene"></g>'+control('paging')+
    takeaway('Attention follows the block table to read the request’s valid K/V.',
              'Physical blocks can be scattered in HBM. Paging still reads the required history.'),
    'Use Next step four times. One physical block holds K/V for four cached token positions. Request C keeps physical blocks 0 and 2 throughout. Initially A has 6 cached tokens: logical blocks 0 and 1 map to physical blocks 4 and 1. A grows to 8 tokens by filling its existing tail block; no old K/V moves. The ninth cached token allocates another free block, physical block 5, with three tail slots unused. A then finishes and releases its unshared blocks. Request B arrives with 3 cached tokens and reuses physical block 4; old A contents are no longer valid. The example deliberately disables prefix retention and sharing; reference-counted shared or cached blocks need different release rules. Token labels count processed positions, not tokens merely sampled. A logical block preserves token order despite nonconsecutive physical addresses. The attention kernel uses the table and sequence length to fetch valid historical K/V, never treating unused slots as valid keys. A storage block here is not a CUDA thread block. PagedAttention can coexist with tiled attention kernels: it addresses KV allocation and lookup, while FlashAttention addresses attention computation and intermediate IO.',('paged','vllm'))

# 8

# 9

# 10

# 11
add('Ordinary attention materializes large intermediates',1.5,
    text(75,198,'After the Q, K, and V projections, one attention head computes:',30)+
    label_box(75,253,365,90,'A = QKᵀ / √d',size=34)+arrow(440,298,555,298)+
    label_box(555,253,400,90,'P = softmax(A + mask)',size=30)+arrow(955,298,1075,298)+label_box(1075,253,430,90,'O = PV',size=34)+
    matrix(180,433,'Scores A',cell=51)+matrix(720,433,'Probabilities P',cell=51)+
    lines(1060,452,['Each matrix has S × S entries', 'for a prompt of S tokens.', '', 'A naïve implementation', 'writes and rereads these', 'intermediates in HBM.'],27,gap=37)+
    text(75,702,'Illustration: a 4-token prompt. Larger prompts make the matrices grow quadratically.',26,color=MUTED)+
    takeaway('FlashAttention avoids storing these full matrices in HBM.'),
    'A denotes scores and S denotes sequence length. Q, K, V have already been projected; attention proper computes two matrix products separated by a row softmax. The drawing shows dense matrices. Causal attention masks future keys, and optimized kernels can skip corresponding work. The problem is intermediate storage and traffic. Computing more efficiently does not require changing the attention definition. Modern PyTorch SDPA may already choose a fused implementation, so ordinary means the explicit separate-operator baseline.',('flash','paper'))

# 12

# 13

# 14
# 15
# A short serving extension follows the single-GPU mechanisms.

add('Prefill–decode disaggregation',2,
    text(75,195,'Different GPU pools run the phases; the request’s KV state moves between them.',29)+
    label_box(75,337,175,93,'Prompt',size=30)+arrow(250,383,340,383)+
    rect(345,310,350,146,PALE)+text(520,365,'Prefill GPUs',32,700,anchor='middle')+
    text(520,417,'Build prompt KV',28,anchor='middle')+
    arrow(695,383,965,383)+text(830,326,'Transfer prompt KV',25,anchor='middle')+
    text(830,425,'Across model layers',24,color=MUTED,anchor='middle')+
    rect(970,310,350,146,'#e6eeeb')+text(1145,365,'Decode GPUs',32,700,anchor='middle')+
    text(1145,417,'Generate more tokens',27,anchor='middle')+
    arrow(1320,383,1370,383)+label_box(1375,337,150,93,'Tokens',size=29)+
    text(75,503,'Both pools need the model weights; each request also needs a KV handoff.',27,color=MUTED)+
    line(75,536,1525,536)+line(775,561,775,657)+
    text(75,579,'Independent resource tuning',29,700,color=BLUE)+
    text(75,628,'Protect decoding from incoming prefill work.',27)+
    text(825,579,'New costs',29,700,color=BLUE)+
    text(825,628,'KV transfer, extra queues, and idle capacity',27)+
    text(75,704,'Transfer-time lower bound = KV bytes / effective link bandwidth',30,700)+
    takeaway('Judge separation by latency targets, workload, and transfer cost.',
              'Different phase bottlenecks alone do not guarantee a speedup.'),
    'PD means prefill–decode disaggregation. Prefill and decode instances use distinct GPU resources and can choose their resource allocation and batch or parallelism strategies independently. They need compatible access to the model weights, often as separate replicas or sharded replicas. For a request, the prefill side builds K/V across layers and passes that state plus relevant request metadata to the decode side. The first sampled token can be produced by prefill and handed over as well; the figure focuses on KV rather than the exact API ownership of first-token delivery. Decode then continues autoregressive generation without recomputing the prompt. A transfer-time lower bound is payload bytes divided by effective link bandwidth; setup, contention, layout conversion and additional queues can add delay. Layerwise transfer may overlap some communication with prefill, so the full transfer duration is not necessarily extra exposed latency. Benefits include reducing prefill-induced tail ITL and independent TTFT/ITL tuning. Gains in goodput—the request rate that meets latency targets—depend on the workload and placement; disaggregation does not inherently increase raw tokens/s. Long KV transfers or poorly balanced pools can outweigh isolation benefits. Compare this design with the preceding chunked-prefill schedule under the same GPU budget and service objectives. Week 4 develops placement and scheduling policies in more detail.',('distserve','berkeley_pd'))

# 16
add('Experiment: sweep batch size on one GPU',1.5,
    text(75,198,'A runnable CUDA notebook accompanies this presentation.',31,700)+
    table(75,243,1450,['Keep fixed','Measure'],[
        ['Model configuration and random weights','Prefill time'],
        ['Prompt length and output length','Average decode-step time'],
        ['GPU, precision, and implementation','Tokens/s per user and across the batch'],
        ['Timing boundaries and warm-up','Peak allocated memory in GB'],
    ],[.5,.5],row_h=72,size=27)+
    f'''<foreignObject x="75" y="630" width="1450" height="92"><div xmlns="http://www.w3.org/1999/xhtml" class="interaction vertical"><a href="https://github.com/zhiweixx/llm-systems-study-group/tree/main/week-2-lab" target="_blank" rel="noopener">Open the notebook, benchmark, and plotting script</a><span>Record t(B). Per-user rate = 1/t(B). Aggregate rate = B/t(B).</span></div></foreignObject>'''+
    takeaway('Warm up, synchronize the timing boundaries, and repeat each measurement.'),
    'The lab is a small causal Transformer with random weights. It studies execution behavior, not language quality or production model throughput. It preallocates KV storage and avoids copying a growing cache with torch.cat. The benchmark records synchronized wall-clock time including host dispatch, GPU work, KV writes, and greedy token selection. Prefill includes the first prediction. Decode rates refer to later tokens only. Reported trial percentiles describe repeated trial-average step durations, not serving tail latency. Hardware, software, model shape, precision and lengths are recorded in metadata. The implementation stays fixed, but automatic SDPA backend selection can change with tensor shapes; profile the selected kernels if explaining a performance change. Use the lab plotting script on the saved CSV; the presentation does not invent a performance curve. No local GPU measurements are bundled. The DistServe figure elsewhere in this deck is a separately attributed published measurement, not a run of this lab.',('kernels','timing','berkeley_bench'))

# 17
add('A measurement should test a bottleneck hypothesis',1,
    table(75,191,1450,['Observation','Possible cause','A controlled experiment'],[
        ['Gaps between short GPU kernels','Host / launch overhead','Compare ordinary execution and graph replay'],
        ['Small batches have low throughput','Little weight reuse','Increase batch at fixed sequence lengths'],
        ['Decode slows with longer context','More attention/KV work','Vary context, profile attention time'],
        ['Prefill creates large intermediates','Attention IO overhead','Compare explicit attention and fused SDPA'],
    ],[.34,.27,.39],row_h=91,size=25)+
    lines(75,696,['A busy GPU timeline alone cannot distinguish compute limits from bandwidth limits.'],28,color=MUTED)+
    takeaway('Profile, predict a change, and check the end-to-end result.'),
    'Each row is a hypothesis, not a diagnosis from one symptom. A profiler timeline helps reveal launch gaps and which operators consume time. Both bandwidth-bound and compute-bound kernels can keep a GPU continuously busy. Use operator shapes, FLOP/byte reasoning, achieved compute or memory metrics where available, and controlled comparisons to distinguish limits. PyTorch SDPA is a dispatching API and may select different kernels, so inspect its actual backend rather than assuming FlashAttention. Keep numerical behavior and workload comparable. A lower kernel time does not guarantee lower application latency if another cost dominates.',('kernels','berkeley_bench'))

# 18
add('Discussion',.5,
    lines(75,224,['A model fits on the GPU, but generation is slow.', 'What would you measure first?'],39,weight=700,gap=60)+
    line(75,371,1525,371)+
    text(75,430,'Change the workload',31,700,color=BLUE)+text(580,430,'One short request versus many long requests',29)+
    line(75,470,1525,470)+text(75,529,'Change the objective',31,700,color=BLUE)+text(580,529,'Interactive response versus offline throughput',29)+
    line(75,569,1525,569)+text(75,628,'Choose the next experiment',31,700,color=BLUE)+text(580,628,'Which result would change your explanation?',29)+
    text(75,716,'Next: how the same costs change when a model spans multiple GPUs.',29,color=MUTED)+
    text(75,793,'Reading: CS336 Lectures 5, 6, 10; Berkeley Spring 2026 Lecture 18; cited original papers',25,color=MUTED),
    'Discussion guidance is filled after ordering and timing are resolved.',('inference','flash','berkeley'))

def mathline(x, y, w, h, markup, size=37):
    """Native MathML keeps fractions and subscripts readable without a CDN."""
    return f'<foreignObject x="{x}" y="{y}" width="{w}" height="{h}"><div xmlns="http://www.w3.org/1999/xhtml" class="math-block" style="font-size:{size}px"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow>{markup}</mrow></math></div></foreignObject>'


def codeblock(x, y, w, h, code, size=27):
    return f'<foreignObject x="{x}" y="{y}" width="{w}" height="{h}"><pre xmlns="http://www.w3.org/1999/xhtml" class="code-block" style="font-size:{size}px">{esc(code)}</pre></foreignObject>'


# Revised teaching sequence: metrics, workload, memory, kernels, serving.
add('Latency and throughput measure different things',1,
    text(75,195,'One request arrives. Its output tokens appear over time.',31)+
    arrow(140,365,1450,365)+text(1450,411,'Time',25,anchor='end',color=MUTED)+
    ''.join(line(x,345,x,385,BLUE,3)+text(x,326,label,27,700,anchor='middle') for x,label in [(160,'Arrival'),(810,'Token 1'),(1080,'Token 2'),(1350,'Token 3')])+
    line(160,439,810,439,BLUE,3)+text(485,483,'Time to first token (TTFT)',29,700,color=BLUE,anchor='middle')+
    line(810,540,1080,540,TEAL,3)+text(945,583,'Inter-token latency (ITL)',29,700,color=TEAL,anchor='middle')+
    text(75,672,'Aggregate throughput counts output tokens from all requests per second.',29)+
    takeaway('A fast first token, a smooth stream, and high total throughput are distinct goals.',
              'TTFT includes queueing and processing. This timeline is schematic, not a measurement.'),
    'Start with user-visible behavior before optimization names. Arrival is the chosen service boundary. TTFT runs to the first delivered output token and includes queueing, tokenization, prefill, sampling and delivery overhead. ITL is an interval between successive delivered tokens; a per-request average of these intervals is often called TPOT. Aggregate output throughput counts tokens across requests in a time window. The drawn distances have no numeric time scale. Rates in the fixed-batch lab exclude the first output token and are not production latency percentiles.',('inference','berkeley'))

add('KV memory limits how many requests fit',1,
    mathline(75,178,1450,100,'<mtext>KV bytes</mtext><mo>=</mo><mn>2</mn><mo>×</mo><mi>B</mi><mo>×</mo><mi>S</mi><mo>×</mo><mi>L</mi><mo>×</mo><msub><mi>d</mi><mtext>KV</mtext></msub><mo>×</mo><mi>p</mi>',43)+
    text(75,317,'B: batch size',29)+text(865,317,'L: number of layers',29)+
    text(75,364,'S: cached sequence length (tokens)',29)+text(865,364,'p: bytes per element',29)+
    mathline(75,391,1030,53,'<msub><mi>d</mi><mtext>KV</mtext></msub><mo>=</mo><mtext>number of KV heads</mtext><mo>×</mo><mtext>head dimension</mtext>',29)+
    text(865,429,'2 accounts for K and V.',27)+
    line(75,469,1525,469)+
    text(75,519,'Example',31,700,color=BLUE)+
    text(75,565,'Batch size B = 16; cached sequence length S = 4,096; L = 32 layers;',29)+
    mathline(75,585,1450,55,'<msub><mi>d</mi><mtext>KV</mtext></msub><mo>=</mo><mn>4,096</mn><mspace width="0.25em"/><mtext>(32 KV heads × 128); BF16,</mtext><mspace width="0.25em"/><mi>p</mi><mo>=</mo><mn>2</mn><mspace width="0.25em"/><mtext>bytes.</mtext>',29)+
    mathline(75,658,1450,66,'<mtext>KV size</mtext><mo>=</mo><mn>2</mn><mo>×</mo><mn>16</mn><mo>×</mo><mn>4,096</mn><mo>×</mo><mn>32</mn><mo>×</mo><mn>4,096</mn><mo>×</mo><mn>2</mn><mspace width="0.25em"/><mtext>bytes</mtext><mo>≈</mo><mn>34.4</mn><mspace width="0.25em"/><mtext>GB</mtext>',35)+
    takeaway('Weights and temporary buffers need additional memory.',
              'Assume equal sequence lengths and K/V widths, with no prefix sharing. GB is decimal.'),
    'The formula counts both K and V for a batch of B sequences, each with S cached tokens, across L layers. d_KV is the number of KV heads times the head dimension: the total K dimension for one token in one layer, and equally the total V dimension. It need not equal the full model hidden dimension. p is the storage size in bytes per element. In the example B=16, S=4096, L=32, d_KV=32×128=4096, and p=2 for BF16. Direct substitution gives 2×16×4096×32×4096×2=34359738368 bytes, or approximately 34.4 GB. The first factor 2 counts K and V; the final factor 2 is BF16 storage per element. S counts cached prompt and generated positions already processed, not the maximum permitted sequence length or only the newly generated tokens. The newest sampled token enters the cache when it is subsequently processed. For unequal lengths, replace B×S with the total cached tokens across the batch. The slide assumes uniform layer widths, equal K and V dimensions, and no prefix sharing between requests. This is stored KV payload, not per-step memory traffic or total GPU allocation; unused cache capacity, metadata, model weights and workspace require additional memory.',('inference','berkeley','kv_size'))

def batch_schedule():
    out=''
    # Each column is an iteration boundary, not an equal-duration interval.
    for start,title,rows in [(75,'Static membership',[['A','A','','',''],['B','B','B','B','B']]),
                             (835,'Continuous membership',[['A','A','C','C','C'],['B','B','B','B','B']])]:
        out+=text(start,285,title,30,700,color=BLUE)
        for i in range(5): out+=text(start+146+i*112,337,str(i+1),25,anchor='middle',color=MUTED)
        for r,row in enumerate(rows):
            out+=text(start,399+r*88,f'Slot {r+1}',26)
            for i,value in enumerate(row):
                fill=PALE if value=='A' else '#e3eeea' if value=='C' else '#f0f1f2' if value else 'white'
                out+=label_box(start+94+i*112,356+r*88,100,64,value or 'idle',fill,size=26)
        out+=text(start+93,564,'Decode iterations',26,color=MUTED)
    return out

add('Continuous batching replaces finished requests',1,
    lines(75,191,['A and B are decoding. A finishes after iteration 2.', 'Request C is ready to decode and waiting for a free slot.'],29,gap=42)+
    batch_schedule()+line(780,260,780,592)+
    text(75,637,'Static batch: wait for B before admitting C.',28)+
    text(835,637,'Continuous batch: admit C at iteration 3.',28)+
    takeaway('The scheduler revisits batch membership at each iteration boundary.',
              'C’s prefill is omitted here. Admission also needs KV space and a token budget.'),
    'The drawing isolates iteration-level membership changes. Each column is one iteration, not a fixed number of milliseconds, so it does not claim a numeric speedup. A needs two more decode tokens, B five, and C three. Under the illustrated static policy, the completed A slot cannot be reused until B finishes. Under continuous membership C takes the slot at the next boundary. Assume C has completed prefill elsewhere/earlier; real scheduling must also budget that work. A vLLM-style scheduler also checks available KV blocks and the step token budget. Different context lengths require appropriate ragged/paged attention handling. Fixed slot count two is purely illustrative.',('inference','orca','berkeley'))

add('Attention is a weighted average',1.25,
    text(75,195,'For one query q, score each valid key: sⱼ = q · kⱼ / √d.',31)+
    mathline(150,238,1300,166,'<mi>o</mi><mo>=</mo><munder><mo>∑</mo><mi>j</mi></munder><msub><mi>p</mi><mi>j</mi></msub><msub><mi>v</mi><mi>j</mi></msub><mo>=</mo><mfrac><mrow><munder><mo>∑</mo><mi>j</mi></munder><msup><mi>e</mi><msub><mi>s</mi><mi>j</mi></msub></msup><msub><mi>v</mi><mi>j</mi></msub></mrow><mrow><munder><mo>∑</mo><mi>j</mi></munder><msup><mi>e</mi><msub><mi>s</mi><mi>j</mi></msub></msup></mrow></mfrac>',46)+
    line(75,429,1525,429)+
    text(75,489,'Numerator',30,700,color=BLUE)+text(405,489,'Sum each value vector, weighted by exp(score).',30)+
    text(75,561,'Denominator',30,700,color=BLUE)+text(405,561,'Sum those same positive weights.',30)+
    text(75,655,'Both sums can accumulate one K/V tile at a time.',33,700)+
    takeaway('Keep the sums. Divide after the final tile.',
              'Each attention weight scales an entire value vector.'),
    'This is attention after Q/K/V projections, restricted to valid positions for a chosen query. The scale sqrt(d) is already included in each score. Substituting p_j=exp(s_j)/sum_k exp(s_k) into sum_j p_j v_j gives the ratio on screen. The denominator is a scalar; the numerator and output are value-dimension vectors. Both unnormalized sums are additive across disjoint tiles, so a tile does not need to know future scores before contributing. This establishes why streaming is mathematically possible. It does not yet address numerical overflow; the next slide introduces the stable scale. Do not independently normalize tiles and average their outputs, because different tiles have different total weight.',('flash','paper'))

add('Stable streaming needs three running values',1.5,
    text(75,194,'Directly evaluating exp(s) can overflow. Subtract a shared maximum m.',30)+
    mathline(120,218,1360,152,'<mi>o</mi><mo>=</mo><mfrac><mrow><munder><mo>∑</mo><mi>j</mi></munder><msup><mi>e</mi><mrow><msub><mi>s</mi><mi>j</mi></msub><mo>−</mo><mi>m</mi></mrow></msup><msub><mi>v</mi><mi>j</mi></msub></mrow><mrow><munder><mo>∑</mo><mi>j</mi></munder><msup><mi>e</mi><mrow><msub><mi>s</mi><mi>j</mi></msub><mo>−</mo><mi>m</mi></mrow></msup></mrow></mfrac><mo>=</mo><mfrac><mi>u</mi><mi>ℓ</mi></mfrac>',43)+
    text(75,395,'The common factor exp(−m) cancels between numerator and denominator.',28,color=MUTED)+
    table(75,438,1450,['State for positions seen so far','Meaning'],[
        ['m = max score seen','Reference scale for exponentials'],
        ['ℓ = Σ exp(sⱼ − m)','Total weight on that scale (scalar)'],
        ['u = Σ exp(sⱼ − m) vⱼ','Weighted value sum on that scale (vector)'],
    ],[.49,.51],row_h=70,size=28)+
    takeaway('A later tile may raise m. The old sums must then change scale.'),
    'The equations describe the state after processing at least one valid key. m is the maximum of the scores processed so far. On this reference scale every exp(s_j−m) is at most one, avoiding overflow of positive exponentials. Multiplying both sums by the same positive factor does not alter their ratio. The state uses two scalars and one vector per query row; it does not grow with the number of processed keys. A real query tile has one such state per row. Online means an incremental scan within a kernel, not online learning or a server processing internet requests. The remaining obstacle is that m is not known globally until all tiles are visited.',('normalizer','paper'))

add('A new maximum rescales the old contributions',2,
    text(75,193,'Suppose the next tile raises the maximum from m to m′.',30)+
    mathline(115,231,1370,112,'<msup><mi>e</mi><mrow><msub><mi>s</mi><mi>j</mi></msub><mo>−</mo><msup><mi>m</mi><mo>′</mo></msup></mrow></msup><mo>=</mo><msup><mi>e</mi><mrow><msub><mi>s</mi><mi>j</mi></msub><mo>−</mo><mi>m</mi></mrow></msup><mo>×</mo><munder><msup><mi>e</mi><mrow><mi>m</mi><mo>−</mo><msup><mi>m</mi><mo>′</mo></msup></mrow></msup><mi>α</mi></munder>',43)+
    text(75,368,'Every old weight changes by the same factor α, so rescale the two sums.',29)+
    line(75,404,1525,404)+
    text(75,452,'1. Common scale',28,700,color=BLUE)+text(445,452,'m′ = max(m, max scores in the new tile)',30)+
    text(75,508,'2. New weights',28,700,color=BLUE)+text(445,508,'wⱼ = exp(sⱼ − m′)  for keys in the new tile',30)+
    text(75,566,'3. Merge',28,700,color=BLUE)+text(445,566,'ℓ′ = αℓ + Σ wⱼ',33,700)+
    text(445,624,'u′ = αu + Σ wⱼ vⱼ',33,700)+
    text(75,701,'If the maximum stays the same, α = 1. After the last tile, output o = u / ℓ.',28)+
    takeaway('Rescale both old sums, then add the new tile on the same scale.'),
    'The identity at the top is the reason this works, not an extra heuristic. For every old score s, exp(s−m′)=exp(s−m) exp(m−m′). Linearity lets us multiply the old denominator and numerator by alpha without retaining old scores or values. The sums in the update range only over the new tile. If the maximum does not increase, alpha is one. Initialization is m=−infinity, l=0, u=0; the first tile with a finite valid maximum gives alpha=0. All-masked tiles require special handling to avoid exp(−infinity−(−infinity)), and are skipped in this introductory derivation. Each state should be updated together, using old l,u in the right-hand sides. The numerator is a vector in real attention.',('normalizer','paper'))

add('FlashAttention streams tiles through the running state',1.5,
    text(75,194,'For one query tile, repeat the same update for each valid K/V tile.',30)+
    text(75,275,'HBM',32,700,color=BLUE)+text(590,275,'On chip',32,700,color=BLUE)+
    ''.join(label_box(75,319+i*85,310,65,f'K/V tile {i+1}',size=29) for i in range(3))+
    arrow(392,433,528,433)+
    rect(555,310,970,285,'white',BLUE)+
    text(585,357,'Query tile Q stays available',30,700)+
    text(585,418,'Compute scores; choose m′ = max(m, tile maximum).',28)+
    text(585,479,'Rescale old ℓ and u; accumulate exp(scores − m′).',28)+
    text(585,551,'Running state persists while the next K/V tile arrives.',28,700,color=BLUE)+
    text(75,665,'After the final tile: normalize u / ℓ and store the output in HBM.',32,700)+
    takeaway('Tiling bounds working memory. Online softmax preserves the full-row result.',
              'Illustration of a query-tile loop. Different query tiles may reread K/V.'),
    'This reconnects the algebra to the GPU memory hierarchy. One query tile and its state are kept on chip, while K/V tiles are streamed from HBM. The score/probability tile is temporary; full S-by-S arrays are never required in HBM. The running state is per query row, and u is a vector. A real kernel partitions storage across registers and shared memory and uses many threads. Do not imply the whole sequence fits on chip, or that all K/V values are read exactly once for the whole attention layer. Loop orders and implementations differ; this is an explanatory query-tile organization consistent with FlashAttention-2-style processing. Causal masks restrict each row’s valid keys.',('flash','paper'))

add('Question 2: Is paged storage enough?',1.5,
    text(75,193,'A prototype stores K/V in pages, but reconstructs dense tensors every decode step.',29)+
    codeblock(75,239,1450,226,'# One head. table lists physical pages in logical token order.\n# q: [1, d]; each page: [block_size, d]; T: valid cached positions\nK = torch.cat([K_pool[p] for p in table], dim=0)[:T]\nV = torch.cat([V_pool[p] for p in table], dim=0)[:T]\nout = torch.softmax(q @ K.T / math.sqrt(d), dim=-1) @ V',26)+
    line(75,490,1525,490)+
    text(75,545,'1. Can this produce the correct attention output?',31,700)+
    text(75,606,'2. What extra traffic and storage does reconstructing K/V create?',31,700)+
    text(75,667,'3. What must change to attend directly to the paged cache?',31,700,color=BLUE)+
    text(75,803,'Assume a last-position causal query, correct positional encoding, and valid KV contents.',25,color=MUTED),
    'Authored diagnostic exercise grounded in the PagedAttention kernel design, not a verified interview question from any company. The code is executable for a one-head float tensor setting with the imports and inputs provided. T includes the current processed token. The current query may attend to all T keys, so no future-position mask is needed here. The block table must preserve token order and every view in the list has shape block_size×d. torch.cat allocates a dense tensor and copies page contents; slicing to T occurs after concatenation and does not undo that allocation. Ask participants to separate numerical correctness, persistent allocation, temporary allocations, and bandwidth. The question targets the O(Td) K/V reconstruction. This unfused reference also creates O(T) scores and probabilities; that is a separate cost. No timing or speedup is supplied because they depend on the implementation and workload.',('inference','paged'))

add('Solution 2: The kernel must consume the block table',1.5,
    text(75,199,'Correct result',31,700,color=BLUE)+
    lines(430,199,['Yes, if logical order, valid length, and attention semantics match.', 'The physical placement of the pages does not change the formula.'],28,gap=43)+
    line(75,294,1525,294)+
    text(75,349,'Extra cost',31,700,color=BLUE)+
    lines(430,349,['Each torch.cat reads pages and writes a dense K or V temporary.', 'Attention then reads those dense tensors. Pages and copies coexist.'],28,gap=43)+
    line(75,444,1525,444)+
    text(75,502,'Direct access',31,700,color=BLUE)+
    lines(430,502,['Pass the block table and valid lengths into a page-aware kernel.', 'Fetch the required K/V tiles and accumulate attention directly.'],28,gap=43)+
    text(75,660,'Measure peak live memory and copy traffic, as well as decode-step latency.',29,700)+
    takeaway('Paged storage and a compatible attention kernel work together.',
              'The kernel still reads the required historical K/V. Paging does not compress the cache.'),
    'The prototype can be mathematically correct. Page allocation already reduces some reservation waste, but every concatenate materializes the requested history again, adding O(Td) temporary storage and extra reads/writes per layer and step. Dense K and V can coexist with the page pool. A page-aware kernel translates token ranges using the block table and consumes page contents directly, retaining online softmax state as necessary. It does not need to recreate a whole contiguous K/V history in HBM. Appending new tokens updates the current tail/new page; old cache contents need not move. The benefit is not a promise of faster single-request attention. Compare the same model, dtype, lengths and batch first, then study admitted concurrency under a fixed memory budget. For numerical checking compare against the dense reference with an appropriate floating-point tolerance.',('inference','paged','vllm'))

def source_image(name, x, y, w, h):
    data=base64.b64encode((ASSETS/name).read_bytes()).decode()
    return f'<image href="data:image/png;base64,{data}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet"/>'

add('A prefill can delay an ongoing decode batch',1.5,
    text(75,194,'Published measurement: a 13B LLM, with input lengths 128 and 1,024.',29)+
    source_image('distserve-figure2.png',68,252,1010,476)+
    line(1090,244,1090,712)+
    text(1122,284,'How to read it',29,700,color=BLUE)+
    lines(1122,335,['x: batch size', 'y: batch execution time', '     in milliseconds'],25,gap=39)+
    text(1122,439,'Dashed: prefill alone',24,color=MUTED)+
    lines(1122,485,['Orange: decoding only', 'Blue: decoding plus', 'one prefill job'],25,gap=39)+
    lines(1122,619,['The gap shows added', 'delay for the batch.'],26,weight=700,color=BLUE)+
    takeaway('Long prompt work can interrupt smooth token delivery.',
              'Original Fig. 2 from DistServe (OSDI 2024). This is a published case, not our benchmark.'),
    'Figure reproduced from DistServe Figure 2, physical PDF p5, proceedings p196, with axes, legend and panel labels retained. The horizontal axis is batch size, and the vertical axis is batch execution time in milliseconds, not tokens/s and not end-to-end request latency. Blue adds one prefill job to a decoding batch; orange is decoding-only. Compare the two solid curves at the same batch size within one panel: their vertical gap is the added decoding delay. The dashed horizontal line is the prefill-only baseline; its distance to the blue curve measures prefill slowdown from sharing the batch. These are two different comparisons, not two names for the same gap. The larger gap in the 1024-token panel motivates protecting decoding from long prefill work. Do not compare absolute heights across panels without noticing their different y ranges. The figure caption specifies 13B but does not give a complete per-figure hardware/backend configuration, so this slide does not claim an H100 or a precise A100 setup. The plot demonstrates interference for the authors’ workload, not a universal speedup for PD separation or a comparison with modern chunked-prefill schedulers.',('distfig','berkeley_pd'))

add('Chunked prefill limits work between decode steps',1,
    text(75,193,'On one GPU, split a long prompt into smaller pieces that the scheduler can interleave.',29)+
    text(75,278,'One long prefill',29,700,color=BLUE)+
    label_box(75,309,220,75,'Decode',size=28)+label_box(315,309,700,75,'Prefill the whole prompt',fill='#e3eeea',size=29)+label_box(1035,309,220,75,'Decode',size=28)+
    line(75,427,1525,427)+text(75,488,'Chunked prefill',29,700,color=BLUE)+
    ''.join(label_box(75+i*238,522,218,75,label,PALE if i%2==0 else '#e3eeea',size=27) for i,label in enumerate(['Decode','Prefill 1','Decode','Prefill 2','Decode','Prefill 3']))+
    text(75,682,'Smaller chunks can reduce stalls, but may add overhead and slow prefill completion.',29)+
    takeaway('Chunking shares GPU time. PD separation uses different GPU resources.',
              'Schematic schedules, not measured durations. Chunk size must respect both latency goals.'),
    'The timeline illustrates bounded prompt work between decode opportunities. It is deliberately schematic: rows do not promise equal total time or a numeric speedup. A scheduler may also place prefill chunks and decode tokens in the same step, using a token budget. Splitting a prompt does not remove causal dependencies or the need to access previous K/V; it changes how much new prompt work is scheduled at once. Too-small chunks can increase scheduling/launch overhead and reduce GEMM efficiency. The next slide contrasts this colocated strategy with spatially separate prefill/decode pools. For deployment compare under a fixed GPU budget and input/output length distribution, rather than assuming separation is always necessary.',('berkeley_chunk','distserve'))

add('Question 3: Take-home — implement GEMM in Triton',0,
    text(75,194,'Implement C = A @ B for contiguous, row-major FP16 matrices on one GPU.',29)+
    label_box(75,232,1450,65,'A[M, K] × B[K, N] = C[M, N]',size=33)+
    text(75,352,'Implementation',31,700,color=BLUE)+
    lines(75,402,['Compute one output tile per program.', 'Loop over K tiles using tl.dot.', 'FP32 accumulation; FP16 output.', 'Handle M, N, and K boundaries safely.'],28,gap=48)+
    line(790,327,790,556)+
    text(830,352,'Validation and timing',31,700,color=BLUE)+
    lines(830,402,['Check torch.matmul with stated tolerances.', 'Measure steady-state GPU time.', 'Exclude compilation and autotuning.', 'Record hardware, shapes, and max error.'],28,gap=48)+
    line(75,574,1525,574)+
    text(75,614,'Test (M, N, K): (128, 256, 128), (127, 193, 259), and (1, 7, 3).',28)+
    text(75,662,'Deliver: kernel, launch wrapper, tests, and a short benchmark.',28)+
    text(75,708,'Optional: tune tile sizes and group programs for L2 reuse.',28)+
    takeaway('Explain one optimization using your measurements.',
              'Based on the Triton tutorial. Complete after the meeting; reference solution follows.'),
    'This is a take-home assignment adapted from the official Triton Matrix Multiplication tutorial, not an in-class coding task or a reported company interview question. Zero minutes means no additional planned lecture time; the assignment can be announced at the end. Implement a positive-dimension, contiguous row-major FP16 GEMM with an FP32 accumulator and FP16 output. tl.dot is permitted; calling torch.matmul or cuBLAS as the implementation is not. PyTorch is the validation baseline. Start with fixed tile sizes, map each program to one output tile, and iterate over K tiles. Handle invalid M/N input positions safely and zero-fill invalid K positions; mask output stores. The official reference wraps M/N load indices, so load masks are not the only valid approach. Test the three visible (M,N,K) cases, state rtol/atol, and report maximum error rather than requiring bitwise equality. Warm up and use CUDA events or a GPU benchmark utility, excluding compilation and autotuning from steady-state timing. Compare against torch.matmul under matching conditions. Grouped program ordering and autotuning are optional extensions. No requirement to outperform cuBLAS; the goal is a correct implementation and an explanation supported by measurements.',('triton_gemm',))

add('Solution 3: Triton GEMM tutorial',0,
    text(75,195,'Official worked solution, executable implementation, tests, and benchmarks.',30)+
    f'<a href="{SOURCES["triton_gemm"][1]}" target="_blank" rel="noopener">'+
    text(75,272,'Open the solution: Triton Matrix Multiplication',36,600,color=BLUE,attrs='text-decoration="underline"')+'</a>'+
    table(75,334,1450,['Tutorial section','What to compare with your implementation'],[
        ['Compute kernel / pointer arithmetic','Output tiles, addresses, K loop, and boundary handling'],
        ['L2 cache optimizations','Grouped program ordering to improve data reuse'],
        ['Final result','Launch configuration and autotuning'],
        ['Unit test / benchmark','Numerical correctness and measured performance'],
    ],[.38,.62],row_h=73,size=27)+
    takeaway('Compare the reference with your first implementation.',
              'Performance depends on the GPU, matrix shapes, and launch configuration.'),
    'The large underlined title and footer both link directly to the user-requested official tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html. Use it as the reference solution after attempting the preceding take-home exercise. The tutorial supplies a blocked FP16 GEMM with FP32 accumulation, pointer arithmetic, boundary handling, grouped program ordering, autotuning, correctness checks, and benchmark code. Its displayed speedups are examples for particular environments, not a target or promise for every GPU and shape. This reference slide is outside the planned lecture time. The slide itself works offline; opening the external solution requires internet access.',('triton_gemm',))

# Keep sources/notes attached to their content while reorganizing the narrative.
by_title={s['title']:s for s in slides}
ORDER=[
    'LLM inference performance',
    'Latency and throughput measure different things',
    'A prompt produces the first token',
    'New token rows are only part of the workload',
    'Many token rows reuse the same weights',
    'Arithmetic intensity connects reuse to the bottleneck',
    'H100 example: same weights, different bottlenecks',
    'The KV cache saves work on earlier tokens',
    'KV memory limits how many requests fit',
    'Question 1: Why did a larger decode batch stop helping?',
    'Solution 1: A throughput plateau does not identify the limit',
    'Batching affects weight traffic and KV traffic differently',
    'Continuous batching replaces finished requests',
    'PagedAttention reduces wasted KV capacity',
    'PagedAttention: logical blocks and physical storage',
    'Ordinary attention materializes large intermediates',
    'Attention is a weighted average',
    'Stable streaming needs three running values',
    'A new maximum rescales the old contributions',
    'FlashAttention streams tiles through the running state',
    'Question 2: Is paged storage enough?',
    'Solution 2: The kernel must consume the block table',
    'A prefill can delay an ongoing decode batch',
    'Chunked prefill limits work between decode steps',
    'Prefill–decode disaggregation',
    'Experiment: sweep batch size on one GPU',
    'A measurement should test a bottleneck hypothesis',
    'Discussion',
    'Question 3: Take-home — implement GEMM in Triton',
    'Solution 3: Triton GEMM tutorial',
]
assert set(ORDER)==set(by_title), set(by_title)^set(ORDER)
slides=[by_title[t] for t in ORDER]
# Existing timings are planning estimates; the detailed derivation is readable
# in the deck even when the presenter chooses a shorter route.
for i,t in {3:1.5,4:1,5:1,6:1.5,7:1.5,8:1,10:1,11:1,12:1,14:1.5,15:1.5,16:1,25:1.5,26:1,27:.5}.items():
    slides[i-1]['minutes']=t
TOTAL_MINUTES=sum(s['minutes'] for s in slides)
SHORT_SKIP=[6,7,26,27]
SHORT_MINUTES=TOTAL_MINUTES-sum(slides[n-1]['minutes'] for n in SHORT_SKIP)
slides[0]['body']=text(75,205,'09/17/26',30,color=MUTED)+lines(75,305,['Why does generation slow down,', 'and which part of the system should change?'],43,weight=700)+line(75,410,1525,410)+text(75,475,'Workload and limits',31,700,color=BLUE)+text(635,475,'Prefill, decode, weight reuse, and KV memory',28)+line(75,515,1525,515)+text(75,580,'Attention and storage',31,700,color=BLUE)+text(635,580,'Online softmax, FlashAttention, and paging',28)+line(75,620,1525,620)+text(75,683,'Serving behavior',31,700,color=BLUE)+text(635,683,'Batch membership, interference, and PD',28)+text(75,796,f'{len(slides)} slides: worked derivations, two diagnostic questions, and a GEMM take-home',27,color=MUTED)
slides[0]['notes']=f'Audience: Transformer/PyTorch familiarity with little GPU systems background. Follow the causal chain from latency metrics and matrix shapes to memory allocation, attention IO, and serving schedules. The complete deck has {TOTAL_MINUTES:g} minutes of suggested content. A roughly {SHORT_MINUTES:g}-minute route skips slides {", ".join(map(str,SHORT_SKIP))}. The online-softmax derivation remains visible in the deck. Reserve 15 minutes for discussion. Slides 29–30 are a take-home GEMM assignment and its reference solution, outside the lecture timing. Questions are authored exercises, not attributed company interview reports. The only empirical figure is clearly attributed to DistServe; other diagrams are schematic and H100 bars are theoretical resource bounds.'
by_title['Discussion']['notes']=f'Use the remaining discussion time to ask what observation could falsify a proposed bottleneck. The full route is {TOTAL_MINUTES:g} minutes; a roughly {SHORT_MINUTES:g}-minute route skips {", ".join(map(str,SHORT_SKIP))}. Week 3 develops multi-GPU parallelism and Week 4 studies scheduling, prefix reuse and serving policies in depth. Revisit the paged-prototype diagnostic if participants confuse memory allocation with the attention kernel.'

def build():
    assert len(slides)==30
    css=(ASSETS/'base.css').read_text()+'''.math-block{height:100%;display:flex;align-items:center;justify-content:flex-start;color:#172329}.math-block math{font-size:inherit}.code-block{margin:0;padding:18px 22px;line-height:1.4;background:#f4f6f8;border-left:3px solid #245675;font-family:ui-monospace,Menlo,Consolas,monospace;white-space:pre}@media print{.math-block,.code-block{break-inside:avoid}}'''+'''\nsvg{font-family:Arial,Helvetica,sans-serif}svg text{font-family:Arial,Helvetica,sans-serif}.interaction{font:25px Arial,Helvetica,sans-serif;display:flex;gap:16px;align-items:center;color:#245675}.interaction button,.interaction select{font:24px Arial,Helvetica,sans-serif;border:1px solid #aec3d1;color:#245675;background:white;padding:10px 17px;cursor:pointer}.interaction button:hover{background:#edf3f7}.interaction button:disabled{color:#88939a;cursor:default}.interaction.vertical{align-items:flex-start;flex-direction:column;gap:12px}.interaction a{font-size:23px}.interaction label{display:flex;gap:18px;align-items:center}button:focus-visible,select:focus-visible{outline:3px solid #397b71;outline-offset:3px}@media print{.interaction button,.interaction select{display:none}.interaction{font-size:23px}}\n'''
    parts=[]
    notes=['# Week 2 speaker notes', '', f'LLM inference performance. {len(slides)} slides, {TOTAL_MINUTES:g} minutes of suggested full content. A roughly {SHORT_MINUTES:g}-minute route skips slides {", ".join(map(str,SHORT_SKIP))}. Reserve 15 minutes for discussion.', '', 'The HTML deck works offline. The online-softmax derivation remains on the slides. The generation and block-table diagrams have step controls. The DistServe plot is an attributed published experiment; the H100 bars are theoretical bounds, and the remaining diagrams are schematic.', '']
    for i,s in enumerate(slides,1):
        footer=line(75,838,1525,838,'#bdc7cc')
        xx=75
        links=[]
        for key in s['sources']:
            label,url=SOURCES[key]
            footer+=f'<a href="{esc(url,quote=True)}" target="_blank" rel="noopener">'+text(xx,873,label,20,color=MUTED,attrs='text-decoration="underline"')+'</a>'
            # Approximate text advance at 20px Arial; widths are checked after render.
            xx+=len(label)*10+32
            links.append(f'<a href="{esc(url,quote=True)}" target="_blank" rel="noopener">{esc(label)}</a>')
        footer+=text(1525,873,f'{i} / {len(slides)}',20,color=MUTED,anchor='end')
        svg=f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" width="1600" height="900" role="img" aria-labelledby="title-{i}"><title id="title-{i}">{esc(s['title'])}</title><defs><marker id="arrow-{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE}"/></marker></defs>{rect(0,0,1600,900,'white','none')}{text(75,98,s['title'],46,700,color=BLUE)}{line(75,132,1525,132)}{s['body'].replace('url(#arrow)',f'url(#arrow-{i})')}{footer}</svg>'''
        aside=f'<aside class="speaker-notes"><p class="notes-meta">{s["minutes"]:g} min</p><p>{esc(s["notes"])}</p><p>{"<br>".join(links)}</p></aside>'
        parts.append(f'<section class="slide" id="slide-{i}" data-title="{esc(s["title"],quote=True)}" data-minutes="{s["minutes"]}" {"hidden" if i>1 else ""}>{svg}{aside}</section>')
        notes += [f'## {i}. {s["title"]}', '',f'**Suggested time: {s["minutes"]:g} minutes.**','',s['notes'],'']
        for key in s['sources']:
            label,url=SOURCES[key];notes+=[f'- [{label}]({url})']
        notes+=['']
    chrome='''<nav class="deck-chrome" aria-label="Presentation controls"><div class="chrome-left"><button id="prev" type="button" aria-label="Previous slide">←</button><span id="counter" aria-live="polite">1 / 30</span><button id="next" type="button" aria-label="Next slide">→</button><span id="slide-title"></span></div><div class="chrome-right"><button id="overview-toggle" type="button">Slides</button><button id="notes-toggle" type="button">Notes</button><button id="fullscreen" type="button">Full screen</button><button id="print" type="button">Print / PDF</button><button id="help-toggle" type="button" aria-label="Keyboard help">?</button></div></nav>
<section id="notes-panel" class="deck-overlay" hidden><div class="panel-header"><h2>Speaker notes</h2><button data-close-overlay="true" type="button">Close</button></div><div id="notes-content"></div></section>
<section id="overview-panel" class="deck-overlay" hidden><div class="panel-header"><h2>30 slides</h2><button data-close-overlay="true" type="button">Close</button></div><div id="overview-list"></div></section>
<section id="help-panel" class="deck-overlay" hidden><div class="panel-header"><h2>Presentation controls</h2><button data-close-overlay="true" type="button">Close</button></div><p>Arrow keys: change slides. Home / End: first / last slide. Escape: close a panel.</p><p>Use Next step inside a diagram to advance its example. Each solution follows its question. Notes includes assumptions and sources.</p><p>Print / PDF shows completed interactive examples. Online-softmax steps each have their own slide.</p></section>'''
    js=(ASSETS/'navigation.js').read_text()+'\n'+(ASSETS/'interactions.js').read_text()
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LLM inference performance · 09/17/26</title><style>'+css+'</style></head><body><main id="viewport" aria-label="Presentation"><div id="stage">'+''.join(parts)+'</div></main>'+chrome+'<script>'+js+'</script></body></html>'
    (ROOT/'week-2-inference.html').write_text(html)
    (ROOT/'week-2-speaker-notes.md').write_text('\n'.join(notes))
    print(f'Built {len(slides)} slides, {TOTAL_MINUTES:g} minutes; shorter route {SHORT_MINUTES:g} minutes.')

if __name__=='__main__': build()
