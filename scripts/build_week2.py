"""Build the self-contained Week 2 HTML presentation and companion notes.

Native SVG teaching diagrams retain the Week 1 design and support interactive
steps. No external fonts, images, scripts, or network calls are required.
"""
from pathlib import Path
from html import escape as esc

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'site/slide-assets/week2'
BLUE, INK, MUTED, PALE, LINE = '#245675', '#172329', '#58656d', '#edf3f7', '#aec3d1'
TEAL, WARM = '#397b71', '#a36330'
SOURCES = {
    'inference': ('CS336 Lecture 10', 'https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py'),
    'flash': ('CS336 Lecture 5, pp. 50–54', 'https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=50'),
    'kernels': ('CS336 Lecture 6', 'https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py'),
    'berkeley': ('Berkeley Lecture 18', 'https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_18.pdf#page=9'),
    'pareto': ('Berkeley Lecture 19, part 1, p. 10', 'https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_19_1.pdf#page=10'),
    'performance': ('Berkeley Lecture 2, pp. 44–45', 'https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture2.pdf#page=44'),
    'paper': ('FlashAttention paper', 'https://arxiv.org/abs/2205.14135'),
    'graphs': ('PyTorch CUDA Graphs', 'https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs'),
    'timing': ('PyTorch CUDA timing', 'https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution'),
    'matmul': ('NVIDIA matmul performance', 'https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-and-memory-bounds'),
    'h100': ('H100 SXM specifications', 'https://www.nvidia.com/en-sg/data-center/h100/'),
    'peak': ('NVIDIA dense compute peaks', 'https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput'),
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

# 1
add('Single-GPU LLM inference',.5,
    text(75,205,'09/17/26',30,color=MUTED)+
    lines(75,310,['The model fits in memory.', 'What determines how fast it generates tokens?'],43,weight=700)+
    line(75,410,1525,410)+
    text(75,475,'The workload',32,700,color=BLUE)+text(610,475,'Prefill, decode, and the KV cache',31)+
    line(75,513,1525,513)+text(75,574,'The bottlenecks',32,700,color=BLUE)+text(610,574,'Weight traffic, attention traffic, and launches',31)+
    line(75,612,1525,612)+text(75,673,'The experiment',32,700,color=BLUE)+text(610,673,'Batch size versus user speed and total throughput',31)+
    text(75,795,'30-minute presentation, followed by 15 minutes of discussion',27,color=MUTED),
    'Connect to Week 1: GPU compute can run ahead of its ability to move data. Today applies that idea to a causal dense Transformer on one GPU. The goal is to explain a bottleneck, predict a useful change, and measure it. The displayed date assumes the weekly meeting after September 10. The two questions are authored teaching exercises, not attributed interview reports.')

# 2
add('A prompt produces the first token',2,
    text(75,198,'Autoregressive generation: each new token depends on the preceding tokens',30)+
    '<g id="generation-scene"></g>'+control('generation')+
    takeaway('Prefill predicts token 1. Each decode step predicts one more token.'),
    'Use Next step twice. The four words are illustrative token labels, not a tokenizer demonstration. Prefill processes all prompt positions with a causal mask and returns logits at the final prompt position. Sampling those logits yields the first generated token. Decode step 1 feeds that generated token, adds its K/V at each layer, and predicts token 2. Decode step 2 feeds token 2 and predicts token 3. The newest sampled token enters the KV cache only when it is subsequently processed. Cache blocks represent per-token K/V across layers, not complete hidden activations. Times are schematic.',('inference','berkeley'))

# 3
add('Prefill and decode have different performance limits',1.5,
    text(75,195,'Inference includes both phases. Let B = requests and S = prompt tokens per request.',29)+
    table(75,244,1450,['','Prefill','One decode step'],[
        ['New positions per request','S prompt tokens together','1 new token'],
        ['Token rows in the dense layers','M = B × S','M = B'],
        ['Typical bottleneck','Compute-bound','Memory-bandwidth-bound'],
        ['Dominant cost in that regime','Performing arithmetic','Moving data from HBM'],
    ],[.32,.34,.34],row_h=84,size=27)+
    text(75,706,'Typical regime: a sufficiently long prompt, or decode with a small batch.',27,color=MUTED)+
    takeaway('The crucial difference is how much work each loaded weight supports.'),
    'Inference is the overall task, with both prefill and decode phases. Compute-bound means arithmetic throughput is the main rate limit; memory-bandwidth-bound means the rate of moving bytes is the main limit. It does not mean running out of memory. Flatten the request and new-token dimensions into M token rows for a dense projection or MLP. Prefill has M=B×S; ordinary decode has M=B because only one new token per request is ready. Long prompts often provide enough reuse to reach compute limits, whereas small decode batches often do not. Short prefills and large batched decode can behave differently, and operators within a pass can have different bottlenecks. The following three slides derive the claim rather than treating it as a rule to memorize.',('inference','berkeley'))

# More token rows reuse the same dense-layer weights.
add('Many token rows reuse the same weights',2,
    text(75,195,'One dense layer: Y = XW. Each row of X is a token processed in this pass.',30)+
    text(75,269,'Decode, one request',30,700,color=BLUE)+
    label_box(75,315,255,60,'1 token row',size=29)+text(369,355,'×',37)+
    label_box(430,282,365,145,'Weights W',size=31)+arrow(823,355,925,355)+label_box(955,315,255,60,'1 output row',size=29)+
    lines(1260,321,['Each weight does', 'little work before', 'the next load.'],26)+
    line(75,465,1525,465)+text(75,513,'Prefill, one long prompt',30,700,color=BLUE)+
    ''.join(label_box(75,543+j*37,255,33,'Token rows' if j==1 else '',size=24) for j in range(4))+
    text(369,620,'×',37)+label_box(430,543,365,145,'The same W',size=31)+arrow(823,620,925,620)+
    ''.join(label_box(955,543+j*37,255,33,'Output rows' if j==1 else '',size=24) for j in range(4))+
    lines(1260,573,['Reuse a weight tile', 'across many rows', 'while it is on chip.'],26)+
    takeaway('Prefill increases arithmetic much faster than it increases weight traffic.'),
    'The difference is matrix-vector versus matrix-matrix work, not different learned weights. With one decode request, a weight contributes to the output for one new token. With a prompt, the same weight contributes to many output rows. Tiled matrix multiplication exploits this reuse in on-chip storage. The entire weight matrix does not need to fit on chip, and real kernels can reload tiles. Only the ideal accounting on the next slides assumes one HBM read for each input. Causal masking restricts attention dependencies, but the tokenwise dense projections and MLP at each layer can still process all available prompt positions together.',('inference','matmul'))

add('Arithmetic intensity connects reuse to the bottleneck',2,
    text(75,194,'BF16 dense matmul: X[M, D] × W[D, F] = Y[M, F]',32,700)+
    table(75,240,1450,['Cost','Ideal accounting'],[
        ['Arithmetic','2MDF FLOPs'],
        ['HBM traffic: read X and W, write Y','2(MD + DF + MF) bytes'],
        ['Arithmetic intensity I','FLOPs / bytes = MDF / (MD + DF + MF)'],
    ],[.46,.54],row_h=78,size=28)+
    text(75,610,'If weight traffic dominates: I ≈ M FLOPs/byte',34,700,color=BLUE)+
    text(75,672,'Compare I with the GPU’s compute peak / HBM bandwidth.',30)+
    takeaway('Below that ratio: a bandwidth limit. Above it: a compute limit.',
              'This roofline model assumes overlap and excludes launch overhead and extra data movement.'),
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
add('Question 1: Why does a longer context slow decode?',.5,
    lines(75,217,['The same model runs on the same GPU.', 'Batch size is 1. KV caching is already enabled.'],34,gap=55)+
    text(75,390,'Short history',29,700)+''.join(rect(380+i*77,350,65,65,PALE) for i in range(4))+
    text(75,508,'Long history',29,700)+''.join(rect(380+i*77,468,65,65,PALE) for i in range(12))+
    lines(75,625,['Why might generating the next token take longer?', 'What did the cache save, and what work remains?'],34,weight=700,color=BLUE)+
    text(75,798,'Discuss the data movement. No arithmetic is needed.',27,color=MUTED),
    'Pause before advancing. Invite a distinction between constructing the historical K/V and reading it for a new query. Hold model, hardware, precision, and batch size fixed. We ask why latency might increase, not for an exact proportionality. The rectangles represent historical positions. This is an authored discussion question.',('inference',))

# 6
add('Solution 1: The cache still has to be read',1,
    text(75,205,'For each layer of full-context attention',31,700)+
    label_box(75,255,250,90,'Current query q')+arrow(325,300,420,300)+
    label_box(420,255,440,90,'Compare q with cached keys')+arrow(860,300,965,300)+
    label_box(965,255,530,90,'Weight and combine cached values')+
    lines(75,425,['Longer history means more K/V values to read.', 'It also means more query–key and weighted-value arithmetic.'],33,gap=58)+
    line(75,545,1525,545)+
    text(75,602,'Saved work',29,700,color=BLUE)+text(425,602,'Recomputing earlier tokens through the network',29)+
    text(75,661,'Remaining work',29,700,color=BLUE)+text(425,661,'Applying the new query to the relevant history',29)+
    takeaway('Longer context can slow attention even when the cache works perfectly.',
              'Whole-model latency also includes weights, other operators, and launch overhead.'),
    'The new query changes at each decode step. Its dot products with the historical keys and its weighted sum of historical values must therefore be computed again. KV caching eliminates rebuilding those historical states, not using them. Under ordinary full-context attention, K/V traffic and attention arithmetic grow with context length. Whole-model latency need not double when context doubles, because other costs remain. Sliding-window, sparse, and compressed attention can change this pattern and are outside this example.',('inference',))

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

# 8
add('Total throughput and per-user speed',1.5,
    text(75,191,'A larger batch can improve aggregate throughput while slowing each user.',30)+
    '<g id="batch-chart"></g>'+
    f'''<foreignObject x="1040" y="615" width="465" height="115"><div xmlns="http://www.w3.org/1999/xhtml" class="interaction vertical" data-deck-ignore-keys="true"><label for="batch-select">Batch size <select id="batch-select"></select></label><span id="chart-data-source">Illustrative model, not measurements</span></div></foreignObject>'''+
    takeaway('Choose throughput subject to the user’s latency requirement.'),
    'The horizontal axis is generation tokens per second per request, so moving right means a faster user experience. The vertical axis counts output tokens per second across all active requests. Each point is one fixed synchronous batch. The default chart is an explicitly illustrative model with decode-step duration t = 4 + 0.5B milliseconds, where B is batch size. Rates are 1000/t per user and 1000B/t in aggregate. This is not a hardware prediction. Import the accompanying benchmark CSV on slide 19 to replace this chart with local measurements. The Berkeley source inspired the axes; its disaggregated-serving curves are not reused as single-GPU measurements.',('pareto','inference'))

# 9
add('Question 2: Which batch meets the user’s target?',.5,
    text(75,201,'A user needs at least 40 generated tokens per second.',33,700)+
    text(75,251,'Choose the highest total throughput that meets that target.',30)+
    table(75,307,1450,['Batch size','Tokens/s per user','Total output tokens/s'],[
        ['1','222','222'],['8','125','1,000'],['32','50','1,600'],['64','28','1,778'],
    ],[.24,.38,.38],row_h=70,size=29)+
    text(75,708,'Why is the batch with the highest total throughput not always the best choice?',30,700,color=BLUE)+
    text(75,798,'Hypothetical example, rounded values. Fixed synchronous batches, no request queue.',25,color=MUTED),
    'This authored question requires comparison, not a lengthy calculation. All values are hypothetical and consistent with the illustrative chart on slide 11 after rounding. Batch 64 maximizes throughput among these rows but only delivers about 28 tokens/s per request. Ask which workload would accept that tradeoff. Offline bulk processing can prioritize aggregate throughput, while interactive use needs an explicit per-request target. These average rates are not p95 serving guarantees.',('pareto',))

# 10
add('Solution 2: Batch 32 meets both objectives',1,
    label_box(75,205,390,105,'Batch 32',size=37)+
    text(540,248,'50 tokens/s per user',35,700,color=BLUE)+text(540,299,'1,600 total output tokens/s',32)+
    line(75,360,1525,360)+
    text(75,422,'Batch 64',31,700)+lines(435,421,['More total throughput, but only 28 tokens/s per user.', 'It misses the 40 tokens/s target.'],30)+
    line(75,527,1525,527)+
    text(75,590,'For an actual deployment',31,700)+
    lines(75,648,['Sweep the batch size at the intended context length.', 'Check memory capacity and latency variation before choosing.'],29)+
    takeaway('“Faster” needs a metric and a workload.'),
    'Among the offered configurations, batch 32 delivers the highest aggregate throughput while meeting 40 tokens/s per user. A batch of 64 would be a reasonable choice if maximizing offline throughput were the objective and memory capacity allowed it. Real serving has variable arrivals and lengths, scheduling, queueing, and tail latency. Those complications belong to Week 4. The demonstration on slide 19 will measure repeated fixed-batch decode loops rather than production per-request percentiles.',('pareto','berkeley'))

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
add('FlashAttention keeps one tile near compute',2.5,
    text(75,197,'Last query of a 4-token prompt: process two keys at a time.',30)+
    '<g id="flash-scene"></g>'+control('flash')+
    takeaway('Accumulate the output as tiles arrive. Write the final output once.'),
    'Use Next step through four states: load the first K/V tile, compute its scores and partial state, load the second K/V tile, and normalize/write the result. This diagram follows one query row with four keys and scalar values. A real kernel handles query blocks and vector-valued outputs. It retains query data and running state on chip while looping over K/V tiles. It discards each score/probability tile after using it. Other query blocks repeat the procedure and may reread K/V. “Write once” refers to this conceptual output block, not a claim about every implementation or every memory transaction. The next slide shows the rescaling that makes softmax correct.',('flash','paper'))

# 13
add('Online softmax combines tiles correctly',3,
    text(75,194,'One query: scores = [ln 1, ln 2, ln 3, ln 4], values = [10, 20, 30, 40]',29)+
    text(75,244,'The full softmax weights are [1, 2, 3, 4] / 10, so the output is 30.',30,700)+
    '<g id="softmax-scene"></g>'+control('softmax')+
    takeaway('Track a running maximum, a normalization sum, and a weighted sum.'),
    'The toy log scores are chosen so exponentials are simple. First tile: m=ln2, exp(scores-m)=[0.5,1], l=1.5, u=25. Second tile raises the maximum to ln4, so old contributions must be multiplied by exp(ln2-ln4)=0.5. New weights are [0.75,1]. Updated l=0.5×1.5+0.75+1=2.5. Updated u=0.5×25+0.75×30+1×40=75. Output u/l=30. In general m′=max(m,max tile), α=exp(m−m′), p=exp(tile−m′), l′=αl+sum(p), u′=αu+sum(pv). The real u is a vector. Do not average independently normalized tile outputs equally: their normalization masses differ. Masked keys contribute zero weight.',('flash','paper'))

# 14
add('What FlashAttention changes',.5,
    table(75,191,1450,['Quantity','Effect'],[
        ['Full score/probability tensors in HBM','Avoided'],
        ['Working data on chip','One set of tiles and running state'],
        ['Dense attention arithmetic','Still quadratic in prompt length'],
        ['Attention definition','Same exact attention, up to floating-point effects'],
        ['Historical KV during decode','Still needed for full-context attention'],
    ],[.47,.53],row_h=77,size=27)+
    text(75,693,'Its main motivation here is reducing attention IO during prefill and training.',28,color=MUTED)+
    takeaway('Less memory traffic can make the same mathematical operation faster.'),
    'FlashAttention is an IO-aware algorithm for exact dense attention, not sparse attention or an approximation that drops keys. Different execution and accumulation orders can change floating-point rounding. It retains quadratic attention arithmetic but avoids full quadratic score/probability materialization. Training implementations also recompute intermediates in backward. The largest motivating savings here are for prefill and training. Decode has only one new query per request and continues streaming relevant KV; optimized decode kernels have additional design concerns. Do not promise constant traffic, a universal speedup, or that every input value is fetched exactly once.',('performance','paper'))

# 15
add('CUDA Graphs reduce repeated launch overhead',1.5,
    text(75,197,'Repeated execution of the same GPU work',31,700)+
    arrow(215,222,1195,222)+text(1260,232,'Time',25,color=MUTED)+
    text(75,285,'CPU',28,700)+text(75,363,'GPU',28,700)+
    ''.join(label_box(215+i*200,245,148,54,'Launch',size=24)+label_box(252+i*200,326,95,54,f'K{i+1}',size=24) for i in range(5))+
    text(1260,278,'Ordinary',28,700)+text(1260,319,'submission',28,700)+
    line(75,427,1525,427)+
    text(75,505,'CPU',28,700)+text(75,583,'GPU',28,700)+label_box(215,465,215,54,'Replay graph',size=24)+
    ''.join(label_box(255+i*110,546,95,54,f'K{i+1}',size=24) for i in range(5))+
    text(1260,498,'Captured',28,700)+text(1260,539,'execution',28,700)+
    text(75,668,'Fusion combines operations. Graph replay submits a captured sequence of operations.',28)+
    text(75,712,'Schematic timeline. Capture needs compatible shapes, control flow, and memory addresses.',25,color=MUTED)+
    takeaway('Fewer CPU submissions can help when the GPU waits between short kernels.'),
    'These timelines are schematic, not profiler measurements and not to scale. A CUDA Graph captures GPU operations and dependencies, then replays them with lower repeated host submission cost. The kernels can remain separate, which distinguishes graphs from operator fusion. Standard PyTorch capture needs stable memory addresses, capture-compatible operations, and static shapes/control flow for that captured graph. Input values can change in existing storage. Prefill lengths and growing decode history require deliberate shape/buffer handling or multiple graphs. Capture/compilation warm-up costs are separate from replay latency. Replay cannot eliminate the underlying arithmetic or required HBM traffic.',('graphs','kernels'))

# 16
add('Experiment: sweep batch size on one GPU',1.5,
    text(75,198,'A runnable CUDA notebook accompanies this presentation.',31,700)+
    table(75,243,1450,['Keep fixed','Measure'],[
        ['Model configuration and random weights','Prefill time'],
        ['Prompt length and output length','Average decode-step time'],
        ['GPU, precision, and implementation','Tokens/s per user and across the batch'],
        ['Timing boundaries and warm-up','Peak allocated memory in GB'],
    ],[.5,.5],row_h=72,size=27)+
    f'''<foreignObject x="75" y="630" width="1450" height="102"><div xmlns="http://www.w3.org/1999/xhtml" class="interaction vertical" data-deck-ignore-keys="true"><div><button type="button" id="import-button">Import benchmark CSV</button><button type="button" id="reset-data">Use illustrative chart</button><a href="https://github.com/zhiweixx/llm-systems-study-group/tree/main/week-2-lab" target="_blank" rel="noopener">Notebook and benchmark</a><input id="csv-file" type="file" accept=".csv,text/csv" hidden="hidden"/></div><span id="import-status" role="status">No GPU measurements loaded. The batch-size chart currently shows an illustrative model.</span></div></foreignObject>'''+
    takeaway('Warm up, synchronize the timing boundaries, and repeat each measurement.'),
    'The lab is a small causal Transformer with random weights. It studies execution behavior, not language quality or production model throughput. It preallocates KV storage and avoids copying a growing cache with torch.cat. The benchmark records synchronized wall-clock time including host dispatch, GPU work, KV writes, and greedy token selection. Prefill includes the first prediction. Decode rates refer to later tokens only. Reported trial percentiles describe repeated trial-average step durations, not serving tail latency. Hardware, software, model shape, precision and lengths are recorded in metadata. The implementation stays fixed, but automatic SDPA backend selection can change with tensor shapes; profile the selected kernels if explaining a performance change. Import the generated CSV to replace slide 11’s illustrative chart for this browser session; no file upload or network request occurs. No measured GPU results are bundled.',('kernels','timing','berkeley'))

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
    'Each row is a hypothesis, not a diagnosis from one symptom. A profiler timeline helps reveal launch gaps and which operators consume time. Both bandwidth-bound and compute-bound kernels can keep a GPU continuously busy. Use operator shapes, FLOP/byte reasoning, achieved compute or memory metrics where available, and controlled comparisons to distinguish limits. PyTorch SDPA is a dispatching API and may select different kernels, so inspect its actual backend rather than assuming FlashAttention. Keep numerical behavior and workload comparable. A lower kernel time does not guarantee lower application latency if another cost dominates.',('kernels','berkeley'))

# 18
add('Discussion',.5,
    lines(75,224,['A model fits on the GPU, but generation is slow.', 'What would you measure first?'],39,weight=700,gap=60)+
    line(75,371,1525,371)+
    text(75,430,'Change the workload',31,700,color=BLUE)+text(580,430,'One short request versus many long requests',29)+
    line(75,470,1525,470)+text(75,529,'Change the objective',31,700,color=BLUE)+text(580,529,'Interactive response versus offline throughput',29)+
    line(75,569,1525,569)+text(75,628,'Choose the next experiment',31,700,color=BLUE)+text(580,628,'Which result would change your explanation?',29)+
    text(75,716,'Next: how the same costs change when a model spans multiple GPUs.',29,color=MUTED)+
    text(75,793,'Reading: CS336 Lectures 5, 6, 10 and Berkeley Scalable AI Lectures 2, 18, 19',25,color=MUTED),
    'The planned talk totals 30 minutes including short question pauses. Use the following 15 minutes for discussion, with five minutes of meeting buffer. Ask participants to choose a metric and workload before suggesting an optimization. Revisit the two questions if time is short. Week 3 covers multi-GPU execution and sharding. Week 4 adds arrivals, continuous batching, paged KV management, prefix reuse, and serving latency objectives. Sources below also appear on the relevant slides.',('inference','flash','berkeley'))

def build():
    assert len(slides)==21
    assert sum(s['minutes'] for s in slides)==30
    css=(ASSETS/'base.css').read_text()+'''\nsvg{font-family:Arial,Helvetica,sans-serif}svg text{font-family:Arial,Helvetica,sans-serif}.interaction{font:25px Arial,Helvetica,sans-serif;display:flex;gap:16px;align-items:center;color:#245675}.interaction button,.interaction select{font:24px Arial,Helvetica,sans-serif;border:1px solid #aec3d1;color:#245675;background:white;padding:10px 17px;cursor:pointer}.interaction button:hover{background:#edf3f7}.interaction button:disabled{color:#88939a;cursor:default}.interaction.vertical{align-items:flex-start;flex-direction:column;gap:12px}.interaction a{font-size:23px}.interaction label{display:flex;gap:18px;align-items:center}#import-status{font-size:23px;max-width:1430px}#chart-data-source{font-size:22px;line-height:1.3}button:focus-visible,select:focus-visible{outline:3px solid #397b71;outline-offset:3px}@media print{.interaction button,.interaction select,#import-button,#csv-file,#reset-data{display:none}.interaction{font-size:23px}}\n'''
    parts=[]
    notes=['# Week 2 speaker notes','', 'Single-GPU LLM inference. 21 slides, 30 minutes, followed by 15 minutes of discussion.', '', 'The deck and its interactive diagrams work offline. Use arrow keys to change slides, the in-slide buttons to advance examples, and Notes for the explanation. Print shows the completed interactive examples. The batch-size chart starts with an explicitly illustrative model. The optional lab produces real measurements on a CUDA GPU.', '']
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
        footer+=text(1525,873,f'{i} / 21',20,color=MUTED,anchor='end')
        svg=f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" width="1600" height="900" role="img" aria-labelledby="title-{i}"><title id="title-{i}">{esc(s['title'])}</title><defs><marker id="arrow-{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE}"/></marker></defs>{rect(0,0,1600,900,'white','none')}{text(75,98,s['title'],46,700,color=BLUE)}{line(75,132,1525,132)}{s['body'].replace('url(#arrow)',f'url(#arrow-{i})')}{footer}</svg>'''
        aside=f'<aside class="speaker-notes"><p class="notes-meta">{s["minutes"]:g} min</p><p>{esc(s["notes"])}</p><p>{"<br>".join(links)}</p></aside>'
        parts.append(f'<section class="slide" id="slide-{i}" data-title="{esc(s["title"],quote=True)}" data-minutes="{s["minutes"]}" {"hidden" if i>1 else ""}>{svg}{aside}</section>')
        notes += [f'## {i}. {s["title"]}', '',f'**Suggested time: {s["minutes"]:g} minutes.**','',s['notes'],'']
        for key in s['sources']:
            label,url=SOURCES[key];notes+=[f'- [{label}]({url})']
        notes+=['']
    chrome='''<nav class="deck-chrome" aria-label="Presentation controls"><div class="chrome-left"><button id="prev" type="button" aria-label="Previous slide">←</button><span id="counter" aria-live="polite">1 / 21</span><button id="next" type="button" aria-label="Next slide">→</button><span id="slide-title"></span></div><div class="chrome-right"><button id="overview-toggle" type="button">Slides</button><button id="notes-toggle" type="button">Notes</button><button id="fullscreen" type="button">Full screen</button><button id="print" type="button">Print / PDF</button><button id="help-toggle" type="button" aria-label="Keyboard help">?</button></div></nav>
<section id="notes-panel" class="deck-overlay" hidden><div class="panel-header"><h2>Speaker notes</h2><button data-close-overlay="true" type="button">Close</button></div><div id="notes-content"></div></section>
<section id="overview-panel" class="deck-overlay" hidden><div class="panel-header"><h2>21 slides · 30 minutes</h2><button data-close-overlay="true" type="button">Close</button></div><div id="overview-list"></div></section>
<section id="help-panel" class="deck-overlay" hidden><div class="panel-header"><h2>Presentation controls</h2><button data-close-overlay="true" type="button">Close</button></div><p>Arrow keys: change slides. Home / End: first / last slide. Escape: close a panel.</p><p>Use Next step inside a diagram to advance its example. Each solution follows its question. Notes includes assumptions and sources.</p><p>CSV imports remain in this browser session. Print / PDF shows completed examples.</p></section>'''
    js=(ASSETS/'navigation.js').read_text()+'\n'+(ASSETS/'interactions.js').read_text()
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Single-GPU LLM inference · 09/17/26</title><style>'+css+'</style></head><body><main id="viewport" aria-label="Presentation"><div id="stage">'+''.join(parts)+'</div></main>'+chrome+'<script>'+js+'</script></body></html>'
    (ROOT/'week-2-inference.html').write_text(html)
    (ROOT/'week-2-speaker-notes.md').write_text('\n'.join(notes))
    print('Built 21 slides, 30 minutes, and speaker notes.')

if __name__=='__main__': build()
