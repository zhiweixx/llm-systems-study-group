"""Context-parallel teaching sequence: original editable diagrams, primary sources."""
from .common import *

CP = 'https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html'
RING = 'https://arxiv.org/abs/2310.01889'
ULYSSES = 'https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md'
JANG = 'https://insujang.github.io/2024-09-20/introducing-context-parallelism/'
STRIPED = 'https://arxiv.org/abs/2311.09431'
LAYOUT = 'https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.context_parallel.layout.html'
DCP = 'https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/'
DCP_BLOG = 'https://vllm.ai/blog/2026-08-07-decode-context-parallelism'
FLASH = 'https://arxiv.org/abs/2205.14135'


def tokens(x, y, values, w=64, h=51, active=None):
    out = ''
    for i, v in enumerate(values):
        selected = active is None or v in active
        out += rect(x+i*w, y, w, h, PALE if selected else 'white')
        out += text(x+(i+.5)*w, y+34, v, 27, 600 if selected else 400,
                    color=BLUE if selected else MUTED, anchor='middle')
    return out


def get_slides():
    slides = []
    body = lines(75, 207, ['A single long request can exceed one GPU’s activation or KV budget.',
                          'Split positions of that same sequence across a context-parallel group.'], 29, 43)
    body += line(790, 295, 790, 675)
    body += text(75, 314, 'Data parallelism: different sequences', 30, 700)
    body += text(840, 314, 'Context parallelism: one sequence', 30, 700)
    for i, y in enumerate([350, 498]):
        body += rect(75, y, 660, 124) + text(95, y+37, f'GPU {i}  ·  complete model', 27, 600)
        body += label_box(95, y+55, 620, 48, f'Request {"A" if i == 0 else "B"}: every position', size=27)
        body += rect(840, y, 660, 124) + text(860, y+37, f'GPU {i}  ·  same model weights', 27, 600)
        body += text(860, y+87, 'Positions', 26) + tokens(998, y+55, range(1+4*i, 5+4*i), w=104, h=48)
    body += text(75, 669, 'A and B do not attend to each other.', 28, color=MUTED)
    body += text(840, 669, 'Positions 1–8 still form one context.', 28, color=BLUE)
    body += takeaway('Pure CP partitions token activations; it does not partition model weights.',
                      'Diagrams use CP alone. CP can also be combined with tensor, pipeline, and data parallelism.')
    slides.append(slide('Context parallelism splits one long sequence', body,
        """Start with the workload, not the acronym. DP processes different examples; CP assigns parts of one example to different devices. The eight-position drawing is a small stand-in for a very long sequence. Weights are replicated within this pure CP group, while the positions whose activations are owned by each device differ. In a hybrid TP/PP/CP layout, CP peers hold the same TP/PP parameter slice rather than necessarily the entire model. LayerNorm and token-wise MLP computation can operate on owned token rows. Attention cannot treat those token shards as independent examples. During training, gradients for replicated parameters must also be combined across CP peers; CP is not independent unsynchronized training. Insu Jang’s supplied article gives an accessible roadmap; Megatron’s current documentation specifies the implementation scope.
Sources: """ + CP + '\nBackground reading: ' + JANG,
        [('Megatron CP', CP), ('Insu Jang: CP overview', JANG)], section='Context parallelism'))

    body = text(75, 207, 'One sequence, eight positions, causal attention. Consider the query at position 6.', 29)
    body += rect(75, 256, 675, 187) + rect(825, 256, 675, 187)
    body += text(97, 297, 'GPU 0 owns positions 1–4', 30, 700)
    body += text(847, 297, 'GPU 1 owns positions 5–8', 30, 700)
    body += text(97, 337, 'K and V', 27) + tokens(240, 316, [1,2,3,4], w=112)
    body += text(847, 337, 'K and V', 27) + tokens(990, 316, [5,6,7,8], w=112, active={5,6})
    body += text(97, 411, 'Remote keys needed by query 6', 27, color=BLUE)
    body += text(847, 411, 'Keys 7–8 are future positions: masked', 27, color=MUTED)
    body += label_box(510, 497, 550, 68, 'Query 6 needs keys 1, 2, 3, 4, 5, 6', size=29)
    body += arrow(412, 445, 610, 496) + arrow(1150, 445, 960, 496)
    body += lines(75, 617, ['The output uses one softmax over all six allowed keys.',
                          'Computing attention only on positions 5–8 would change the model’s result.'], 29, 44)
    body += takeaway('Sharding storage does not remove cross-token dependencies.',
                      'The communication scheme must preserve the original attention mask and position IDs.')
    slides.append(slide('Attention must still cross the token shards', body,
        """Here each numbered cell stands for both the key and value belonging to that original token position, for one attention head. The query q6 resides on GPU 1. The correct causal key set includes positions 1 through 6, including q6’s own position. Positions 7 and 8 must not contribute even if their KV data happen to be on the same GPU. Position ownership is an implementation detail; causal visibility follows original positions. The complete softmax normalization spans the union of allowed keys. Attention restricted to the GPU-local chunk is a different sparse-attention computation, not exact context parallelism. Ring exchange and Ulysses are two ways to satisfy this dependency.
Sources: """ + RING + '\n' + CP,
        [('Ring Attention', RING), ('Megatron CP', CP)], section='Context parallelism'))

    scenes = []
    for phase in [0, 1, 2]:
        phase_label = ['Step 1: compute against the local KV shard.',
                       'Step 2: exchange KV shards; keep Q and accumulated summaries in place.',
                       'Finish: every query has processed all of its allowed keys.'][phase]
        scene = text(75, 246, phase_label, 29, 600, color=BLUE)
        for gpu, x in [(0,75),(1,825)]:
            scene += rect(x, 279, 675, 336)
            scene += text(x+22, 320, f'GPU {gpu}', 29, 700)
            scene += label_box(x+22, 339, 630, 57, f'Q stays here: positions {"1–4" if gpu == 0 else "5–8"}', size=28)
            kv = ('1–4' if gpu == 0 else '5–8') if phase == 0 else ('5–8' if gpu == 0 else '1–4')
            if phase < 2:
                scene += label_box(x+22, 415, 630, 56, f'{"Local" if phase == 0 else "Received"} K/V: positions {kv}', size=28)
                detail = (['For query i, use keys 1…i.', 'Keep the running summary (m, ℓ, u).'] if gpu == 0 else
                          ['For query i, use local keys 5…i.', 'Keep the running summary (m, ℓ, u).']) if phase == 0 else (
                          ['All these keys are in the future.', 'Mask this block; keep the prior summary.'] if gpu == 0 else
                          ['Keys 1–4 are valid for every local query.', 'Merge their contribution into (m, ℓ, u).'])
                scene += lines(x+22, 516, detail, 26, 39)
            else:
                scene += label_box(x+22, 415, 630, 56, f'Output O for positions {"1–4" if gpu == 0 else "5–8"}', size=28)
                scene += lines(x+22, 516, ['Normalize each query: O = u / ℓ.',
                                         'Continue the layer on these token rows.'], 27, 39)
        if phase == 1:
            scene += arrow(750,438,825,438)
            scene += line(825,469,750,469,BLUE,2) + f'<path d="M 758 464 L 750 469 L 758 474" fill="none" stroke="{BLUE}" stroke-width="2"/>'
        scenes.append(scene)
    body = text(75, 201, 'Two GPUs; each initially owns four Q/K/V rows. This is a scheduling schematic.', 28)
    body += steps('cp-ring', scenes)
    body += takeaway('Stream KV blocks past fixed queries; do not collect the full score matrix.',
                      'With more GPUs, repeat the exchange. Computation may overlap transfer when enough work is available.')
    slides.append(slide('Ring attention keeps Q local and circulates KV', body,
        """Start at local computation, then press Next step twice. The first frame is the diagonal pair of causal attention blocks. In the second, GPU 0 receives only future keys and can skip their computation; GPU 1 receives past keys and adds a second contribution. A two-GPU ring has one remote shard per device. General P-GPU rings process local KV plus P−1 remote shards, with causal optimizations able to skip fully masked work. This is not a claim that both GPUs perform equal work in the toy. The query shard and its running per-query statistics stay local while KV communication uses temporary buffers. Original owned KV can be retained or recomputed according to the implementation and training policy. Incoming KV need not accumulate into a full-sequence resident tensor. The final output stays token-partitioned. Real ring implementations may overlap communication with attention, but useful overlap is conditional on compute per block, transfer time, and scheduling.
Source: """ + RING,
        [('Ring Attention, §3', RING)], section='Context parallelism'))

    body = lines(75, 202, ['For one query, summarize two disjoint sets of allowed keys: A and B.',
                          'Score sⱼ = q · kⱼ / √d.  Value vector vⱼ has dimension d.'], 29, 42)
    body += rect(75, 279, 690, 177) + rect(810,279,690,177)
    for x, name in [(98,'A'),(833,'B')]:
        body += text(x, 319, f'Keys in set {name}', 29, 700)
        body += text(x, 359, f'm{name} = maximum score in {name}', 27)
        body += text(x, 399, f'ℓ{name} = Σⱼ∈{name} exp(sⱼ − m{name})', 27)
        body += text(x, 439, f'u{name} = Σⱼ∈{name} exp(sⱼ − m{name}) vⱼ', 27)
    body += text(75, 502, '1. Use the same reference maximum', 29, 700)
    body += text(710, 502, 'm = max(mA, mB)', 31, 600, color=BLUE)
    body += text(710, 543, 'α = exp(mA − m),   β = exp(mB − m)', 28)
    body += line(75,562,1500,562)
    body += text(75, 607, '2. Rescale, add, then normalize', 29, 700)
    body += text(710, 607, 'ℓ = αℓA + βℓB,     u = αuA + βuB', 29)
    body += text(710, 657, 'O = u / ℓ', 34, 700, color=BLUE)
    body += takeaway('Each shard contributes to one global softmax—not an average of local outputs.',
                      'm and ℓ are scalars; u and O are vectors. This is the Week 2 online-softmax merge, per query.')
    slides.append(slide('Merge summaries, not local softmax outputs', body,
        """The two sets partition the visible keys of one query, not two independent queries. Define scores including the usual head-dimension scaling. m is a stability reference; ℓ is the normalizer in that reference; u is the unnormalized weighted sum of value vectors. Each set has initially used a different maximum, so its ℓ and u must be multiplied by exp(old maximum − new maximum) before addition. Both the numerator and denominator must be rescaled. A plain arithmetic average of O_A and O_B is generally wrong because the two sets have different total probability mass. The merged output equals dense attention in exact arithmetic, with floating-point-order differences in practice. Fully masked sets contribute zero and should be skipped rather than evaluating undefined −infinity differences. This algebra is an original derivation of the online-softmax reduction used in memory-efficient attention and ring implementations.
Sources: """ + FLASH + '\n' + RING,
        [('FlashAttention', FLASH), ('Ring Attention', RING)], section='Context parallelism'))

    body = text(75, 202, 'Causal query i sees i keys. Later query positions therefore have more work.', 29)
    body += text(75, 262, 'Contiguous assignment', 31, 700)
    body += text(840, 262, 'Balanced early + late assignment', 31, 700)
    body += line(790,244,790,639)
    body += text(75, 311, 'GPU', 26, 700) + text(204, 311, 'Query positions', 26, 700) + text(493,311,'Allowed Q–K pairs',25,700)
    body += text(840,311,'GPU',26,700) + text(969,311,'Query positions',26,700) + text(1260,311,'Q–K pairs',26,700)
    for i in range(4):
        y=335+73*i
        body += text(96,y+35,i,28,600) + tokens(240,y,[2*i+1,2*i+2],w=84,h=49)
        body += text(560,y+35,4*i+3,30,700,color=BLUE)
        body += text(861,y+35,i,28,600) + tokens(1005,y,[i+1,8-i],w=84,h=49)
        body += text(1310,y+35,9,30,700,color=BLUE)
    body += line(75,649,1500,649)
    body += lines(75, 688, ['Same 36 valid pairs; the ownership changes, not the attention mask.'], 29)
    body += takeaway('Keep the original position IDs; never infer causality from local storage order.',
                      'The toy balances pair counts. Real time also depends on kernel tiles, communication, and sequence layout.')
    slides.append(slide('Balance causal work across the GPUs', body,
        """This is an original eight-position, four-GPU arithmetic example. Contiguous pairs of query positions have valid-key counts 1+2=3, 3+4=7, 5+6=11, 7+8=15. Assigning mirrored pairs gives 1+8=2+7=3+6=4+5=9. There are still 8×9/2=36 allowed attention interactions. The example illustrates zigzag-style balanced chunk assignment: with larger sequences one pairs early and late chunks, not necessarily individual tokens. This is related to, but not identical to, Striped Attention’s uniform interleaving of tokens. The source paper establishes the causal load-imbalance motivation; Megatron exposes contiguous-to-zigzag layout conversion. Preserve original absolute/rotary positions and sequence boundaries. Communication kernels must also know those positions so that future tokens are masked even when storage order is permuted. Equal pair counts do not guarantee equal wall time; block geometry matters.
Sources: """ + STRIPED + '\n' + LAYOUT,
        [('Striped Attention', STRIPED), ('Megatron zigzag layout', LAYOUT)], section='Context parallelism'))

    body = text(75,202,'Example: eight tokens, four attention heads, two GPUs. Q, K and V follow the same exchange.',28)
    xs=[75,590,1105]
    headings=['Before attention','Compute attention','After attention']
    for stage,x in enumerate(xs):
        body += text(x,262,headings[stage],30,700)
        for gpu,y in [(0,293),(1,448)]:
            body += rect(x,y,395,137)
            body += text(x+16,y+34,f'GPU {gpu}',28,700)
            if stage==1:
                desc=[f'All positions 1–8',f'Heads {"0–1" if gpu==0 else "2–3"}']
            else:
                desc=[f'Positions {"1–4" if gpu==0 else "5–8"}', 'All four heads']
            body += lines(x+16,y+77,desc,27,36)
    body += arrow(471,386,585,386) + arrow(986,386,1100,386)
    body += text(528,348,'All-to-All',23,600,anchor='middle',color=BLUE)
    body += text(1043,348,'All-to-All',23,600,anchor='middle',color=BLUE)
    body += text(75,635,'Exchange Q/K/V',28,700) + text(590,635,'Each head sees its full context',26,700) + text(1105,635,'Exchange the output',28,700)
    body += text(75,692,'All-to-All: each GPU sends different tensor slices to the other GPUs.',29)
    body += takeaway('Ulysses redistributes activations: token shards → head shards → token shards.',
                      'Here MHA has four KV heads too. Head counts and implementation constraints limit the exchange layout.')
    slides.append(slide('Ulysses trades token shards for head shards', body,
        """Read this original diagram left to right. Initially each device has four token rows and all four heads for those rows. GPU 0 keeps its head-0/1 slices and sends head-2/3 slices to GPU 1, while GPU 1 sends its head-0/1 slices to GPU 0 and keeps head-2/3. After the All-to-All, each device has eight token rows but only two heads: total per-device Q/K/V element count is unchanged. Each local head can then run a normal full-context attention kernel, including FlashAttention. A second All-to-All on attention outputs restores four token rows and all four heads to each original owner. This is activation-layout redistribution, not weight tensor parallelism: pure Ulysses keeps projection and MLP weights replicated. For the toy, Q/K/V all have four heads (MHA). GQA/MQA and combinations with TP require care about available KV-head partitions, replication, or specialized schemes. Do not claim arbitrary CP degree is possible or that Ulysses always outperforms ring attention.
Source: """ + ULYSSES + '\nAdditional walkthrough: ' + JANG,
        [('DeepSpeed Ulysses', ULYSSES), ('Insu Jang: Ulysses', JANG)], section='Context parallelism'))

    body = text(75,202,'“Sequence parallelism” is overloaded. State which implementation you mean.',29)
    body += rect(75,254,690,420) + rect(810,254,690,420)
    body += text(98,302,'Megatron sequence parallelism (SP)',29,700)
    body += text(833,302,'Context parallelism (CP)',30,700)
    body += lines(98,360,['Works together with tensor parallelism.',
                          'Shards selected token-wise activations',
                          '(for example, LayerNorm and dropout).'],27,43)
    body += lines(833,360,['Keeps token rows partitioned across layers.',
                          'Each GPU owns a subset of token rows.',
                          'Attention needs a cross-shard algorithm.'],27,43)
    body += line(98,479,740,479) + line(833,479,1475,479)
    body += lines(98,525,['Typical communication around TP regions:',
                          'AllGather → compute → ReduceScatter.',
                          'Goal: reduce activation replication in TP.'],27,46)
    body += lines(833,525,['Examples: ring KV exchange or Ulysses.',
                          'Can operate without tensor parallelism.',
                          'Goal: distribute long-context work/storage.'],27,46)
    body += takeaway('Ask what is sharded, during which operators, and what must communicate.',
                      'Ulysses also uses the name “sequence parallelism”; it is not the same mechanism as Megatron SP.')
    slides.append(slide('Context parallelism and Megatron SP differ', body,
        """This is a terminology clarification after teaching the concrete mechanisms. In Megatron’s TP-associated SP, selected activations outside tensor-parallel linear regions are sequence-partitioned. A typical forward region gathers token rows before TP computation and returns a sequence shard with ReduceScatter instead of leaving an AllReduce-replicated activation. CP retains token ownership across the network and introduces distributed attention to satisfy remote-key dependencies. Do not generalize the label SP to every paper: Ulysses is explicitly called sequence parallelism by DeepSpeed, and older literature also uses SP for long-sequence attention distribution. CP and TP-associated SP can coexist. The ownership and exchange diagrams are more reliable than an acronym alone.
Sources: """ + CP + '\n' + ULYSSES,
        [('Megatron CP vs SP', CP), ('DeepSpeed Ulysses', ULYSSES)], section='Context parallelism'))

    body = lines(75,201,['Prefill has many query positions to split. Decode has only one new query per request.',
                         'For long-context decode, distribute the stored KV history and combine partial attention.'],28,42)
    body += label_box(75,326,278,89,'New query q',size=30)
    body += arrow(353,361,500,331) + arrow(353,391,500,508)
    for gpu,y,keys in [(0,282,'History positions 1–4'),(1,467,'History positions 5–8')]:
        body += rect(500,y,475,142)
        body += text(522,y+35,f'GPU {gpu}: KV shard',28,700)
        body += text(522,y+75,keys,27)
        body += text(522,y+113,'q attends to this shard → (m, ℓ, u)',26)
        body += arrow(975,y+71,1114,445)
    body += rect(1115,350,385,190)
    body += text(1135,393,'Merge both summaries',28,700)
    body += text(1135,436,'One global normalizer',27)
    body += text(1135,488,'Final attention output',28,700,color=BLUE)
    body += lines(75,658,['Same exact-attention principle; different work distribution and communication.',
                         'The query must reach KV owners; the final result must reach its next operator.'],28,42)
    body += takeaway('Long-prefill CP and decode KV sharding are related, but not the same execution plan.',
                      'Schematic for one query/head and eight visible keys; the new KV entry must also be assigned to an owner.')
    slides.append(slide('Decode parallelism shards the cached history', body,
        """Return to Week 2’s distinction: prefill supplies many new query rows, while ordinary autoregressive decode supplies one per active request. For the diagram, q is a query whose visible history is positions 1–8; the current token’s own K/V may be included among these eight depending on where the step is drawn. Each shard computes a partial attention summary, and the global output follows the same max-rescaled merge as the earlier slide. A production implementation may exchange output and log-sum-exp rather than literal m,ℓ,u triples; these encode equivalent normalizer information. The drawing explains the dependency, not a prescribed NCCL sequence. vLLM separates prefill and decode context parallelism and can use DCP inside an existing TP group to reduce KV duplication; therefore DCP need not multiply the total GPU count. Explain the runtime’s particular rank layout before writing a GPU-count product. Sharding stored KV does not automatically speed up weight projections or the MLP, and communication may exceed the saved KV-read time at short contexts.
Sources: """ + DCP + '\n' + DCP_BLOG,
        [('vLLM context parallelism', DCP), ('vLLM DCP design', DCP_BLOG)], section='Context parallelism'))

    body = text(75,202,'For fixed batch size and sequence length, adding CP redistributes existing work.',29)
    body += table(75,260,1425,['Resource','Effect of pure context parallelism'],[
        ('Token activations / owned KV','Distributed across CP ranks; buffers add overhead.'),
        ('Model weights','Replicated within the CP group.'),
        ('Full-attention arithmetic','Still quadratic in sequence length across the group.'),
        ('Network traffic','Remote attention data or summaries must be exchanged.')
    ],ratios=[.37,.63],row_h=74,size=27)
    body += lines(75,672,['FlashAttention avoids storing the full score matrix; CP distributes the remaining context state.'],28)
    body += takeaway('Use CP for long-context pressure; combine other parallelism when weights are the problem.',
                      'Useful overlap and speedup depend on context length, causal balance, head layout, and interconnects.')
    slides.append(slide('What context parallelism saves', body,
        """This closes the section by connecting the mechanism to resource accounting. For fixed global work and a balanced pure-CP partition, the primary owned token activations scale approximately as S/P per rank; peak memory also includes attention statistics, communication/double buffers, backward saved tensors, parameters, and optimizer state. Do not promise total allocated HBM falls exactly by P. Dense causal attention still has S(S+1)/2 valid query-key pairs per head across all ranks; distributing them does not make the mathematical workload linear. FlashAttention has already removed the need to keep an S×S score/probability tensor in HBM, so CP should not be motivated only by splitting that materialized tensor. Pure CP replicates weights; combine FSDP/ZeRO, TP or PP for parameter/state capacity. Long blocks can provide enough attention computation to overlap ring transfer; shorter blocks, weak links or imbalance may expose communication. Show memory, kernel time and collective time in measurement before assuming additional CP improves latency.
Sources: """ + RING + '\n' + FLASH + '\n' + CP,
        [('Ring Attention', RING), ('FlashAttention', FLASH)], section='Context parallelism'))
    return slides
