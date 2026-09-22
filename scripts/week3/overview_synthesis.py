"""Five diagram-led synthesis slides for the short Week 3 overview."""
from .common import *

VLLM = ('vLLM parallelism', 'https://docs.vllm.ai/en/stable/serving/parallelism_scaling/')
MEG = ('Megatron-LM', 'https://arxiv.org/abs/2104.04473')
SCALE = ('Scaling Book: inference', 'https://jax-ml.github.io/scaling-book/inference/')
PLAY = ('Ultra-Scale Playbook', 'https://nanotron-ultrascale-playbook.static.hf.space/index.html')
TP_CHECK = ('Megatron: TP constraints', 'https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py')
CP_CHECK = ('Megatron: CP constraints', 'https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py')


def _hybrid_groups():
    b = text(75, 211, '8 GPUs = 2 serving replicas × 2 PP stages × 2 TP shards', 34, 700, color=BLUE)
    for replica, top in enumerate((280, 498)):
        b += text(75, top + 57, f'Replica {replica}', 30, 700)
        b += text(75, top + 103, f'Requests {"A, B" if replica == 0 else "C, D"}', 28)
        b += arrow(270, top + 82, 306, top + 82)
        for stage, left in enumerate((320, 1015)):
            b += rect(left, top, 510, 156, 'white')
            b += text(left + 255, top + 37, f'PP stage {stage}', 29, 700, anchor='middle')
            for shard, dx in enumerate((25, 315)):
                gpu = replica * 4 + stage * 2 + shard
                b += label_box(left + dx, top + 55, 170, 60, f'GPU {gpu}', size=29)
            b += arrow(left + 208, top + 80, left + 300, top + 80)
            b += arrow(left + 300, top + 92, left + 208, top + 92)
            b += text(left + 255, top + 144, 'TP = 2', 27, anchor='middle', color=BLUE)
        b += arrow(844, top + 83, 1002, top + 83)
    b += text(923, 252, 'Hidden states', 27, anchor='middle')
    b += text(75, 710, 'Training additionally synchronizes gradients between matching weight shards.', 29)
    b += takeaway('DP separates inputs; PP separates layers; TP separates work inside a layer.')
    notes = (
        'This diagram combines three dimensions without introducing another parallelism algorithm. '
        'Each row is one logical model replica spread across four GPUs. Within that row, the first two GPUs '
        'own the early layers and the second pair own the later layers. Within each pair, tensor parallelism '
        'partitions the weights and computation inside those layers, with the required tensor collectives. '
        'Hidden states flow from pipeline stage zero to stage one for the same inputs. The second replica '
        'uses an identical model partition but different inputs. For ordinary replicated training DP, '
        'corresponding weight shards synchronize gradients: for example, GPU zero with GPU four, rather '
        'than reducing all eight GPUs as though they held the same weights. Independent serving replicas '
        'instead accept separate requests and do not perform training gradient synchronization. The diagram '
        'shows logical groups, not physical nodes; deciding where those groups should run is the next step. '
        'Communication arrows describe dependencies, not a measured communication schedule.'
    )
    return slide('Combine groups to build one GPU layout', b, notes,
                 (MEG, VLLM), section='Choosing a layout')


def _placement():
    b = text(75, 208, 'Typical assumption: fast GPU links within a node; a slower network between nodes.', 29)
    for node, left in enumerate((75, 875)):
        b += rect(left, 273, 650, 250, 'white')
        b += text(left + 28, 317, f'Node {node} · PP stage {node}', 30, 700, color=BLUE)
        for shard, dx in enumerate((30, 365)):
            b += label_box(left + dx, 355, 255, 78, f'TP shard {shard}', size=30)
        b += arrow(left + 294, 388, left + 354, 388)
        b += arrow(left + 354, 400, left + 294, 400)
        b += text(left + 325, 490, 'Frequent layer collectives', 28, anchor='middle')
    b += arrow(738, 391, 862, 391)
    b += text(800, 567, 'PP transfers hidden states between stages', 30, 700, color=BLUE, anchor='middle')
    b += line(75, 602, 1525, 602)
    b += text(75, 650, 'DP replicas often span nodes. CP and EP can also span nodes.', 30)
    b += text(75, 699, 'These are placement choices, not restrictions imposed by the method.', 29)
    b += takeaway('Place frequent, latency-sensitive communication on the fastest available links.')
    notes = (
        'The example places a two-way TP group inside each node and pipeline boundaries between nodes. '
        'This is a common starting configuration when local GPU links are substantially faster than the '
        'inter-node network. TP frequently communicates within a Transformer layer, so exposed collective '
        'latency and bandwidth can directly delay the next operation. PP sends hidden states between '
        'stages, and corresponding activation gradients during training. Ordinary training DP replicates '
        'the model partition and synchronizes gradients; some communication can overlap backward work. '
        'Sharded DP such as FSDP also gathers parameters, so it deserves a separate communication analysis. '
        'CP and EP are not limited to a node: their attention exchanges or token routing can use inter-node '
        'networks when the implementation and topology support it. This schematic does not require that '
        'a physical server boundary equal a fast-interconnect boundary. Benchmark the actual network, '
        'message sizes, overlap, and workload instead of treating the acronym as a placement rule.'
    )
    return slide('Choose placement from the communication pattern', b, notes,
                 (VLLM, PLAY), section='Choosing a layout')


def _scaling():
    b = text(75, 208, 'One layer’s critical path: local computation + communication that is not hidden.', 30)
    start = 325
    b += text(75, 345, 'Fewer GPUs', 31, 700)
    b += rect(start, 286, 730, 98, PALE, BLUE)
    b += text(start + 365, 347, 'Local computation', 30, 700, anchor='middle')
    b += rect(start + 730, 286, 210, 98, '#eee6dc', WARM)
    b += text(start + 835, 347, 'Exchange', 29, anchor='middle')
    b += text(75, 521, 'More GPUs', 31, 700)
    b += rect(start, 462, 425, 98, PALE, BLUE)
    b += text(start + 212.5, 523, 'Less local work', 30, 700, anchor='middle')
    b += rect(start + 425, 462, 475, 98, '#eee6dc', WARM)
    b += text(start + 662.5, 523, 'More exposed communication', 29, anchor='middle')
    b += line(start + 940, 271, start + 940, 578, MUTED, 1.5, '7 6')
    b += arrow(start, 602, 1455, 602)
    b += text(1455, 645, 'Elapsed time →', 27, anchor='end', color=MUTED)
    b += text(75, 703, 'Schematic only: smaller GEMMs, extra transfers, and waiting can offset the compute savings.', 27)
    b += takeaway('More GPUs improve latency only when saved work exceeds added overhead.')
    notes = (
        'The bars are a qualitative illustration, not benchmark measurements or a prediction for a particular '
        'model. The blue section represents local computation on the critical path. The warm section '
        'represents communication and waiting that remain exposed after any overlap. Dividing a layer '
        'among more GPUs can reduce local work, but the reduction need not be proportional: smaller GEMMs '
        'may run less efficiently, and communication startup does not shrink with tensor size. A larger '
        'group or a slower network boundary can increase communication cost. The drawn result shows only '
        'a small end-to-end improvement despite visibly less local computation. Other workloads can show '
        'strong scaling, no improvement, or a regression. Compare the same model, precision, batch, '
        'context length, and objective, and inspect exposed communication rather than summing every '
        'communication operation regardless of overlap. A configuration that improves single-request '
        'latency may still reduce throughput for a fixed total GPU budget by using fewer independent replicas.'
    )
    return slide('Communication can offset the benefit of more GPUs', b, notes,
                 (SCALE, VLLM), section='Choosing a layout')


def _degrees():
    b = text(75, 204, 'Degree = number of GPUs in the group. Shapes must support the chosen split.', 30)
    b += line(800, 250, 800, 697)
    b += text(75, 292, 'TP: 8 query heads', 32, 700, color=BLUE)
    b += text(75, 349, '4 GPUs → 2 heads each', 29)
    for gpu in range(4):
        xx=75+gpu*169
        b += rect(xx,375,151,93,'white')
        b += text(xx+75,407,f'GPU {gpu}',25,600,anchor='middle')
        for h in range(2):b += rect(xx+24+h*53,424,46,26,PALE,BLUE)
    b += text(75, 543, '3 GPUs → no equal split', 29, color=WARM)
    for gpu in range(3):
        xx=75+gpu*169
        b += rect(xx,573,151,70,'white')
        for h in range(2):b += rect(xx+24+h*53,595,46,26,PALE,BLUE)
    for h in range(2):b += rect(604+h*55,595,46,26,'#f0e8df',WARM)
    b += text(75, 691, '2 heads left over', 28, color=WARM)
    b += text(850, 292, 'CP: 9 token positions', 32, 700, color=BLUE)
    b += text(850, 349, '3 GPUs → 3 positions each', 29)
    for gpu in range(3):
        xx=850+gpu*226
        b += rect(xx,375,205,115,'white')
        b += text(xx+102,414,f'GPU {gpu}',27,600,anchor='middle')
        for t in range(3):b += label_box(xx+13+t*61,432,58,43,str(3*gpu+t+1),size=25)
    b += text(850, 559, 'This equal token split is possible.', 29)
    b += lines(850, 627, ['Framework and attention algorithms', 'can impose additional constraints.'],27,gap=43)
    b += takeaway('Powers of two are common starting points, not a universal requirement.',
                   'Toy equal partitions. A valid shape split alone does not guarantee backend support.')
    return slide('Choose a degree that fits the model', b,
        'A parallel degree counts the GPUs or ranks in that communication group. These simple examples separate tensor '
        'divisibility from the belief that a degree must always be even or a power of two. Eight query heads can be '
        'partitioned into four equal groups of two. Trying the same equal-head scheme on three ranks leaves two heads '
        'unassigned. Other model dimensions and KV layouts impose further requirements. On the right, nine positions '
        'can be partitioned into three contiguous groups of three, so the basic CP partition itself does not require '
        'an even number of ranks. This is a generic equal-token partition, not a claim that a specific Megatron '
        'configuration accepts sequence length nine. For example, Megatron’s balanced causal layout commonly uses '
        '2*CP chunks and therefore imposes a different divisibility check. Head-redistribution algorithms may have '
        'head-count restrictions too. Backend support, layout, GPU count, and kernel constraints must all be checked. '
        'The full reference lecture gives the implementation-specific examples.',
        (TP_CHECK, CP_CHECK), section='Choosing a layout')


def _mini_replica(x, y):
    out = ''
    for dx in (0, 54):
        out += rect(x + dx, y, 38, 34, PALE, BLUE)
        out += line(x + dx + 7, y + 11, x + dx + 31, y + 11, BLUE, 2)
        out += line(x + dx + 7, y + 23, x + dx + 31, y + 23, BLUE, 2)
    return out


def _mini_tp(x, y):
    return rect(x, y, 92, 34, PALE, BLUE) + line(x + 46, y, x + 46, y + 34, BLUE, 2)


def _mini_pp(x, y):
    return rect(x, y, 32, 34, PALE, BLUE) + rect(x + 60, y, 32, 34, PALE, BLUE) + arrow(x + 37, y + 17, x + 54, y + 17)


def _mini_cp(x, y):
    return ''.join(rect(x + i * 16, y + 4, 13, 26, PALE if i < 3 else 'white', BLUE) for i in range(6))


def _mini_ep(x, y):
    return ''.join(rect(x + 55 * (i % 2), y - 3 + 22 * (i // 2), 37, 17, PALE, BLUE) for i in range(4))


def _mini_state(x, y):
    return ''.join(rect(x, y + i * 13, 92, 10, PALE if i % 2 == 0 else 'white', BLUE) + line(x + 46, y + i * 13, x + 46, y + i * 13 + 10, BLUE, 1.5) for i in range(3))


def _summary():
    b = rect(75, 199, 1450, 60, PALE, 'none')
    for x, value in ((218, 'Method'), (555, 'What is split?'), (980, 'What is exchanged?')):
        b += text(x, 240, value, 29, 700)
    rows = [
        ('DP', 'Inputs', 'Training gradients', _mini_replica),
        ('TP', 'Work inside a layer', 'Activation pieces / sums', _mini_tp),
        ('PP', 'Layers', 'Stage inputs / gradients', _mini_pp),
        ('CP', 'One sequence', 'Attention data', _mini_cp),
        ('EP', 'Experts', 'Routed tokens / outputs', _mini_ep),
        ('ZeRO / FSDP', 'Training state', 'Gradients; parameters*', _mini_state),
    ]
    for i, (method, split, exchanged, icon) in enumerate(rows):
        y = 260 + i * 68
        b += icon(92, y + 15)
        b += text(218, y + 44, method, 29, 700, color=BLUE)
        b += text(555, y + 44, split, 29)
        b += text(980, y + 44, exchanged, 28)
        b += line(75, y + 66, 1525, y + 66)
    b += text(75, 709, '*The exchanged training state depends on the sharding stage and implementation.', 27)
    b += takeaway('For every layout, identify tensor ownership and the communication needed next.')
    notes = (
        'Use this final table as a compact map rather than six separate prescriptions. DP separates '
        'inputs while replicating the logical model; the gradient entry refers to synchronized training, '
        'not independent inference replicas. TP divides work inside layers, producing activation shards '
        'or partial sums that require the appropriate collective. PP divides layer ownership and sends '
        'hidden states forward and activation gradients backward. CP divides one sequence and exchanges '
        'attention information; the exact payload can be KV chunks, head-transposed tensors, or partial '
        'attention results depending on the algorithm and phase. EP divides expert ownership and moves '
        'selected token activations to experts before returning and combining outputs. ZeRO and FSDP '
        'divide training state: optimizer state, gradients, and possibly parameters depending on the '
        'stage and implementation. Their collectives therefore differ. These axes can be combined, but '
        'their group sizes, memory savings, and communication costs must be evaluated in the resulting '
        'layout. The small drawings recall the partition patterns; they are not exact tensor shapes.'
    )
    return slide('Six methods: tensor ownership and communication', b, notes,
                 (PLAY, MEG), section='Choosing a layout')


def get_slides():
    return [_hybrid_groups(), _placement(), _scaling(), _degrees(), _summary()]
