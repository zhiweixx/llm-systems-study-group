"""Practical configuration and placement lessons for the Week 3 deck."""
from .common import *

TP_CHECK = ('Megatron: TP constraints', 'https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py')
CP_CHECK = ('Megatron: sequence constraints', 'https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py')
VLLM = ('vLLM: parallel layouts', 'https://docs.vllm.ai/en/stable/serving/parallelism_scaling/')
FSDP = ('PyTorch: hybrid sharding', 'https://docs.pytorch.org/docs/main/fsdp.html')
HYBRID_CP = ('Megatron: hierarchical CP', 'https://docs.nvidia.com/nemo/megatron-bridge/0.4.0/training/hybrid-context-parallel.html')
MULTINODE_EP = ('vLLM: multi-node EP', 'https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/')


def degree_lessons():
    b = text(75, 190, '1, 2, 4, 8, … are common starting points: powers of two, not all multiples of two.', 28)
    b += text(75, 234, 'The actual constraints come from tensor partitions and the chosen implementation.', 28)
    b += line(794, 282, 794, 664)
    b += text(75, 308, 'TP: can the model be split this way?', 30, 700, color=BLUE)
    b += lines(75, 359, ['For equal query-head partitions:', 'number of query heads must be divisible by TP.'], 27, gap=41)
    b += label_box(75, 434, 660, 64, '32 heads / TP 4 = 8 heads per GPU', size=27)
    b += text(75, 544, '32 heads / TP 6 is not an integer.', 28)
    b += lines(75, 601, ['FFN dimensions, KV-head layout, and kernels', 'can add further constraints.'], 26, gap=39)
    b += text(844, 308, 'CP: can the sequence be partitioned?', 30, 700, color=BLUE)
    b += lines(844, 359, ['Example: Megatron’s balanced causal split', 'uses 2 × CP chunks of equal length.'], 27, gap=41)
    b += label_box(844, 434, 681, 64, '6,144 tokens / (2 × CP 3) = 1,024', size=27)
    b += text(844, 544, 'CP 3 passes this sequence-length test.', 28)
    b += lines(844, 601, ['Ulysses also partitions heads; ring CP does not.', 'The selected backend must support the layout.'], 26, gap=39)
    b += text(75, 706, 'An integer partition is one necessary check; it does not certify the full configuration.', 27)
    b += takeaway('Use shapes, topology, and backend support to choose TP and CP.',
                   'Neither “must be even” nor “must be a power of two” is a universal rule.')
    return [slide('TP and CP degrees must match the tensor layout', b,
        'Typical GPU groups and common Transformer dimensions make 1, 2, 4, 8, and 16 convenient starting points. '
        'These are powers of two, not the set of all even numbers. This is a practical starting heuristic rather than a mathematical requirement. '
        'In Megatron, the number of query heads must be divisible by TP for its equal partition. Thus 32 heads can split over TP4 '
        'but not TP6 in this design. This does not certify a complete configuration: feedforward dimensions, KV head handling, '
        'kernel alignment, communication backend, and available ranks also matter. A model with 12 query heads can pass this '
        'head-partition test at TP3, provided the remaining requirements are met. As a concrete implementation counterexample '
        'to the power-of-two claim, vLLM custom_all_reduce.py lists supported world sizes including 6; this is only collective support, '
        'not support for arbitrary models at TP6. Source: https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py . '
        'For CP, Megatron’s ordinary balanced causal sequence partition checks sequence_length modulo 2*CP. '
        'CP3 therefore creates six chunks; 6144/6=1024. The factor two comes from assigning two chunks per rank to balance '
        'causal attention, as shown earlier. It does not require CP itself to be even. This example checks partition arithmetic '
        'only; backend and layout-conversion paths may impose additional restrictions. Classic Ulysses partitions heads as well as '
        'redistributing tokens, so head divisibility is another requirement, whereas ring CP does not distribute work by sharding heads. '
        'See https://www.deepspeed.ai/tutorials/ds-sequence/ for Ulysses. Always validate the specific framework version and model.',
        (TP_CHECK, CP_CHECK), section='Context parallelism')]


def placement_lessons():
    slides = []
    b = text(75, 190, 'Assume fast GPU links inside each node and a slower network between nodes.', 28)
    b += text(75, 234, 'These are starting layouts to benchmark, not restrictions on where a method can run.', 28)
    cols = [75, 325, 944]
    b += rect(75, 279, 1450, 64, PALE, 'none')
    for x, h in zip(cols, ['Method', 'Communication pattern', 'Common placement']):
        b += text(x+14, 321, h, 28, 700)
    rows = [
        (343, 'TP', ['Activation collectives within many layers;', 'often on the next operation’s critical path.'],
         ['Keep each TP group on fast GPU links.', 'Often within a node.']),
        (452, 'PP', ['Hidden states between stages;', 'activation gradients in training.'],
         ['Often connect stages across nodes.', 'Communication at stage boundaries.']),
        (561, 'Replicated DP', ['Gradient synchronization in training;', 'can overlap with backward computation.'],
         ['Often extend replica groups across nodes.', 'Each replica executes the full model path.']),
    ]
    for y, name, pattern, choice in rows:
        b += line(75, y, 1525, y)
        b += text(89, y+48, name, 28, 700, color=BLUE)
        b += lines(339, y+43, pattern, 26, gap=39)
        b += lines(958, y+43, choice, 26, gap=39)
    b += line(75, 670, 1525, 670)
    b += text(75, 711, 'FSDP / ZeRO-3 adds frequent parameter gathers: its placement needs separate analysis.', 27)
    b += takeaway('Place latency-sensitive communication on the fastest available interconnect.',
                   'For example, hybrid FSDP shards within a node and replicates the shard group across nodes.')
    slides.append(slide('Placement follows communication, not the acronym', b,
        'The heuristic TP inside a node and PP/DP across nodes assumes a server with fast GPU-to-GPU links and a slower '
        'inter-server network. TP commonly places collectives on a layer’s critical path, so latency and bandwidth matter strongly. '
        'PP typically communicates activation tensors at stage boundaries, and corresponding activation gradients during training, '
        'rather than at every internal TP partition. Replicated training DP synchronizes gradient buckets; some transfers can '
        'overlap backward compute, and accumulation may avoid synchronization on intermediate microbatches. It is not communication-free '
        'or guaranteed to scale on a slow network. DP inference replicas normally serve independent requests and do not synchronize '
        'gradients, so the table’s DP communication refers to training. FSDP/ZeRO-3 is different from ordinary replicated DP: '
        'it repeatedly gathers parameters and reduces/scatters gradients, potentially making a cross-node sharding group expensive. '
        'PyTorch HYBRID_SHARD keeps full sharding within a node and replicates groups across nodes, reducing inter-node parameter traffic '
        'while retaining cross-node gradient synchronization. vLLM documents TP within nodes plus PP across nodes as one deployment '
        'layout and also supports TP across nodes. Fast network domains need not coincide with physical server boundaries. '
        'The correct choice follows actual connectivity, communication volume and frequency, overlap, workload, and memory capacity.',
        (VLLM, FSDP), section='Choosing a layout'))

    b = text(75, 190, 'CP exchanges attention data; EP routes token activations. Both can span nodes.', 29)
    b += text(75, 245, 'Example: hierarchical CP = 4 on two nodes, with two GPUs per node', 30, 700, color=BLUE)
    for node, x in enumerate([75, 845]):
        b += rect(x, 278, 680, 238, 'white')
        b += text(x+24, 316, f'Node {node}', 28, 700)
        b += label_box(x+24, 335, 632, 55, f'GPU {2*node}: head partition A', size=28)
        b += label_box(x+24, 434, 632, 55, f'GPU {2*node+1}: head partition B', size=28)
        b += arrow(x+266, 395, x+266, 429) + arrow(x+281, 429, x+281, 395)
        b += text(x+303, 420, 'Local All-to-All', 25)
    for y in (357, 456):
        b += arrow(762, y, 837, y) + arrow(837, y+12, 762, y+12)
    b += text(800, 557, 'KV exchange: GPU 0 ↔ 2; GPU 1 ↔ 3 (matching head partitions).', 28, anchor='middle')
    b += line(75, 585, 1525, 585)
    b += text(75, 628, 'EP across nodes', 30, 700, color=BLUE)
    b += lines(440, 626, ['Experts can live on remote GPUs; tokens are dispatched and outputs returned.',
                          'It needs an efficient inter-node network and balanced expert workloads.'], 27, gap=43)
    b += text(75, 709, 'Choose the group layout and communication algorithm together.', 29, 700)
    b += takeaway('CP and EP are not confined to one node.',
                   'CP example assumes the KV-head count supports two local head partitions, with TP = 1.')
    slides.append(slide('CP and EP can use communication across nodes', b,
        'This is a topology schematic of hierarchical context parallelism, not a complete tensor-layout trace. '
        'The four-rank CP group is factored into a local group size two and an outer group size two. '
        'Inside a node, an All-to-All redistributes token/head ownership, as in the Ulysses idea introduced earlier. '
        'Across the outer groups, point-to-point communication circulates KV data, as in ring attention. '
        'The diagram labels head ownership after the local All-to-All. Ranks 0 and 2 share one head partition and form '
        'one outer P2P group; ranks 1 and 3 share the other and form the second group. Each exchanges the corresponding KV '
        'while retaining its local queries. The KV-head count remaining after any TP must be divisible by the local '
        'All-to-All factor of two; the example uses TP1. Megatron Bridge documents this a2a+p2p strategy '
        'and hierarchical_context_parallel_sizes; its documentation includes a larger [8,2] arrangement. '
        'The two-node/two-GPU illustration here uses the same idea at smaller scale, with no measured performance claim. '
        'A CP group spanning nodes may use other algorithms; this diagram is not mandatory for all CP. '
        'For EP, expert weights remain on their owning ranks while token activations and expert outputs cross the network. '
        'vLLM explicitly documents multi-node EP deployment, including 16 GPUs across two nodes. The performance depends on '
        'routing volume, imbalance, message sizes, dispatch/combine implementation, network connectivity, and workload. '
        'Keeping frequent traffic local is a useful goal, but it does not establish that every non-DP/non-PP method must '
        'stay within one server. Modern fast interconnect domains can also span physical server boundaries.',
        (HYBRID_CP, MULTINODE_EP), section='Choosing a layout'))
    return slides
