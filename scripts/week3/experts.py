"""Week 3: expert parallelism, taught through one two-GPU routing example."""
from .common import *

MIXTRAL = ('Mixtral §2', 'https://arxiv.org/abs/2401.04088')
EP = ('vLLM: expert parallelism', 'https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/')
DP = ('vLLM: data parallelism', 'https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/')
DISPATCH = ('Megatron: token dispatch', 'https://docs.nvidia.com/megatron-core/developer-guide/0.19.0/apidocs/core/core.transformer.moe.token_dispatcher.html')


def _routing_scene(stage):
    titles = [
        '1. Dispatch token activations to the GPUs that own the selected experts',
        '2. Group the received tokens by expert, then run each expert MLP',
        '3. Return expert outputs to the original token owner and combine them',
    ]
    s = text(75, 208, titles[stage], 29, 600)
    for x, gpu in [(75, 0), (860, 1)]:
        s += rect(x, 245, 665, 345, 'white')
        s += text(x + 25, 292, f'GPU {gpu}', 30, 700, BLUE)
    if stage == 0:
        s += lines(110, 350, ['Original tokens: A, B', 'Keep A for E0 and B for E1.', 'Send A to E2 and B to E3.'], 28, 68)
        s += lines(895, 350, ['Original tokens: C, D', 'Keep C for E2 and D for E3.', 'Send C to E0 and D to E1.'], 28, 68)
        s += arrow(740, 371, 860, 371) + text(800, 352, 'A, B', 26, 600, BLUE, 'middle')
        s += line(860, 500, 740, 500, TEAL, 2)
        s += '<path d="M 748 495 L 740 500 L 748 505" stroke="#397b71" stroke-width="2" fill="none"/>'
        s += text(800, 480, 'C, D', 26, 600, TEAL, 'middle')
        s += text(75, 636, 'All-to-all: each GPU sends different token subsets to different destination GPUs.', 28)
    elif stage == 1:
        for x, e in [(105, 0), (425, 1), (890, 2), (1210, 3)]:
            tokens = 'A, C' if e % 2 == 0 else 'B, D'
            s += rect(x, 320, 285, 205, PALE)
            s += text(x + 142.5, 366, f'Expert E{e}', 30, 700, BLUE, 'middle')
            s += text(x + 142.5, 423, f'Inputs: {tokens}', 28, anchor='middle')
            s += text(x + 142.5, 481, '2 output vectors', 26, anchor='middle')
        s += text(75, 636, 'Four tokens × two selected experts = eight token–expert computations.', 28)
    else:
        s += text(110, 347, 'For token A, assume expert outputs:', 27)
        s += text(110, 394, 'E0(A) = [2, 4]    E2(A) = [10, 0]', 28)
        s += text(110, 466, 'yA = 0.75 × [2, 4] + 0.25 × [10, 0]', 27)
        s += text(110, 524, 'yA = [4, 3]', 34, 700, BLUE)
        s += lines(895, 350, ['GPU 1 receives E0(C) and E1(D).', 'It combines C and D using their', 'own router weights.', 'GPU 0 also combines B.'], 28, 57)
        s += text(75, 636, 'The return exchange restores token ownership; combine is a weighted sum, not a concatenation.', 27)
    return s


def get_slides():
    slides = []

    # An expert is a complete learned MLP, not a GPU or an attention head.
    s = text(75, 205, 'A mixture-of-experts (MoE) layer replaces one dense MLP with several expert MLPs.', 29)
    s += label_box(90, 369, 180, 90, 'Token x', size=30)
    s += arrow(270, 414, 340, 414)
    s += rect(340, 354, 260, 120, PALE)
    s += text(470, 400, 'Router', 32, 700, BLUE, 'middle')
    s += text(470, 444, 'Select top 2', 27, anchor='middle')
    for e, y in enumerate([260, 353, 446, 539]):
        selected = e in (0, 2)
        name = f'Expert E{e}' if selected else f'E{e} (skipped)'
        s += label_box(755, y, 230, 62, name, PALE if selected else '#f7f7f7', 28)
        if selected:
            s += arrow(600, 414, 755, y + 31)
            s += arrow(985, y + 31, 1140, 414)
    s += rect(1140, 345, 340, 138, PALE)
    s += lines(1310, 392, ['Weighted sum', '→ output vector y'], 29, 47, weight=600, anchor='middle')
    s += text(75, 660, 'Top-2 example:  y = 0.75 E0(x) + 0.25 E2(x)', 34, 600, BLUE)
    s += text(75, 710, 'With P parameters per expert: 4P expert parameters stored, 2P active for this token.', 27)
    s += takeaway('Only selected experts compute this token; all experts still have parameters.',
                   'Here router weights sum to 1. Routing and normalization details depend on the model.')
    slides.append(slide('MoE: choose a few expert MLPs for each token', s,
        'An expert is a learned feed-forward network with its own weight matrices. The router reads a token hidden vector and selects a small subset of experts. The selected experts each map that hidden vector to an output vector of the same width, which is weighted and summed. This slide uses a four-expert, top-2 teaching model and weights normalized over the selected experts. It does not prescribe a universal gate normalization: models can use different scoring, scaling, shared-expert, and routing designs. The attention sublayer is omitted here, not removed from the Transformer. Sparse activation means few expert MLPs run per token; it does not mean the unselected experts have no weights or need no storage. Total parameters matter for storage while active parameters help describe arithmetic per token. The specific values and original schematic are illustrative, not a measurement.',
        [MIXTRAL], section='Expert parallelism'))

    s = text(75, 205, 'The same MoE layer: four experts, two GPUs. Each expert contains several weight matrices.', 29)
    s += text(75, 273, 'Expert parallelism (EP)', 32, 700, BLUE)
    s += text(855, 273, 'Tensor parallelism within experts', 32, 700, BLUE)
    s += line(795, 245, 795, 697)
    for x, gpu in [(85, 0), (425, 1)]:
        s += rect(x, 315, 290, 263, 'white')
        s += text(x + 145, 358, f'GPU {gpu}', 29, 700, anchor='middle')
        for j in range(2):
            e = 2 * gpu + j
            s += label_box(x + 20, 391 + j * 89, 250, 65, f'Complete E{e}', PALE, 28)
    for x, gpu, shard in [(855, 0, 'a'), (1195, 1, 'b')]:
        s += rect(x, 315, 290, 263, 'white')
        s += text(x + 145, 358, f'GPU {gpu}', 29, 700, anchor='middle')
        for e in range(4):
            s += label_box(x + 20, 380 + e * 47, 250, 40, f'E{e}: shard {shard}', PALE, 26)
    s += lines(85, 632, ['Place different experts on different GPUs.', 'Send tokens to their selected experts.'], 27, 45)
    s += lines(855, 632, ['Split each expert’s matrices across GPUs.', 'GPUs cooperate to compute an expert.'], 27, 45)
    s += takeaway('EP partitions the set of experts; TP partitions computation inside an expert.',
                   'They can be combined. This diagram isolates each strategy to show the difference.')
    slides.append(slide('Expert parallelism places different experts on different GPUs', s,
        'Expert ownership is the new partition axis. In pure two-way EP, GPU 0 owns E0 and E1 and GPU 1 owns E2 and E3. A selected expert can run locally on its owner because its complete matrices are present. In expert TP, each GPU owns a shard of every expert’s matrices; computing a selected expert requires the tensor-parallel collaboration introduced earlier. A real system can combine expert TP with EP, so these are illustrative alternatives rather than mutually exclusive deployment modes. Unselected weights remain resident in this example; there is no weight offload or on-demand model loading. Placement is repeated independently for each MoE layer.',
        [EP, ('DeepSpeed MoE', 'https://www.deepspeed.ai/tutorials/mixture-of-experts/')], section='Expert parallelism'))

    s = text(75, 205, 'One MoE layer: A–D are current token vectors from four decode requests. Each selects two experts.', 28)
    s += label_box(75, 245, 690, 63, 'GPU 0 owns E0, E1 and initially holds A, B', size=27)
    s += label_box(835, 245, 690, 63, 'GPU 1 owns E2, E3 and initially holds C, D', size=27)
    s += table(75, 337, 1450,
               ['Token', 'Original owner', 'Selected experts and weights', 'Remote destination'],
               [['A', 'GPU 0', '0.75 × E0 + 0.25 × E2', 'E2 on GPU 1'],
                ['B', 'GPU 0', '0.60 × E1 + 0.40 × E3', 'E3 on GPU 1'],
                ['C', 'GPU 1', '0.20 × E0 + 0.80 × E2', 'E0 on GPU 0'],
                ['D', 'GPU 1', '0.30 × E1 + 0.70 × E3', 'E1 on GPU 0']],
               [.10, .20, .41, .29], row_h=64, size=27)
    s += text(75, 705, 'Every expert receives two tokens. These routing choices are a balanced toy example.', 28)
    s += takeaway('The router chooses experts per token; the system locates those experts on GPUs.')
    slides.append(slide('Follow four tokens through a two-GPU MoE layer', s,
        'Use exactly this routing table on the next slide. A and B start on GPU 0, C and D start on GPU 1. E0 and E1 live on GPU 0, E2 and E3 live on GPU 1. Token A therefore runs E0 locally and sends its hidden vector to E2 remotely. Repeat the reasoning for each row. Expert batches are E0:[A,C], E1:[B,D], E2:[A,C], E3:[B,D]. There are eight token-expert assignments but only four distinct input tokens. With this placement every token has one local expert and one remote expert. Router weights are supplied, sum to one within each row, and belong to the token, not the GPU. All weights, placement, and token identifiers are original teaching examples. A–D need not be consecutive positions in a shared sequence.',
        [DISPATCH], section='Expert parallelism'))

    s = steps('ep-route', [_routing_scene(i) for i in range(3)])
    s += takeaway('Move token activations to expert weights, then return the expert outputs.',
                   'A common dispatcher uses an all-to-all exchange in each direction; implementations vary.')
    slides.append(slide('Dispatch → expert computation → return and combine', s,
        'Interactive sequence using the previous routing table. Step 1: GPU 0 sends A and B to GPU 1 while GPU 1 sends C and D to GPU 0. Local routes do not require crossing the network. More generally each source packs separate activation subsets for each destination, and counts can differ; this is the all-to-all communication pattern. It is not an AllReduce, because the destinations need different token subsets rather than a common sum. Step 2: each expert processes its assigned tokens, normally grouped to make its MLP matrix multiplications more efficient. Step 3: expert outputs return to the original token owners and are unpermuted into token order. Token A has E0(A)=[2,4] and E2(A)=[10,0], chosen illustrative expert outputs. Its given gate weights produce [0.75*2+0.25*10,0.75*4+0.25*0]=[4,3]. The first component is 4, not 12: combining is a router-weighted vector sum. The other tokens are combined with their own weights. The explicit sum illustrates semantics; implementations may fuse weighting with expert kernels or communication. The Megatron all-to-all dispatcher is one concrete design; all-gather/reduce-scatter and specialized fused backends also exist. No model weights or KV cache are migrated in this example.',
        [DISPATCH], section='Expert parallelism'))

    s = text(75, 205, 'Equal expert sizes do not imply equal work: the router controls the number of token assignments.', 28)
    s += text(75, 275, 'Previous step: balanced routes', 31, 700, BLUE)
    s += text(845, 275, 'Another step: skewed routes', 31, 700, BLUE)
    s += line(790, 253, 790, 665)
    for x, counts in [(75, [4, 4]), (845, [6, 2])]:
        for j, n in enumerate(counts):
            y = 325 + 98*j
            s += text(x, y+38, f'GPU {j}', 27, 600)
            s += rect(x+130, y, n*58, 54, PALE, BLUE)
            s += text(x+146+n*58, y+37, f'{n} assignments', 26)
    s += lines(75, 560, ['E0: 2     E1: 2     E2: 2     E3: 2', '4 tokens × top-2 = 8 assignments'], 28, 49)
    s += lines(845, 560, ['A, B → E0 and E1', 'C → E0 and E2;  D → E1 and E3', 'E0: 3     E1: 3     E2: 1     E3: 1'], 27, 46)
    s += text(75, 706, 'Measure expert loads and communication; remapping or replicating hot experts can help.', 28)
    s += takeaway('The busiest expert owner can become a straggler while other GPUs wait.',
                   'Assignment counts illustrate load, not exact latency. Extra expert replicas consume memory.')
    slides.append(slide('Routing imbalance can leave some GPUs waiting', s,
        'Expert count is a capacity allocation, while routed token count is a workload allocation. The balanced case is the previous table: each expert has two token assignments, giving four per GPU. In the skewed example A and B both select E0/E1, C selects E0/E2, D selects E1/E3. The counts are [3,3,1,1], hence GPU totals [6,2], with eight assignments in both examples. Assuming equally costly experts makes the imbalance visible, but runtime is not simply proportional to counts: GEMM shapes, batching, overlap, kernels, and network costs matter. vLLM monitors load and can adjust expert placement and use redundant replicas. Replication uses extra weight memory and placement changes have cost. Training can use load-balancing objectives; inference cannot simply reroute every token to an arbitrary different expert without changing the model result. Capacity limits are an implementation/model policy: dropping, buffering, padding, or dropless dispatch must be specified, not silently assumed. Very small token groups also give small expert GEMMs and may expose communication latency even when load is balanced.',
        [EP], section='Expert parallelism'))

    s = text(75, 204, 'Four decode requests: A, B on GPU 0; C, D on GPU 1. Attention weights are replicated.', 29)
    for x, gpu, tokens, experts in [(85,0,'A, B','E0 + E1'),(865,1,'C, D','E2 + E3')]:
        s += rect(x, 241, 650, 442, 'white')
        s += text(x+25, 282, f'GPU {gpu}: owns requests {tokens}', 29, 700, BLUE)
        s += label_box(x+25, 307, 360, 60, 'Attention weights: copy', PALE, 26)
        s += label_box(x+400, 307, 225, 60, 'Request KV', '#f7f7f7', 26)
        s += label_box(x+115, 480, 420, 58, f'Local expert weights: {experts}', PALE, 27)
        s += text(x+325, 667, f'Combine outputs for {tokens}', 27, anchor='middle')
        s += arrow(x+325, 367, x+325, 400)
        s += arrow(x+325, 450, x+325, 480)
        s += arrow(x+325, 538, x+325, 568)
        s += arrow(x+325, 618, x+325, 638)
    s += label_box(285, 400, 1030, 50, 'Dispatch: route token activations to expert owners across GPUs', 'white', 27)
    s += label_box(285, 568, 1030, 50, 'Return: send expert outputs back to the original request workers', 'white', 27)
    s += text(75, 719, 'KV stays with its request worker here; expert routing exchanges activations, not KV history.', 27)
    s += takeaway('These workers are independent in attention, but coordinate at the MoE layers.',
                   'Scope: DP attention + EP experts, with no attention TP or context sharding in this example.')
    slides.append(slide('Different layers can use different parallelism strategies', s,
        'This connects EP to the earlier distinction between request replicas and cooperating GPUs. In this particular deployment, attention weights are replicated across two workers, each worker attends over only its own requests, and each maintains independent request KV storage. The expert weights are distributed across both GPUs, so every MoE layer can exchange token activations between those workers. The entire models are therefore not independent dense serving replicas: even a worker with no local requests may need to participate in the coordinated expert computation. The data-parallel label describes attention ownership in this example; it does not imply gradient synchronization during inference. Router and other small shared weights can be replicated; they are omitted visually. TP or CP can further shard attention and KV, but neither is enabled in this schematic. This is supported by vLLM’s DP-attention plus EP-expert deployment description, rather than a universal claim about every MoE inference runtime.',
        [EP, DP], section='Expert parallelism'))
    return slides
