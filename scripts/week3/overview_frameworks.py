"""Framework relationships and a short, source-linked history."""
from .common import *


def framework_history():
    b = text(75, 190, 'Megatron and DeepSpeed build large-model training systems on PyTorch.', 30)
    for x, name, detail, color in (
        (125, 'Megatron-LM / Core', 'Transformer parallelism: TP, PP, SP, CP, EP', BLUE),
        (865, 'DeepSpeed', 'ZeRO state sharding and offload', TEAL),
    ):
        b += rect(x, 220, 610, 128, 'white', color)
        b += text(x + 305, 267, name, 34, 700, color=color, anchor='middle')
        b += text(x + 305, 316, detail, 26, anchor='middle')
        b += arrow(x + 305, 355, x + 305, 402)
        b += text(x + 325, 386, 'built on', 24, color=MUTED)

    b += rect(125, 413, 1350, 114, PALE)
    b += text(153, 458, 'PyTorch', 34, 700, color=BLUE)
    b += text(153, 500, 'Tensors, autograd, distributed communication', 26)
    b += line(810, 433, 810, 508)
    b += text(850, 458, 'Native parallelism', 29, 700, color=BLUE)
    b += text(850, 500, 'DDP, FSDP, TP, PP', 28)

    b += text(75, 576, 'Selected milestones', 25, 700, color=MUTED)
    b += arrow(125, 631, 1475, 631)
    milestones = (
        (200, '2017', ('PyTorch', 'released')),
        (500, '2019', ('Megatron + ZeRO', 'papers')),
        (800, '2020', ('DeepSpeed', 'released')),
        (1100, '2021', ('FairScale', 'FSDP')),
        (1400, '2022', ('PyTorch FSDP', 'prototype')),
    )
    for x, year, label in milestones:
        b += text(x, 611, year, 28, 700, color=BLUE, anchor='middle')
        b += line(x, 623, x, 639, BLUE, 3)
        b += lines(x, 674, label, 25, gap=32, anchor='middle')

    b += takeaway('Parallelism methods have implementations in multiple frameworks.',
                  'For example, Megatron, DeepSpeed, and native PyTorch all support TP.')
    notes = (
        'The boxes show software dependencies, not GPU partitions. Megatron-LM is a Transformer training '
        'project, with reusable parallelism components in Megatron Core. DeepSpeed is a training/runtime '
        'and optimization library. Both use PyTorch tensors, autograd, model definitions, and distributed '
        'facilities. PyTorch also provides its own distributed implementations, so using native FSDP or '
        'TP does not require DeepSpeed or Megatron. The capability labels are examples, not exclusive '
        'or exhaustive lists: DeepSpeed also implements TP and PP, and Megatron implements training-state '
        'sharding, including its own FSDP. Generic DP, TP, PP, CP, and EP are strategies rather than '
        'project-owned features. The name SP needs a specific algorithm: Megatron SP around TP differs '
        'from DeepSpeed Ulysses for long-context attention. '
        'The timeline shows selected events, with spacing chosen for readability. PyTorch became public '
        'in January 2017. The Megatron paper appeared in September 2019 and the ZeRO paper in October 2019. '
        'Megatron popularized an efficient Transformer-specific tensor-parallel scheme; tensor '
        'parallelism itself predates that paper. Microsoft launched DeepSpeed in February 2020, initially '
        'shipping ZeRO-1 rather than all later ZeRO stages. FairScale released FSDP in early 2021, and '
        'PyTorch 1.11 introduced native FSDP as a prototype in March 2022. The PyTorch FSDP history '
        'explicitly credits ZeRO for developing and popularizing sharded data parallelism. FSDP has '
        'its own implementation rather than wrapping DeepSpeed ZeRO-3. Later in 2022, PyTorch 1.12 '
        'promoted FSDP to beta. Megatron and DeepSpeed have also been integrated, for example for '
        'MT-NLG in 2021, but supported combinations depend on the implementation and configuration. '
        'The diagram is an authored synthesis of the linked project documentation and history.'
    )
    sources = (
        ('PyTorch history', 'https://pytorch.org/blog/a-year-in/'),
        ('Megatron', 'https://arxiv.org/abs/1909.08053'),
        ('ZeRO', 'https://arxiv.org/abs/1910.02054'),
        ('DeepSpeed launch', 'https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/'),
        ('FSDP history', 'https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/'),
        ('PyTorch TP', 'https://docs.pytorch.org/docs/stable/distributed.tensor.parallel'),
        ('DeepSpeed TP', 'https://www.deepspeed.ai/tutorials/autotp-training/'),
        ('Megatron today', 'https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html'),
    )
    return slide('PyTorch, Megatron, and DeepSpeed', b, notes, sources,
                 section='Frameworks and history')
