"""Practical pipeline-parallel lessons, with an explicit toy timing model."""
from .common import *

MEG_PIPE = ('Megatron: pipeline model', 'https://arxiv.org/html/2104.04473v5#S2.SS2')
MEG_BATCH = ('Megatron: microbatch trade-off', 'https://arxiv.org/html/2104.04473v5#S3.SS4')
PT_PIPE = ('PyTorch pipelining', 'https://docs.pytorch.org/docs/2.14/distributed.pipelining.html')
DS_PIPE = ('DeepSpeed pipeline', 'https://www.deepspeed.ai/tutorials/pipeline/')


def get_slides():
    s = []
    b = text(75, 185, 'Toy model: balanced stages, a GPipe-style schedule, and no communication cost.', 28)
    b += text(75, 236, 'p = number of pipeline stages; m = microbatches in one training iteration.', 28)
    b += text(75, 302, 'Bubble fraction = (p − 1) / (m + p − 1)', 35, 700, color=BLUE)
    b += text(75, 345, 'Fraction of total GPU-time spent idle while filling and draining the pipeline.', 26, color=MUTED)
    bar_x, bar_w = 400, 915
    for y, p, m in [(391, 4, 16), (474, 8, 16), (557, 8, 64)]:
        idle = (p - 1) / (m + p - 1)
        useful = 1 - idle
        b += text(75, y + 34, f'PP = {p}, m = {m}', 30, 700)
        b += rect(bar_x, y, bar_w * useful, 52, BLUE, 'none')
        b += rect(bar_x + bar_w * useful, y, bar_w * idle, 52, '#e0e5e8', 'none')
        b += text(1350, y + 34, f'{100 * idle:.1f}%', 31, 700, color=BLUE)
    b += text(1365, 369, 'Idle', 25, color=MUTED)
    b += rect(400, 637, 26, 26, BLUE, 'none') + text(442, 659, 'Compute', 26)
    b += rect(630, 637, 26, 26, '#e0e5e8', 'none') + text(672, 659, 'Pipeline bubble', 26)
    b += text(75, 711, 'More stages also reduce work per stage: a higher idle fraction does not imply a longer step.', 27)
    b += takeaway('At fixed m, deeper PP increases the bubble fraction; more microbatches reduce it.',
                   'This is an idealized schedule model. Stage imbalance, transfers, and other schedules change the result.')
    s.append(slide('Deeper pipelines need more microbatches', b,
        'The lesson concerns idle fraction, not an unconditional increase in absolute iteration time. '
        'Let p be the number of pipeline stages and m the number of microbatches per training iteration '
        'for one data-parallel replica. Under a balanced non-interleaved GPipe-style schedule, '
        'with equal stage forward times and equal stage backward times across stages and with communication omitted, '
        'the fill/drain overhead is (p−1) stage forward-plus-backward intervals, compared with m useful intervals. '
        'The bubble fraction of total GPU-time is therefore (p−1)/(m+p−1). '
        'Do not confuse it with overhead divided by useful compute, (p−1)/m; these have different denominators. '
        'The examples are 3/19=15.8%, 7/23=30.4%, and 7/71=9.9%. '
        'Blue and gray bars are normalized ideal compute/idle shares, not measurements. '
        'Increasing PP also partitions the model into smaller stages and can reduce stage time, so the chart does not predict '
        'that the full step must take longer. Interleaved schedules and other pipeline schedules can alter the bubble model; '
        'uneven layers, communication, and launch overhead add real costs. '
        'The fixed-m comparison holds microbatch count constant; the next slide shows how that count is chosen.',
        (MEG_PIPE,), section='Pipeline parallelism'))

    b = text(75, 185, 'Example: global batch = 256 sequences, DP = 4 replicas, PP = 8 stages.', 29)
    b += text(75, 235, 'Each replica processes 64 sequences per optimizer step, split into microbatches.', 28)
    b += text(75, 298, 'm = global batch size / (DP × microbatch size)', 34, 700, color=BLUE)
    b += table(75, 339, 1450,
               ['Sequences per microbatch', 'Microbatches per replica', 'Ideal PP bubble fraction'],
               [['4', '256 / (4 × 4) = 16', '7 / (16 + 7) = 30.4%'],
                ['1', '256 / (4 × 1) = 64', '7 / (64 + 7) = 9.9%']],
               [.35, .34, .31], row_h=69, size=27)
    b += text(75, 593, 'Smaller microbatches reduce bubbles but may make GEMMs less efficient.', 28)
    b += text(75, 638, 'Increasing the microbatch count does not change the configured PP degree.', 28)
    b += line(75, 661, 1525, 661)
    b += text(75, 706, 'Available in Megatron, PyTorch pipelining, and DeepSpeed—not exclusive to Megatron.', 27)
    b += takeaway('Choose microbatch size by balancing kernel efficiency and pipeline bubbles.',
                   'Same ideal timing model as the previous slide. One training step; no change to the global batch.')
    s.append(slide('Microbatch count is not microbatch size', b,
        'A microbatch contains a number of sequences. The microbatch count is how many such chunks pass through '
        'one pipeline replica during the gradient-accumulation window before an optimizer update. '
        'For equal-size batches, global batch size = DP degree × microbatch size × number of microbatches. '
        'PP does not multiply the global batch because each stage processes the same examples at a different group of layers. '
        'With global batch 256 and DP4, each pipeline replica processes 64 sequences: microbatch size 4 gives m=16, '
        'while size 1 gives m=64. At PP8 the ideal bubble fractions are 7/23 and 7/71. '
        'These are sequence counts, not token counts; equal sequence lengths are assumed for the toy timing comparison. '
        'This arithmetic does not assert that the size-1 configuration runs faster. Reducing microbatch size may reduce GEMM '
        'efficiency and increase per-operation overhead, while increasing it may improve kernels but retain more activations. '
        'The batch/PP choice therefore requires measurement. '
        'Microbatches do not allocate additional stages; PP degree is a separate model/deployment configuration. '
        'Megatron is one implementation, but PyTorch distributed.pipelining and DeepSpeed also explicitly support '
        'pipeline training with microbatches. This training formula is not permission to schedule the unknown future tokens '
        'of one autoregressive request as independent microbatches. The following decode-dependency slide explains that distinction.',
        (MEG_BATCH, PT_PIPE, DS_PIPE), section='Pipeline parallelism'))
    return s
