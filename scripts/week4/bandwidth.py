"""Source-labelled bandwidth comparison; original editable logarithmic chart."""
from math import log10, sqrt
from .common import *

SOURCES = [
    ('H100 specs', 'https://www.nvidia.com/en-us/data-center/h100/'),
    ('H200 specs', 'https://www.nvidia.com/en-us/data-center/h200/'),
    ('DGX B200', 'https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html'),
    ('B200 HBM', 'https://www.nvidia.com/en-us/data-center/dgx-b200/'),
    ('DGX Hopper', 'https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html'),
    ('FA3 Table 1', 'https://tridao.me/publications/flash3/flash3.pdf'),
    ('FA4', 'https://tridao.me/blog/2026/flash4/'),
    ('B200 L2 test', 'https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut'),
    ('L1 test', 'https://github.com/RRZE-HPC/gpu-benches'),
]


def get_slide():
    # Values are in decimal TB/s. Keep ranges and missing values explicit.
    rows = [
        ('Shared memory (SRAM)', 'All SMs summed',
         [(31, '≈31', True), (31, '≈31*', True), ((35, 38), '≈35–38*', True)]),
        ('L2 cache (SRAM)', 'Across the GPU',
         [(12, '≈12 (paper)', True), (None, 'H200: not verified', True),
          ((16.8, 21), '16.8–21 (test)', True)]),
        ('HBM', 'Per GPU',
         [(3.35, '3.35', False), (4.8, '4.8', False), (8, 'up to 8', False)]),
        ('NVLink', 'Per GPU, one-way',
         [(.45, '0.45', False), (.45, '0.45', False), (.9, '0.90', False)]),
        ('InfiniBand', 'Per 400 Gb/s NIC, one-way',
         [(.05, '0.05', False), (.05, '0.05', False), (.05, '0.05', False)]),
    ]
    colors = [BLUE, TEAL, WARM]

    def x(v):
        return 445 + (log10(v) - log10(.03)) / (log10(50) - log10(.03)) * 880

    def marker(px, py, series, hollow=False):
        color = colors[series]
        fill = 'white' if hollow else color
        attrs = f'fill="{fill}" stroke="{color}" stroke-width="2.5"'
        if series == 0:
            return f'<circle cx="{px}" cy="{py}" r="6" {attrs}/>'
        if series == 1:
            return f'<rect x="{px-6}" y="{py-6}" width="12" height="12" {attrs}/>'
        return f'<path d="M {px} {py-7} L {px+7} {py+6} L {px-7} {py+6} Z" {attrs}/>'

    body = text(75, 183, 'H100 / H200 SXM and DGX B200. Link bandwidths use one direction.', 29)
    for i, (name, xx) in enumerate([('H100', 87), ('H200', 263), ('B200', 439)]):
        body += marker(xx, 218, i) + text(xx+19, 226, name, 26, 600, color=colors[i])
    body += marker(684, 218, 0, True)
    body += text(703, 226, 'Open markers: SRAM references / estimates', 25, color=MUTED)

    for v in [.05, .1, 1, 10, 50]:
        body += line(x(v), 244, x(v), 672, '#e0e5e8', 1)

    for r, (label, scope, values) in enumerate(rows):
        yy = 270 + r*88
        body += text(75, yy+3, label, 27, 700)
        body += text(75, yy+32, scope, 22, color=MUTED)
        for i, (value, label, hollow) in enumerate(values):
            py = yy + (i-1)*23
            if value is None:
                body += text(467, py+8, label, 23, color=colors[i])
                continue
            high = value[-1] if isinstance(value, tuple) else value
            if isinstance(value, tuple):
                low, high = value
                px = x(sqrt(low*high))
                body += line(x(low), py, x(high), py, colors[i], 3)
                body += line(x(low), py-6, x(low), py+6, colors[i], 2)
                body += line(x(high), py-6, x(high), py+6, colors[i], 2)
            else:
                px = x(value)
            body += marker(px, py, i, hollow)
            body += text(x(high)+14, py+8, label, 23, 600, color=colors[i])

    body += line(445, 672, 1325, 672, MUTED)
    for v, label in [(.05, '0.05'), (.1, '0.1'), (1, '1'), (10, '10'), (50, '50')]:
        body += line(x(v), 672, x(v), 679, MUTED)
        body += text(x(v), 704, label, 23, anchor='middle')
    body += text(885, 735, 'Bandwidth (TB/s, log scale)     0.05 TB/s = 50 GB/s', 24, anchor='middle')
    body += line(75, 753, 1525, 753)
    body += text(75, 785, '* Architectural estimates. SRAM throughput depends on clock and access pattern.', 24, color=MUTED)
    body += text(75, 819, 'L1: H100 ≈31 TB/s estimated across all SMs; no matched H200/B200 data shown.', 24, color=MUTED)

    notes = (
        'This chart compares bandwidth scales, not latency. It is an original chart of published specifications, '
        'published microbenchmarks, and explicitly stated architectural estimates; we ran no GPU benchmark. '
        'The horizontal axis is logarithmic and all plotted values are decimal TB/s (1 TB/s = 1,000 GB/s). '
        'H100 and H200 mean SXM, and the networking reference is an eight-GPU DGX configuration. '
        'The entire set of NVLink interfaces on one GPU supplies 900/900/1,800 GB/s bidirectionally for '
        'H100/H200/B200. The plotted one-way rates are therefore 450/450/900 GB/s, not per-link rates '
        'or independent simultaneous bandwidth to every peer. HBM is 3.35/4.8/up to 8 TB/s per GPU. '
        'Do not double HBM bandwidth as if it were a full-duplex network interface. '
        'InfiniBand is a server configuration rather than a fixed GPU property. These DGX systems have '
        'eight 400 Gb/s ConnectX-7 compute-network adapters: 400/8 = 50 GB/s per adapter per direction, '
        'or 400 GB/s per node per direction when all eight adapters are used. Only the per-adapter rate is '
        'plotted. Usable application and collective bandwidth will be lower and topology-dependent. '
        'The hollow SRAM markers are not official bandwidth guarantees or a controlled generational benchmark. '
        'FlashAttention-3 Table 1 uses 12 TB/s for H100 L2 and estimates SMEM bandwidth as '
        '128 bytes/cycle/SM × 132 SMs × 1.83 GHz = 30.92 TB/s. Other L2 microbenchmarks obtain different '
        'rates because clock, working-set size, read/write mix, partition locality, and kernel design differ. '
        'We leave H200 L2 unplotted because an independent comparable value was not verified. '
        'The H200 SMEM marker uses the same Hopper throughput, 132 SMs, and an assumed 1.83 GHz; '
        'it is an architectural estimate, not a measurement. The B200 SMEM range uses the FlashAttention-4 '
        'values of 128 bytes/cycle/SM and 148 SMs with assumed clocks of 1.85–2.0 GHz, yielding '
        '35.05–37.89 TB/s. This range reflects clock assumptions, not a statistical confidence interval. '
        'Chips and Cheese measured B200 L2 at 21 TB/s for local-partition working sets and 16.8 TB/s when '
        'accesses cross partitions. That plotted range represents these access patterns. '
        'RRZE-HPC reports H100 L1 hits approaching 128 bytes/cycle/SM. Applying the SXM SM count and '
        '1.83 GHz gives the approximate 31 TB/s L1 aggregate mentioned below the chart; the microbenchmark '
        'itself used a PCIe H100, so this is a normalized SXM estimate, not its measured total. '
        'No matched H200/B200 L1 result is plotted. L1 and SMEM share on-chip storage resources but have '
        'different access semantics; their rates must not be added. All-SM aggregate means independent '
        'SMs accessing their own local storage concurrently. One H100 SM at 1.83 GHz contributes '
        'only about 234 GB/s, not 31 TB/s by itself. Register and B200 tensor-memory datapaths are '
        'not represented by the SMEM numbers. Use this figure to motivate data reuse within a GPU and '
        'to understand why moving tensors between GPUs or nodes can dominate serving costs.'
    )
    return slide('Bandwidth: on-chip memory, HBM and GPU links', body, notes, SOURCES, 'Hardware bandwidth')
