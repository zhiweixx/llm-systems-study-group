"""Editable memory pyramid and a two-GPU bandwidth reference table."""
from .common import *

SOURCES = [
    ('H100 specs', 'https://www.nvidia.com/en-us/data-center/h100/'),
    ('DGX B200', 'https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html'),
    ('B200 HBM', 'https://www.nvidia.com/en-us/data-center/dgx-b200/'),
    ('DGX Hopper', 'https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html'),
    ('FA3 Table 1', 'https://tridao.me/publications/flash3/flash3.pdf'),
    ('FA4', 'https://tridao.me/blog/2026/flash4/'),
    ('B200 L2 test', 'https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut'),
]


def get_slide():
    body = text(75, 196, 'Memory inside one GPU', 30, 700, color=BLUE)
    body += text(840, 196, 'Bandwidth: H100 SXM and DGX B200', 30, 700, color=BLUE)

    # A schematic pyramid like the user's reference; width does not encode data.
    levels = [
        ('Registers', 'Per-thread working values', '#ffffff'),
        ('L1 + shared memory', 'Local to each SM (SRAM)', '#f3f6f8'),
        ('L2 cache', 'Shared across SMs (SRAM)', '#e6edf2'),
        ('HBM', 'GPU main memory (DRAM)', '#d8e4ed'),
    ]
    for i, (name, detail, fill) in enumerate(levels):
        top = 240 + i*110
        left, right = 260-i*45, 580+i*45
        body += (
            f'<polygon points="{left},{top} {right},{top} '
            f'{right+45},{top+110} {left-45},{top+110}" '
            f'fill="{fill}" stroke="{MUTED}" stroke-width="1.5"/>'
        )
        body += text(420, top+45, name, 30, 700, anchor='middle')
        body += text(420, top+82, detail, 25, anchor='middle')
    body += text(420, 720, 'Hierarchy schematic; widths are not to scale.', 23, color=MUTED, anchor='middle')

    # Scope stays next to the resource; values use familiar GB/s and TB/s units.
    rows = [
        ('Shared memory*', 'All SMs summed', '≈31 TB/s', '≈35–38 TB/s'),
        ('L2 cache†', 'Across the GPU', '≈12 TB/s', '16.8–21 TB/s'),
        ('HBM', 'Per GPU', '3.35 TB/s', 'Up to 8 TB/s'),
        ('NVLink', 'Per GPU, one-way', '450 GB/s', '900 GB/s'),
        ('InfiniBand', 'Per 400 Gb/s NIC, one-way', '50 GB/s', '50 GB/s'),
    ]
    body += rect(840, 230, 685, 58, PALE, 'none')
    body += text(855, 269, 'Resource', 28, 700)
    body += text(1210, 269, 'H100', 28, 700, color=BLUE, anchor='middle')
    body += text(1410, 269, 'B200', 28, 700, color=BLUE, anchor='middle')
    body += line(840, 288, 1525, 288)
    for i, (name, scope, h100, b200) in enumerate(rows):
        top = 288+i*82
        body += text(855, top+34, name, 27, 700)
        body += text(855, top+63, scope, 21, color=MUTED)
        body += text(1210, top+48, h100, 27, anchor='middle')
        body += text(1410, top+48, b200, 27, anchor='middle')
        body += line(840, top+82, 1525, top+82, LINE if i==2 else '#c7cdd1', 2.5 if i==2 else 1.5)

    body += line(75, 748, 1525, 748)
    body += text(75, 784, 'NVLink connects GPUs within a node; InfiniBand connects nodes.', 29, 700, color=BLUE)
    body += text(75, 820, '* Shared-memory estimates sum all SMs. † L2 values are published references / measurements.', 23, color=MUTED)

    notes = (
        'The pyramid describes storage inside one GPU. Registers hold each thread’s working values; '
        'L1 and shared memory are local to each SM; L2 is shared across SMs; and HBM is the GPU’s '
        'main memory, implemented with stacked DRAM. Pyramid widths are schematic and do not encode '
        'capacity or bandwidth. NVLink and InfiniBand are communication fabrics, so they appear in '
        'the table rather than as extra cache levels. The reference uses H100 SXM and an eight-GPU '
        'DGX B200. NVLink can also span servers in other systems; the within-node label applies to '
        'these reference configurations. L1 and shared memory use unified on-chip storage resources '
        'with different access semantics. The table reports SMEM bandwidth, not a sum of L1 and SMEM. '
        'Registers and Blackwell tensor memory are not assigned an unsupported bandwidth number. '
        'All units are decimal: 1 TB/s = 1,000 GB/s; 400 Gb/s = 50 GB/s. '
        'The 450/900 GB/s NVLink values are aggregate one-way bandwidth per H100/B200 GPU, obtained '
        'from NVIDIA’s 900/1,800 GB/s bidirectional specifications. They are not per-link rates or '
        'independent simultaneous bandwidth to every peer. HBM is 3.35/up to 8 TB/s per GPU; '
        'the official DGX B200 page specifies 64 TB/s summed across eight GPUs. HBM is not a '
        'full-duplex network interface whose quoted rate should be doubled. '
        'Both DGX reference configurations have eight 400 Gb/s ConnectX-7 compute-network adapters. '
        'The table uses 50 GB/s per adapter per direction, not the 400 GB/s node aggregate. '
        'InfiniBand speed depends on the server networking configuration, not only the GPU model. '
        'Application and collective bandwidth depend on topology, message size and contention. '
        'The asterisk and dagger distinguish SRAM approximations from product specifications. '
        'FlashAttention-3 Table 1 uses 12 TB/s as its H100 L2 reference and estimates shared memory '
        'as 128 bytes/cycle/SM × 132 SMs × 1.83 GHz = 30.92 TB/s. '
        'The B200 SMEM estimate uses FlashAttention-4’s 128 bytes/cycle/SM and 148 SMs with assumed '
        'clocks of 1.85–2.0 GHz, giving 35.05–37.89 TB/s. This is a clock-based estimate, not a '
        'benchmark confidence interval. All-SM aggregate means SMs concurrently access their own '
        'local storage. A single H100 SM at 1.83 GHz contributes approximately 234 GB/s. '
        'Chips and Cheese measured B200 L2 at 21 TB/s for local-partition working sets and '
        '16.8 TB/s when accesses cross partitions. Cache throughput varies with clock, access pattern '
        'and benchmark; these cache entries are not a controlled generational comparison or '
        'guaranteed peaks. Sources are linked below. This table reproduces and derives cited values; '
        'we ran no GPU benchmark.'
    )
    return slide('GPU memory hierarchy and bandwidth', body, notes, SOURCES, 'Hardware bandwidth')
