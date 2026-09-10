"""Reproduce the A100-normalized GPU comparison with Matplotlib 3.10.8.

Run: python plot.py
Raw measurements and their qualifications are in data.csv and README.md.
"""

from pathlib import Path
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parent
SERIES = [
    ("bf16_fp16_dense_tflops", "BF16 / FP16 compute", "#245675", "o", 3.2),
    ("hbm_bandwidth_tb_s", "HBM bandwidth", "#AD7029", "s", 2.3),
    ("hbm_capacity_gb", "HBM capacity", "#7964A1", "D", 2.3),
    ("l2_capacity_mb", "L2 cache", "#37816D", "^", 2.3),
    ("l1_shared_pool_kb_per_sm", "L1 / shared-memory pool", "#697783", "v", 2.3),
]


def read_data():
    with (ROOT / "data.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["gpu"] for r in rows] == ["A100", "H100", "H200", "B100", "B200"]
    numeric = {
        key: [float(r[key]) if r[key] else math.nan for r in rows]
        for key, *_ in SERIES
    }
    relative = {key: [v / vals[0] for v in vals] for key, vals in numeric.items()}
    assert all(vals[0] == 1.0 for vals in relative.values())
    assert math.isnan(relative["l2_capacity_mb"][3])
    return rows, numeric, relative


def main():
    rows, numeric, relative = read_data()
    with (ROOT / "normalized.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu"] + [key + "_relative_to_a100" for key, *_ in SERIES])
        for i, row in enumerate(rows):
            writer.writerow([row["gpu"]] + [
                "" if math.isnan(relative[key][i]) else f"{relative[key][i]:.8f}"
                for key, *_ in SERIES
            ])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.edgecolor": "#9BA3A8",
        "axes.labelcolor": "#242C32",
        "text.color": "#242C32",
        "xtick.color": "#414A50",
        "ytick.color": "#414A50",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(14, 8.2), facecolor="white")
    ax = fig.add_axes([0.083, 0.238, 0.633, 0.596])

    fig.text(0.083, 0.94, "GPU compute and memory scaling", fontsize=23,
             fontweight="bold", color="#245675")
    fig.text(0.083, 0.896,
             "Each series is normalized to its A100 value  ·  A100 80GB SXM = 1.0",
             fontsize=12.5, color="#56626B")
    fig.add_artist(plt.Line2D([0.083, 0.95], [0.868, 0.868],
                            transform=fig.transFigure, color="#BDCBD4", lw=1))

    x = list(range(len(rows)))
    ax.set_xlim(-0.12, 4.18)
    ax.set_ylim(0.65, 7.8)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(axis="y", color="#E5E9EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(1, color="#9CA7AE", lw=1.1, linestyle=(0, (4, 4)), zorder=1)
    ax.set_ylabel("Relative to A100 (×)", fontsize=13, labelpad=11)
    ax.set_xticks(x, ["A100", "H100", "H200", "B100*", "B200"])
    ax.tick_params(axis="x", length=0, pad=11, labelsize=13)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)

    # A missing B100 L2 observation deliberately breaks the line: no interpolation.
    for key, label, color, marker, width in reversed(SERIES):
        vals = relative[key]
        ax.plot(x, vals, color=color, marker=marker, markersize=7,
                markeredgewidth=1.15, linewidth=width, solid_capstyle="round",
                zorder=5, label=label)
        # Hollow B100 markers distinguish preliminary/architecture-level values.
        if not math.isnan(vals[3]):
            ax.plot([3], [vals[3]], linestyle="none", marker=marker,
                    markersize=7.4, markeredgewidth=1.6, markeredgecolor=color,
                    markerfacecolor="white", zorder=8)
        ax.annotate(f"{label}\n{vals[-1]:.2f}×", xy=(4, vals[-1]),
                    xytext=(22, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=12,
                    fontweight="bold" if key == "bf16_fp16_dense_tflops" else "normal",
                    color=color, linespacing=1.35, annotation_clip=False)

    ax.annotate("B100 L2: not verified", xy=(3, 3.15),
                ha="center", va="center", fontsize=10.5, color="#63736D",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})

    fig.add_artist(plt.Line2D([0.083, 0.95], [0.166, 0.166],
                            transform=fig.transFigure, color="#D3DADF", lw=0.8))
    notes = [
        "Compute: dense BF16 / FP16 Tensor Core peak. Capacity and bandwidth are separate series.",
        "HBM and L2: per GPU. L1: combined L1 / shared-memory / texture pool per SM, not dedicated L1 cache.",
        "* B100: preliminary specs; L1 uses the Blackwell architecture value. Missing L2 data are not interpolated.",
        "Sources: NVIDIA product specifications, architecture/tuning guides and HGX reference architecture; see accompanying data and notes.",
    ]
    for i, note in enumerate(notes):
        fig.text(0.083, 0.129 - i * 0.03, note, fontsize=10.2, color="#58646D")

    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(ROOT / f"gpu-relative-growth.{suffix}", dpi=300,
                    metadata={"Creator": "Matplotlib; data and sources in accompanying README.md"})
    plt.close(fig)
    print("A100-normalized B200 endpoints:")
    for key, label, *_ in SERIES:
        print(f"  {label}: {relative[key][-1]:.8f}x")
    print("Created PNG, SVG, PDF and normalized.csv in", ROOT)


if __name__ == "__main__":
    main()
