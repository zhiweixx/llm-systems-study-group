"""Render the same comparison as a compact, vector chart for the HTML deck.

Run with Matplotlib 3.10.8: python plot-slide.py
The full standalone figure is reproduced by plot.py.
"""
import math
from pathlib import Path

from plot import read_data, SERIES
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parent
rows, _, relative = read_data()
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 15,
    "svg.fonttype": "none",
    "axes.edgecolor": "#9BA3A8",
    "axes.labelcolor": "#242C32",
    "text.color": "#242C32",
    "xtick.color": "#414A50",
    "ytick.color": "#414A50",
})
fig = plt.figure(figsize=(14.5, 4.7), facecolor="white")
ax = fig.add_axes([0.072, 0.13, 0.646, 0.84])
ax.set_xlim(-0.12, 4.12)
ax.set_ylim(0.65, 7.8)
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.grid(axis="y", color="#E5E9EB", linewidth=0.8)
ax.set_axisbelow(True)
ax.axhline(1, color="#9CA7AE", lw=1, linestyle=(0, (4, 4)), zorder=1)
ax.set_ylabel("Relative to A100 (×)", fontsize=15, labelpad=11)
ax.set_xticks(range(5), ["A100", "H100", "H200", "B100*", "B200"])
ax.tick_params(axis="x", length=0, pad=9, labelsize=16)
ax.tick_params(axis="y", length=0, pad=8, labelsize=14)
ax.spines[["top", "right", "bottom"]].set_visible(False)

for key, label, color, marker, width in reversed(SERIES):
    values = relative[key]
    # NaN at B100 deliberately leaves a gap in L2; no interpolation.
    ax.plot(range(5), values, color=color, marker=marker, markersize=6.5,
            linewidth=width, markeredgewidth=1.1, zorder=5)
    if not math.isnan(values[3]):
        ax.plot([3], [values[3]], linestyle="none", marker=marker,
                markersize=7, markeredgewidth=1.5,
                markeredgecolor=color, markerfacecolor="white", zorder=8)
    label = "L1 / shared pool" if key == "l1_shared_pool_kb_per_sm" else label
    label = "BF16 / FP16 Tensor Core" if key == "bf16_fp16_dense_tflops" else label
    ax.annotate(f"{label}\n{values[-1]:.2f}×", xy=(4, values[-1]),
                xytext=(21, 0), textcoords="offset points", fontsize=15.5,
                ha="left", va="center", linespacing=1.10,
                fontweight="bold" if key == "bf16_fp16_dense_tflops" else "normal",
                color=color, annotation_clip=False)

ax.annotate("B100 L2: unverified", xy=(3, 3.15), ha="center", va="center",
            fontsize=12, color="#63736D",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
fig.savefig(ROOT / "gpu-relative-growth-slide.svg",
            metadata={"Creator": "Matplotlib; values and sources in data.csv and README.md"})
svg = ROOT / "gpu-relative-growth-slide.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
plt.close(fig)
print(ROOT / "gpu-relative-growth-slide.svg")
