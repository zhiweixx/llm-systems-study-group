"""Plot YOUR measured results.csv; no demonstration or synthetic timings included."""

import argparse
import csv
import json
import math
from pathlib import Path


def load_rows(path):
    rows = []
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "ok":
                continue
            values = {key: float(row[key]) for key in
                      ("batch_size", "prefill_ms_p50", "decode_step_ms_p50",
                       "per_user_tokens_s", "aggregate_output_tokens_s", "peak_allocated_gb")}
            if not all(math.isfinite(value) and value > 0 for value in values.values()):
                raise ValueError("Plot requires finite, positive measured values")
            rows.append(values)
    if not rows:
        raise ValueError("No successful measurements in this CSV")
    return sorted(rows, key=lambda row: row["batch_size"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("inference-batch-sweep"), help="Output filename stem")
    args = parser.parse_args()
    rows = load_rows(args.csv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metadata_file = args.csv.with_suffix(".json")
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.titlecolor": "#245875"})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    batch = [row["batch_size"] for row in rows]
    blue = "#245875"
    for ax, metric, label, title in (
        (axes[0, 0], "prefill_ms_p50", "Prefill + first argmax (ms)", "First-token compute time"),
        (axes[0, 1], "decode_step_ms_p50", "Average decode-step duration (ms)", "Per-user generation latency"),
        (axes[1, 1], "peak_allocated_gb", "Peak tensor allocation (GB)", "Memory required by the batch"),
    ):
        ax.plot(batch, [row[metric] for row in rows], "o-", color=blue)
        ax.set(xlabel="Batch size", ylabel=label, title=title)
        ax.set_xscale("log", base=2)
        ax.set_xticks(batch, labels=[str(int(value)) for value in batch])
        ax.grid(alpha=0.18)
    ax = axes[1, 0]
    ax.scatter([row["per_user_tokens_s"] for row in rows],
               [row["aggregate_output_tokens_s"] for row in rows], color=blue, s=40)
    for row in rows:
        ax.annotate(f"B={int(row['batch_size'])}",
                    (row["per_user_tokens_s"], row["aggregate_output_tokens_s"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Per-user output tokens/s (steady decode)",
           ylabel="Aggregate output tokens/s (steady decode)", title="What does a larger batch trade off?")
    ax.grid(alpha=0.18)
    gpu = metadata.get("gpu", "GPU metadata unavailable")
    precision = metadata.get("precision", "unknown precision").upper()
    fig.suptitle(f"Measured batch sweep — {gpu}, {precision}", fontsize=16, color=blue)
    if "parameter_count" in metadata:
        fig.text(0.5, 0.925,
                 f"{metadata['parameter_count'] / 1e6:.1f}M parameters | "
                 f"prompt {metadata['prompt_tokens']} | output {metadata['output_tokens']} | "
                 f"{metadata['trials']} trials | PyTorch {metadata['pytorch']}",
                 ha="center", fontsize=10, color="#555555")
    fig.text(0.5, 0.025,
             "Random-weight decoder; fixed lengths. Medians across trials. Decode = whole-loop time / steps.\n"
             "Local synchronized wall-clock timing, not production serving latency or an H100 performance claim.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        output = args.output.with_suffix(suffix)
        fig.savefig(output, dpi=180, bbox_inches="tight")
        print(output)


if __name__ == "__main__":
    main()
