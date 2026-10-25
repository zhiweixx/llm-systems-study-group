# Week 6 figure sources

The HTML generator embeds these assets as data URIs. The presentation needs no external fonts, scripts, or images when used offline. White backgrounds, blue titles, divider rules, navigation and notes follow the existing course template.

## Source assets

- `aloha-frame0.png` and `aloha-frame100.png`: unmodified 640×480 decoded RGB frames at 0.00 and 2.00 seconds from episode 0 of [LeRobot's ALOHA simulated human insertion dataset](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human), `videos/observation.images.top/chunk-000/file-000.mp4`. Metadata declares 50 Hz. The frames depict a simulated scene, not physical robot footage. Numeric state/action examples come from rows 0–1 of the same dataset; timestamps are nominal frame-index/fps. The original action interface is documented in [ACT's simulation environment](https://github.com/tonyzhaozh/act/blob/main/sim_env.py).
- `pi05-fig7.png`: crop of Figure 7(b), PDF page 8, of [*π0.5: a Vision-Language-Action Model with Open-World Generalization*](https://www.physicalintelligence.company/download/pi05.pdf). The crop retains task labels, axes, home/mock groups, and error bars. Metric: average task progress, not binary success; ten trials per task/environment.
- `pi05-fig13.png`: crop of Figure 13, PDF page 11, of the same paper, preserving axes, method labels, p-values, and error bars. It compares high-level-inference and training-data ablations. The figures are presented for scholarly explanation with attribution to their authors; no ownership of source artwork is claimed.

Paper figure crop bounds in PDF points (left, bottom, right, top): Figure 7(b) `(346, 420, 558, 557)`; Figure 13 `(48, 544, 300, 737)`. Crops were rendered with Poppler at 1,800 pixels on the long edge. No plotted values or labels were altered.

## Original figures

Other figures are editable SVG in `scripts/week6/`. They are authored teaching illustrations unless explicitly labeled as a plot of reported results. The RoboDojo category bars transcribe the single-seed summary in its cited primary paper. The final operating curves are schematic and contain no measured data. Interactive examples expose chunk execution and two π0.5 evaluation figures.

Primary citations appear in each slide's footer and in `week-6-speaker-notes.md`. Recent benchmark numbers are kept within their own task suites and protocol definitions; no cross-benchmark ranking is intended.
