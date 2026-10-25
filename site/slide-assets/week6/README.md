# Week 6 figure sources

The HTML generator embeds these assets as data URIs. The presentation needs no external fonts, scripts, or images when used offline. White backgrounds, blue titles, divider rules, navigation and notes follow the existing course template.

## Source assets

- `aloha-frame0.png` and `aloha-frame100.png`: unmodified 640×480 decoded RGB frames at 0.00 and 2.00 seconds from episode 0 of [LeRobot's ALOHA simulated human insertion dataset](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human), `videos/observation.images.top/chunk-000/file-000.mp4`. Metadata declares 50 Hz. The frames depict a simulated scene, not physical robot footage. Numeric state/action examples come from rows 0–1 of the same dataset; timestamps are nominal frame-index/fps. The original action interface is documented in [ACT's simulation environment](https://github.com/tonyzhaozh/act/blob/main/sim_env.py).
- `pi05-fig7.png`: crop of Figure 7(b), PDF page 8, of [*π0.5: a Vision-Language-Action Model with Open-World Generalization*](https://www.physicalintelligence.company/download/pi05.pdf). The crop retains task labels, axes, home/mock groups, and error bars. Metric: average task progress, not binary success; ten trials per task/environment.
- `pi05-fig13.png`: crop of Figure 13, PDF page 11, of the same paper, preserving axes, method labels, p-values, and error bars. It compares high-level-inference and training-data ablations. The figures are presented for scholarly explanation with attribution to their authors; no ownership of source artwork is claimed.

Paper figure crop bounds in PDF points (left, bottom, right, top): Figure 7(b) `(346, 420, 558, 557)`; Figure 13 `(48, 544, 300, 737)`. Crops were rendered with Poppler at 1,800 pixels on the long edge. No plotted values or labels were altered.

## Original figures

Other figures are editable SVG in `scripts/week6/`. They are authored teaching illustrations unless explicitly labeled as a plot of reported results. The RoboDojo category bars transcribe the single-seed summary in its cited primary paper. The final operating curves are schematic and contain no measured data. The expanded deck has interactive explanations of multimodal actions, chunk execution, flow sampling, and simulation timing; the preserved reference deck retains its original interactions.

Primary citations appear in each slide's footer and in `week-6-speaker-notes.md`. Recent benchmark numbers are kept within their own task suites and protocol definitions; no cross-benchmark ranking is intended.

## Expanded visual presentation

The 60-slide main deck uses `visual_foundations.py`, `visual_policies.py`, and `visual_recent.py`. The original 30-slide HTML and notes are preserved byte-for-byte as `week-6-robot-learning-reference.html` and `week-6-reference-notes.md`; the earlier content modules remain available.

- `aloha-episode0.webm`: the same source recording as the still frames above. Frames at nominal source times 0.0, 0.1, …, 9.9 seconds were decoded and encoded as a silent 10 Hz WebM demonstration. This is a sampled playback of a recorded simulation, not a live policy or a control-latency measurement. Browser recording introduces a small playback-duration difference. A source still replaces the video when printing.
- `mit-pick.png`, `mit-pick-sketch.png`, `mit-frames.png`: original illustrations from Russ Tedrake’s [Basic Pick and Place](https://manipulation.mit.edu/pick.html), respectively `figures/pick.png`, `figures/pick_sketch.png`, and `figures/kinematic_frames_iiwa.png`. Used to ground the robot task and coordinate-frame discussion; copyright remains with the source authors.
- `foundations-aloha-teleop.jpg`: original ACT setup photograph from [the ACT paper](https://arxiv.org/html/2304.13705v1/figures/setup.jpg). The displayed viewport selects the leader-arm teleoperation setup.
- `foundations-umi.png`: original hardware illustration from [the UMI paper](https://arxiv.org/html/2402.10329v3/UMI_hardware_v2.png). The displayed viewport selects the handheld gripper and camera. Both source images are stored intact; SVG clipping selects the relevant portions.

Playback, action-chunk, and flow-sampling controls are teaching demonstrations. Schematics are distinguished from recorded robot footage and reported experimental measurements. The main presentation embeds all required media for offline use.

### VLM and policy source photographs

All `pol-pi05-*` files and `pol-vlm-scene.png` are crops of the original 19-page [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf), rendered with Poppler. They retain source attribution on the corresponding pages:

- Figure 2, PDF page 2: `pol-pi05-kitchen.png`, the ordered household-demonstration photographs.
- Figure 4, PDF page 6: `pol-pi05-mobile.png`, `pol-pi05-other-homes.png`, `pol-pi05-lab.png`, `pol-pi05-hl.png`, `pol-pi05-web.png`, and `pol-pi05-verbal.png`, illustrating the source data categories.
- Figure 6, PDF page 8: `pol-pi05-real-rooms.png`, the real-room collage with labels; and `pol-vlm-scene.png`, the blue mock kitchen.
- `pol-rt2-germany.png`: photograph from Figure 2, PDF page 5 of the [RT-2 paper](https://robotics-transformer2.github.io/assets/rt2.pdf), illustrating semantic target selection.

The flow example is an analytic one-dimensional conditional distribution with two action modes; it is not a trained robot policy. Its derivation, numerical integration, and distinction between generation time and physical time are documented in the slide notes.

### LLM control source figures

See [recent-assets.md](recent-assets.md) for the Code as Policies, VoxPoser, and RoboDojo source URLs and crop details. These illustrate the named projects, not new experiments conducted for these slides.
