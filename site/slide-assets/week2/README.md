# Week 2 visual sources

## Published experiment: DistServe Figure 2

`distserve-figure2.png` reproduces Figure 2 from:

Yinmin Zhong et al., **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**, OSDI 2024.

- [Paper page](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [Original PDF, physical page 5 / proceedings page 196](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=5)
- Crop coordinates in PDF points, measured from the top-left: `[50, 68, 294, 183]`. Rendered at 720 DPI; both panels, axes, legend, and input-length labels are retained. The surrounding paper caption is excluded from the image.
- The caption identifies a 13B LLM and input lengths 128 and 1,024. It does not supply a complete hardware/backend configuration for this figure. The slides therefore do not attach an inferred GPU model to it.
- Purpose: compare decoding-only batch execution with decoding plus one prefill job at the same batch size. This motivates discussion of interference, chunked prefill, and PD disaggregation. The vertical axis measures batch execution time, not throughput or end-to-end request latency.

The figure is embedded in the standalone HTML. Source links appear on its slide and in the speaker notes. It is a published experiment, not a measurement from the study group's lab.

## Authored diagrams and calculations

Other diagrams are authored teaching schematics. The generation and paged-KV examples have step controls; scheduling diagrams show ordering rather than measured durations. The H100 bars show ideal resource-time bounds under stated assumptions.

The earlier synthetic throughput plot, based on an assumed step-time formula, has been removed. The optional lab generates plots from the presenter's measured CSV data.

## Teaching references

- [Stanford CS336 Lecture 10](https://cs336.stanford.edu/lectures/?trace=lecture_10): inference metrics, compute and memory costs, batching, and PagedAttention.
- [Stanford CS336 Lecture 5, PDF pp. 52–54](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=52): online softmax. Lecture 10 does not derive this recurrence.
- [Berkeley Scalable AI, Spring 2026 Lecture 18](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf): phase-specific metrics, KV memory, scheduling, chunked prefill, PD disaggregation, and benchmarking. Page references in the slides use physical PDF pages.
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) and [FlashAttention](https://arxiv.org/abs/2205.14135): stable incremental normalization and tiled exact attention.
- [PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180): paged KV storage and compatible attention kernels.

Questions are authored diagnostic exercises, not verified interview questions from particular employers.
