# References

## Core readings

| Material | How it supports the study group |
| --- | --- |
| [Stanford CS336 lectures](https://github.com/stanford-cs336/lectures) | Resource accounting, GPU diagrams, kernels, parallelism, and inference. |
| [Berkeley Scalable AI](https://scalable-ai.eecs.berkeley.edu/) | Performance reasoning, inference phases, benchmarking, and serving tradeoffs. |
| [Horace He: Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) | Compute, memory bandwidth, overhead, fusion, and profiler intuition. |
| [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html) | Training memory and distributed training; especially DP, ZeRO, TP, and PP. |
| [Scaling Book](https://jax-ml.github.io/scaling-book/) | GPU architecture, rooflines, Transformer accounting, and inference estimates. |
| [CurryTang: ML PhD Interview Notes](https://github.com/CurryTang/mlphdinterview) | Study organization, discussion prompts, and further interview preparation. |

The Ultra-Scale Playbook is also available through its [Hugging Face Space](https://huggingface.co/spaces/nanotron/ultrascale-playbook).

## Week 1 primary sources

- [NVIDIA CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA GPU performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
- [NVIDIA dense peak throughput](https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Hugging Face KV cache explanation](https://huggingface.co/docs/transformers/main/cache_explanation)
- [PaLM: MFU, Appendix B](https://jmlr.org/papers/volume24/22-1144/22-1144.pdf#page=90)
- [vLLM prefix-cache design](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM metrics](https://docs.vllm.ai/en/stable/usage/metrics/)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)

## Week 2 sources and teaching selections

| Source | Selected material |
| --- | --- |
| [CS336 Lecture 10: inference](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py#L162-L260) | Prefill versus decode, KV caching, and arithmetic intensity. |
| [CS336 Lecture 10: throughput and latency](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py#L332-L368) | Batch-size reasoning and simplified bandwidth-based estimates. These estimates are bounds under stated assumptions, not measured latency. |
| [CS336 Lecture 5, PDF pp. 50–54](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf#page=50) | FlashAttention as an application of tiling, data reuse, and incremental softmax. |
| [CS336 Lecture 6: benchmarking and profiling](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py#L144-L302) | Warm-up, device timing, profiling, and comparing implementations. |
| [Berkeley Lecture 18, PDF pp. 9–11](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_18.pdf#page=9) | Inference phases and the latency experienced by a user. |
| [Berkeley Lecture 19, part 1, PDF pp. 10–11](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_19_1.pdf#page=10) | Throughput/latency tradeoffs and the performance-improvement workflow. |
| [Berkeley Lecture 2, PDF pp. 44–45](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture2.pdf#page=44) | FlashAttention's effect on intermediate memory traffic. |
| [Berkeley Lecture 18, PDF p. 47](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_18.pdf#page=47) | Benchmark workloads, comparison conditions, and latency targets. |
| [FlashAttention paper](https://arxiv.org/abs/2205.14135) | Exact attention with less communication between HBM and on-chip storage. |
| [PyTorch CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) | Capture, replay, and the conditions needed for graph execution. |
| [Hugging Face KV cache explanation](https://huggingface.co/docs/transformers/main/cache_explanation) | Reusing past keys and values during autoregressive generation. |

PDF references use file page numbers, which can differ from printed slide counters. CS336 Lectures 6 and 10 are Python-based lecture presentations in the linked repository. The Week 2 figures are teaching adaptations with explicit assumptions; hardware performance must be measured for the workload and software version in use.

## Optional practice

- [LeetGPU challenges](https://leetgpu.com/challenges)
- [Scaling Book: Transformer calculations](https://jax-ml.github.io/scaling-book/transformers/)

## Attribution and scope

The course organization is inspired by CurryTang's study repository. The Week 1 GPU schematics are adapted with attribution from Stanford CS336 and NVIDIA documentation. Performance explanations draw on Horace He's article and the linked primary sources.

The numerical and conceptual questions are authored teaching exercises, not verified verbatim interview questions from named employers. Week 1 hardware examples use H100 SXM 80 GB. Week 2's optional lab records its actual hardware and software environment; implementation details may change across hardware variants and software versions. Third-party materials remain subject to their original terms.
