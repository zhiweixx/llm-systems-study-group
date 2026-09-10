# References

## Core readings

| Material | How it supports the study group |
| --- | --- |
| [Stanford CS336 lectures](https://github.com/stanford-cs336/lectures) | Resource accounting, GPU diagrams, kernels, parallelism, and inference. |
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

## Optional practice

- [LeetGPU challenges](https://leetgpu.com/challenges)
- [Scaling Book: Transformer calculations](https://jax-ml.github.io/scaling-book/transformers/)

## Attribution and scope

The course organization is inspired by CurryTang's study repository. The Week 1 GPU schematics are adapted with attribution from Stanford CS336 and NVIDIA documentation. Performance explanations draw on Horace He's article and the linked primary sources.

The numerical questions are authored teaching exercises, not verified verbatim interview questions from named employers. Hardware examples use H100 SXM 80 GB; implementation details may change across hardware variants and software versions. Third-party materials remain subject to their original terms.
