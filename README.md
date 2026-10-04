# LLM Systems Study Group

Slides, worked questions, cheatsheets, and references for a study group on GPUs, LLM systems, inference, model serving, and linear attention.

**[Live site](https://zhiweixx.github.io/llm-systems-study-group/)** · **[Present Week 1](https://zhiweixx.github.io/llm-systems-study-group/week-1/slides.html#slide-1)** · **[Present Week 2](https://zhiweixx.github.io/llm-systems-study-group/week-2/slides.html#slide-1)** · **[Present Week 3](https://zhiweixx.github.io/llm-systems-study-group/week-3/slides.html#slide-1)** · **[Present Week 4](https://zhiweixx.github.io/llm-systems-study-group/week-4/slides.html#slide-1)** · **[Present Week 5](https://zhiweixx.github.io/llm-systems-study-group/week-5/slides.html#slide-1)**

The audience knows Transformer models and PyTorch but has little GPU systems background. Materials use a plain academic style and state the assumptions behind calculations.

## Materials

| Week | Topic | Status |
| --- | --- | --- |
| 1 | GPU memory and performance | [24-slide presentation](week-1-gpu-memory-short.html), [cheatsheet](week-1-llm-systems-cheatsheet.md), [speaker notes](week-1-short-speaker-notes.md) |
| 2 | LLM inference performance | [31-slide presentation](week-2-inference.html), [speaker notes](week-2-speaker-notes.md), [benchmark lab](week-2-lab/README.md) |
| 3 | Multi-GPU parallelism and sharding | [22-slide visual overview](week-3-parallelism-overview.html), [overview notes](week-3-overview-speaker-notes.md), [50-slide reference](week-3-parallelism.html), [reference notes](week-3-speaker-notes.md), [PyTorch exercise](week-3-tp-exercise.py) |
| 4 | Speculative decoding and LLM serving | [30-slide presentation](week-4-serving.html), [speaker notes](week-4-speaker-notes.md), [optional serving lab](week-4-lab/README.md) |
| 5 | Linear attention | [49-slide presentation](week-5-linear-attention.html), converted from Gaotang Li’s [original PDF](sources/week-5/Linear_Attention_gaotang_li.pdf) |

Week 1 is dated **September 10, 2026**, with a 30-minute presentation. The story connects GPU memory traffic to lower precision, fusion, coalescing, and tiling, then applies the ideas to memory budgets and MFU. Memory calculations use decimal GB and MB. Each of Questions 1–2 has a separate solution slide immediately afterward. Prefix-cache hit rates are reserved for Week 4. The full plan is in [curriculum.md](curriculum.md).

Week 2 is dated **September 17, 2026**, with about 35 minutes of suggested full content. It starts from latency metrics and generation, establishes why KV can be reused, and compares weight and KV traffic before the first diagnostic question. Continuous batching and paged storage lead to a visible online-softmax derivation, following one query row through weighted sums, stable state, and rescaling. A mixed-batch example establishes the context for the published DistServe figure, chunked prefill, and prefill–decode disaggregation. Both in-class questions have an immediate solution slide. The deck ends with a take-home tiled GEMM exercise and the official Triton tutorial as its reference solution (slides 30–31). These two slides are outside the lecture timing. The optional benchmark lab produces measurements from the presenter's GPU run.

A suggested **approximately 32-minute Week 2 route** skips slides 8, 27, and 28: the H100 numerical example and the optional experiment section. It retains the arithmetic-intensity prerequisites, complete online-softmax derivation (17–19), and mixed-batch setup (23). All 31 slides remain available for reading. Timings are planning estimates, not rehearsed durations; the shorter route reserves 15 minutes for discussion and roughly 3 minutes of buffer in a 50-minute meeting. Advanced disaggregation scheduling and network placement belong to Week 4.

Week 3 is an expanded **50-slide** deck with no fixed presentation duration. It covers serving replicas, tensor and pipeline parallelism, data-parallel training, ZeRO/FSDP, context parallelism, and expert parallelism. The sequence follows tensor ownership and communication through a paired MLP, a layer pipeline, distributed attention, and token routing to MoE experts. Context parallelism includes ring attention, stable summary merging, causal load balance, Ulysses, the distinction from Megatron sequence parallelism, and decode-history sharding. Additional lessons explain pipeline bubbles, the tradeoff between microbatch count and size, framework scheduling, TP/CP configuration constraints, communication-aware GPU placement, and multi-node CP/EP. Questions 1–2 appear on slides 45 and 47 with immediate solutions; slides 49–50 give a take-home MLP simulation and reference solution. The [runnable exercise](week-3-tp-exercise.py) checks the partition on a CPU or one GPU; it does not claim distributed performance measurements.

Week 4 opens with speculative decoding on slides 2–11. Slides 2–3 introduce sequential drafting and parallel verification, followed by the exact stochastic sampling algorithm. KV continuation and an interactive timing comparison connect the algorithm to its implementation cost. A multiple-choice quiz on slides 6–7 asks for the output distribution when a rejected proposal is replaced by a fresh target-model sample; its solution precedes four theory slides deriving exact sampling, acceptance as distribution overlap, expected tokens per round, and expected speedup.

Slide 12 introduces the hardware for the serving case with a GPU memory-hierarchy pyramid and a simple H100-versus-B200 bandwidth table. The table covers shared memory, L2, HBM, NVLink, and InfiniBand, with specifications distinguished from measurements and estimates.

Slides 13–25 apply Week 3's sharding methods to **serving Llama 3.1 405B on eight nodes, each with eight H100 GPUs: 64 GPUs in total**. The case combines the question-led approach of the [PyTorch Scaling Book's applied-inference chapter](https://liltom-eth.github.io/scaling-book-pytorch/chapters/applied-inference.html) with explicit model, memory, and communication assumptions. It compares BF16 TP8 × PP2, TP16, and context-parallel cache placement; traces the difference between pipeline latency and throughput; estimates decode and prefill costs; and considers long contexts and an FP8 deployment. The case concludes with launching and choosing a layout based on the workload and measurements.

The final section starts with the request lifecycle on slide 26, then follows successive model calls within one coding-agent session on slides 27–30. It explains how tool outputs grow the prompt, what token cache hit rate measures, and why editing old context invalidates later KV states. An interactive example on slide 30 compares retaining history, compacting it, and reusing the compacted history on the next call.

The 30-slide presentation is an expanded teaching deck with no fixed presentation duration. Use the Slides menu to select sections. The sampling quiz has an immediate solution; the serving case develops its calculations and tradeoffs across successive slides. Analytical performance estimates are not GPU measurements. The optional companion lab prints reproducible vLLM benchmark commands by default and runs them only with `--run`; it remains available as a separate measurement exercise.

Week 5 converts Gaotang Li’s *Linear Attention* deck into a **49-slide** standalone HTML presentation, retaining the source sequence and appendix. Topics include linear attention, DeltaNet, Gated DeltaNet, Kimi Delta Attention, and chunkwise parallelism. Gaotang Li is credited at the beginning of the presentation; the [original PDF](sources/week-5/Linear_Attention_gaotang_li.pdf) is preserved in the repository.

## Presenting and sharing

The separate [22-slide Week 3 overview](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-1) introduces the same parallelism and sharding concepts through ownership diagrams, a pipeline timeline, and a step-through ring-attention example. Slide 21 connects PyTorch, Megatron, and DeepSpeed with a software diagram and historical milestones. The original 50-slide deck remains intact at its existing URL for detailed study and exercises.

Open the live presentation URL in a browser. Use the arrow keys, the Slides overview, or the Notes control. The HTML file is self-contained and can also be downloaded and opened offline. Use the presentation's Print / PDF control for a printable copy.

Week 1 slide 6 compares A100-normalized compute, HBM bandwidth, and memory capacities to motivate data reuse. Its [data, source notes, and plotting code](site/slide-assets/gpu-relative-growth/) are included in the repository.

Week 1 slide 14 includes an interactive tiling example. Its Previous step and Next step buttons show input loading, partial-sum accumulation, and the final write to HBM. Print / PDF shows the completed example. Week 2 has interactive generation and KV block-table diagrams. The published figure's [source and reproduction details](site/slide-assets/week2/README.md) are recorded alongside its image.

Week 3 includes step-through examples for the tensor-parallel MLP (slide 9), ring attention (26), and expert dispatch, computation, and return (37). Its native SVG diagrams are editable in the source modules and are included in the standalone HTML. Print / PDF uses the completed interactive state.

Share the live site URL for the complete materials, or a direct slide link such as `week-1/slides.html#slide-17` for a question. All files in this repository and the Pages site are public.

## Updating the materials

- Edit `week-1-gpu-memory-short.html` for the Week 1 presentation.
- For Week 2, edit `scripts/build_week2.py` for slide content and speaker notes, or `site/slide-assets/week2/` for the interactive examples and styling. Run `python3 scripts/build_week2.py` to regenerate `week-2-inference.html` and `week-2-speaker-notes.md`.
- Edit `week-2-lab/` for the optional inference benchmark and notebook.
- For Week 3, edit the content modules in `scripts/week3/` and the frame generator `scripts/build_week3.py`, or `site/slide-assets/week3/` for controls and styling. Run `.venv/bin/python scripts/build_week3.py` to regenerate `week-3-parallelism.html` and `week-3-speaker-notes.md`. Edit `week-3-tp-exercise.py` for the runnable companion.
- For the separate Week 3 overview, edit `scripts/week3/overview_*.py` and run `.venv/bin/python scripts/build_week3_overview.py`. Check it with `scripts/check_week3_overview.cjs`; this build does not modify the 50-slide deck.
- For Week 4, edit `scripts/week4/` and `site/slide-assets/week4/`, then run `.venv/bin/python scripts/build_week4.py`. Run `scripts/check_week4.cjs` with Playwright and Chrome for layout, navigation, interaction, and print checks. The optional serving experiment is in `week-4-lab/`.
- For Week 5, `scripts/build_week5.py` converts the original PDF in `sources/week-5/` to vector SVG pages embedded in `week-5-linear-attention.html`. Regeneration requires Python with `pypdf` and Poppler’s `pdftocairo`. Edit the converter for attribution or presentation controls, and run `scripts/check_week5.cjs` with Playwright and Chrome to check fidelity, navigation, and printing. Preserve the attribution to Gaotang Li and the original PDF.
- Edit the Markdown sources for the cheatsheet, speaker notes, curriculum, and references.
- Edit `site/index.html`, `site/page.html`, and `site/styles.css` for the site layout.
- Push to `main`. GitHub Actions rebuilds and publishes the site automatically.

### Local build

Requires Python 3.10 or later.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/build_site.py
.venv/bin/python scripts/check_site.py
.venv/bin/python -m http.server 8000 --directory _site
```

The generated `_site/` directory is ignored by Git. The build uses one pinned Markdown dependency; the published pages need no server or external rendering libraries.

Week 2 browser checks are available in `scripts/check_week2.cjs` for environments with Playwright and Chrome. They check all slide layouts, interactive examples, navigation, offline resources, and print-state restoration. The lab's `test_lab.py` checks measurement semantics and notebook/source consistency; `benchmark.py --check` compares cached decoding with full causal recomputation.

Week 3 browser checks are in `scripts/check_week3.cjs`. Run the companion exercise with `python week-3-tp-exercise.py` in an environment with PyTorch installed; the default correctness simulation needs no distributed setup.

## Sources and contributions

See [references.md](resources.md) for readings and attribution. The organization is inspired by [CurryTang/mlphdinterview](https://github.com/CurryTang/mlphdinterview); the slides and numerical exercises here were prepared for this study group. These are authored teaching questions, not verified verbatim questions from named employers.

Corrections and additions are welcome through issues or pull requests. Please include units, assumptions, and a primary source for factual changes. See [CONTRIBUTING.md](CONTRIBUTING.md).
