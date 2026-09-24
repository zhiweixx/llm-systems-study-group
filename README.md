# LLM Systems Study Group

Slides, worked questions, cheatsheets, and references for a four-week study group on GPUs, LLM systems, inference, and model serving.

**[Live site](https://zhiweixx.github.io/llm-systems-study-group/)** · **[Present Week 1](https://zhiweixx.github.io/llm-systems-study-group/week-1/slides.html#slide-1)** · **[Present Week 2](https://zhiweixx.github.io/llm-systems-study-group/week-2/slides.html#slide-1)** · **[Present Week 3](https://zhiweixx.github.io/llm-systems-study-group/week-3/slides.html#slide-1)**

The audience knows Transformer models and PyTorch but has little GPU systems background. Materials use a plain academic style and state the assumptions behind calculations.

## Materials

| Week | Topic | Status |
| --- | --- | --- |
| 1 | GPU memory and performance | [24-slide presentation](week-1-gpu-memory-short.html), [cheatsheet](week-1-llm-systems-cheatsheet.md), [speaker notes](week-1-short-speaker-notes.md) |
| 2 | LLM inference performance | [31-slide presentation](week-2-inference.html), [speaker notes](week-2-speaker-notes.md), [benchmark lab](week-2-lab/README.md) |
| 3 | Multi-GPU parallelism and sharding | [21-slide visual overview](week-3-parallelism-overview.html), [overview notes](week-3-overview-speaker-notes.md), [50-slide reference](week-3-parallelism.html), [reference notes](week-3-speaker-notes.md), [PyTorch exercise](week-3-tp-exercise.py) |
| 4 | LLM serving, scheduling, and KV management | Slides planned; [prefix-cache teaching notes](week-4-prefix-cache-notes.md) |

Week 1 is dated **September 10, 2026**, with a 30-minute presentation. The story connects GPU memory traffic to lower precision, fusion, coalescing, and tiling, then applies the ideas to memory budgets and MFU. Memory calculations use decimal GB and MB. Each of Questions 1–2 has a separate solution slide immediately afterward. Prefix-cache hit rates are reserved for Week 4. The full plan is in [curriculum.md](curriculum.md).

Week 2 is dated **September 17, 2026**, with about 35 minutes of suggested full content. It starts from latency metrics and generation, establishes why KV can be reused, and compares weight and KV traffic before the first diagnostic question. Continuous batching and paged storage lead to a visible online-softmax derivation, following one query row through weighted sums, stable state, and rescaling. A mixed-batch example establishes the context for the published DistServe figure, chunked prefill, and prefill–decode disaggregation. Both in-class questions have an immediate solution slide. The deck ends with a take-home tiled GEMM exercise and the official Triton tutorial as its reference solution (slides 30–31). These two slides are outside the lecture timing. The optional benchmark lab produces measurements from the presenter's GPU run.

A suggested **approximately 32-minute Week 2 route** skips slides 8, 27, and 28: the H100 numerical example and the optional experiment section. It retains the arithmetic-intensity prerequisites, complete online-softmax derivation (17–19), and mixed-batch setup (23). All 31 slides remain available for reading. Timings are planning estimates, not rehearsed durations; the shorter route reserves 15 minutes for discussion and roughly 3 minutes of buffer in a 50-minute meeting. Advanced disaggregation scheduling and network placement belong to Week 4.

Week 3 is an expanded **50-slide** deck with no fixed presentation duration. It covers serving replicas, tensor and pipeline parallelism, data-parallel training, ZeRO/FSDP, context parallelism, and expert parallelism. The sequence follows tensor ownership and communication through a paired MLP, a layer pipeline, distributed attention, and token routing to MoE experts. Context parallelism includes ring attention, stable summary merging, causal load balance, Ulysses, the distinction from Megatron sequence parallelism, and decode-history sharding. Additional lessons explain pipeline bubbles, the tradeoff between microbatch count and size, framework scheduling, TP/CP configuration constraints, communication-aware GPU placement, and multi-node CP/EP. Questions 1–2 appear on slides 45 and 47 with immediate solutions; slides 49–50 give a take-home MLP simulation and reference solution. The [runnable exercise](week-3-tp-exercise.py) checks the partition on a CPU or one GPU; it does not claim distributed performance measurements.

## Presenting and sharing

The separate [21-slide Week 3 overview](https://zhiweixx.github.io/llm-systems-study-group/week-3/overview.html#slide-1) introduces the same parallelism and sharding concepts through ownership diagrams, a pipeline timeline, and a step-through ring-attention example. The original 50-slide deck remains intact at its existing URL for detailed study and exercises.

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
