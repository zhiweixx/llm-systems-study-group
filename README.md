# LLM Systems Study Group

Slides, worked questions, cheatsheets, and references for a four-week study group on GPUs, LLM systems, inference, and model serving.

**[Live site](https://zhiweixx.github.io/llm-systems-study-group/)** · **[Present Week 1](https://zhiweixx.github.io/llm-systems-study-group/week-1/slides.html#slide-1)** · **[Present Week 2](https://zhiweixx.github.io/llm-systems-study-group/week-2/slides.html#slide-1)**

The audience knows Transformer models and PyTorch but has little GPU systems background. Materials use a plain academic style and state the assumptions behind calculations.

## Materials

| Week | Topic | Status |
| --- | --- | --- |
| 1 | GPU memory and performance | [24-slide presentation](week-1-gpu-memory-short.html), [cheatsheet](week-1-llm-systems-cheatsheet.md), [speaker notes](week-1-short-speaker-notes.md) |
| 2 | Single-GPU inference performance | [18-slide presentation](week-2-inference.html), [speaker notes](week-2-speaker-notes.md), [benchmark lab](week-2-lab/README.md) |
| 3 | Multi-GPU parallelism and sharding | Planned |
| 4 | LLM serving, scheduling, and KV management | Slides planned; [prefix-cache teaching notes](week-4-prefix-cache-notes.md) |

Week 1 is dated **September 10, 2026**, with a 30-minute presentation. The story connects GPU memory traffic to lower precision, fusion, coalescing, and tiling, then applies the ideas to memory budgets and MFU. Memory calculations use decimal GB and MB. Each of Questions 1–2 has a separate solution slide immediately afterward. Prefix-cache hit rates are reserved for Week 4. The full plan is in [curriculum.md](curriculum.md).

Week 2 is dated **September 17, 2026**, with a 30-minute presentation. It connects prefill, decode, and KV caching to batching, FlashAttention, CUDA Graphs, and performance measurement. Each conceptual question has a solution immediately afterward. The optional benchmark lab measures a fixed workload across batch sizes; illustrative examples in the slides are labeled separately from hardware measurements.

## Presenting and sharing

Open the live presentation URL in a browser. Use the arrow keys, the Slides overview, or the Notes control. The HTML file is self-contained and can also be downloaded and opened offline. Use the presentation's Print / PDF control for a printable copy.

Slide 6 compares A100-normalized compute, HBM bandwidth, and memory capacities to motivate data reuse. Its [data, source notes, and plotting code](site/slide-assets/gpu-relative-growth/) are included in the repository.

Slide 14 includes an interactive tiling example. Its Previous step and Next step buttons show input loading, partial-sum accumulation, and the final write to HBM. Print / PDF shows the completed example.

Share the live site URL for the complete materials, or a direct slide link such as `week-1/slides.html#slide-17` for a question. All files in this repository and the Pages site are public.

## Updating the materials

- Edit `week-1-gpu-memory-short.html` for the Week 1 presentation.
- For Week 2, edit `scripts/build_week2.py` for slide content and speaker notes, or `site/slide-assets/week2/` for the interactive examples and styling. Run `python3 scripts/build_week2.py` to regenerate `week-2-inference.html` and `week-2-speaker-notes.md`.
- Edit `week-2-lab/` for the optional inference benchmark and notebook.
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

Week 2 browser checks are available in `scripts/check_week2.cjs` for environments with Playwright and Chrome. They exercise the examples, CSV import, navigation, and print layout. The lab's `test_lab.py` checks measurement semantics and notebook/source consistency; `benchmark.py --check` compares cached decoding with full causal recomputation.

## Sources and contributions

See [references.md](resources.md) for readings and attribution. The organization is inspired by [CurryTang/mlphdinterview](https://github.com/CurryTang/mlphdinterview); the slides and numerical exercises here were prepared for this study group. These are authored teaching questions, not verified verbatim questions from named employers.

Corrections and additions are welcome through issues or pull requests. Please include units, assumptions, and a primary source for factual changes. See [CONTRIBUTING.md](CONTRIBUTING.md).
