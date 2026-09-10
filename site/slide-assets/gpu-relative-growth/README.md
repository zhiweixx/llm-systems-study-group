# GPU compute and memory scaling

The five curves compare A100, H100, H200, B100 and B200. Each metric is divided by its own A100 value, so all five series start at exactly 1.0. The y-axis is linear. The x-axis lists GPU models, not equally spaced release dates.

## Files

- `gpu-relative-growth.png`: 4200 × 2460 pixels, white background, suitable for slides.
- `gpu-relative-growth.svg`: vector figure with editable text.
- `gpu-relative-growth.pdf`: vector figure for export or printing.
- `data.csv`: raw values, units in the column names, and configuration qualifications.
- `normalized.csv`: values divided by the corresponding A100 baseline; missing data stay blank.
- `plot.py`: complete Matplotlib source, tested with Matplotlib 3.10.8.
- `gpu-relative-growth-slide.svg`: compact vector chart embedded in slide 6.
- `plot-slide.py`: reproduces the compact chart from the same CSV and plotting definitions.

## Values and normalization

| GPU | Dense BF16/FP16 Tensor Core (TFLOP/s) | HBM bandwidth (TB/s) | HBM (GB/GPU) | L2 (MB/GPU) | Combined L1/shared/texture pool (KB/SM) |
|---|---:|---:|---:|---:|---:|
| A100 80GB SXM | 312 | 2.039 | 80 | 40 | 192 |
| H100 SXM | 989 | 3.35 | 80 | 50 | 256 |
| H200 SXM | 989 | 4.8 | 141 | 50 | 256 |
| B100, preliminary | 1,750 | 8 | 192 | not verified | 256, architecture-level value |
| B200 SXM/HGX | 2,250 | 8 | 180 | 126 | 256 |

For example, the B200 compute ratio is 2250 / 312 = 7.2115, and its HBM bandwidth ratio is 8 / 2.039 = 3.9235.

## Interpretation and qualifications

- These are specification peaks and capacities, not measured application performance. They describe this selected comparison, not a universal growth law. H100 and H200 have the same stated BF16 compute peak while H200 increases HBM capacity and bandwidth.
- The L1 series is **not dedicated L1 data-cache capacity**. NVIDIA provides a combined L1/shared-memory/texture capacity whose allocation can change with kernel configuration. It is measured per SM. L2 and HBM capacities are measured per whole GPU. This plot does not compare aggregate L1 SRAM across all SMs.
- B100 values come from NVIDIA's early Blackwell Technical Brief V1.1, explicitly marked preliminary. Its Tensor Core peaks are given with sparsity; the figure divides those BF16/FP16 and FP8 specifications by two to use dense throughput. The B100 L1/shared-memory value is a Blackwell architecture reference rather than a verified final B100 SKU measurement.
- No verified B100 L2 value was available in the reviewed sources. It is missing in the CSV and the plotted line is broken at B100. No value is imputed and no line crosses the gap. The B200 L2 endpoint remains visible.
- B200 uses 180 GB from NVIDIA's HGX reference architecture and CUDA tuning guide, consistent with the preceding comparison. Other NVIDIA documents give an up-to-192 GB value. Using 192 GB would make the B200 HBM endpoint 2.40× instead of 2.25×; it does not alter the other series. The B100-to-B200 HBM decrease reflects the preliminary B100 versus selected B200 SKU values, not a general architectural regression.
- B200 compute uses the conventional rounded 2.25 PFLOP/s dense peak. NVIDIA's V2.1 HGX table displays 2.2/4.5 PFLOP/s dense/sparse. The HGX reference architecture gives up to 8 TB/s bandwidth; the V2.1 brief gives 7.7 TB/s. This figure uses the same 8 TB/s peak reference as the preceding table.
- Cache labels follow NVIDIA's KB/MB nomenclature. Ratios compare the same kind of capacity within each series; no binary/decimal conversion is used to create a generational increase.

## Primary sources

Source review date: September 10, 2026.

1. [NVIDIA A100 specifications](https://www.nvidia.com/en-us/data-center/a100/) — 80 GB, 2,039 GB/s, 312 dense BF16/FP16 TFLOP/s.
2. [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) — A100 40 MB L2 and 192 KB combined L1/shared-memory/texture capacity per SM.
3. [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/) and [Blackwell Ultra architecture article](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) — H100 capacity, compute and 3.35 TB/s bandwidth reference. The conventional rounded dense BF16 estimate is 989 TFLOP/s.
4. [NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/) — 141 GB, 4.8 TB/s and H100-class Tensor Core compute.
5. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) and [NVIDIA CUDA programming/optimization lecture](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/CUDA-Programming-and-Optimization.pdf) — Hopper combined L1/shared-memory pool; H100/H200 50 MB L2.
6. [NVIDIA Blackwell Technical Brief V1.1, pp. 19–20 (mirror of NVIDIA-authored PDF)](https://catalogone.com/wp-content/uploads/2024/06/NVIDIA-Blackwell-Technical-Brief.pdf) — B100 preliminary memory/bandwidth and sparse Tensor Core peaks; source explicitly marks preliminary specifications.
7. [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) — B200 180 GB, 256 KB combined per-SM pool, and Blackwell 126 MB L2 reference.
8. [NVIDIA HGX reference architecture: components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html) — B200 SXM 180 GB and up to 8 TB/s.
9. [NVIDIA Blackwell Technical Brief V2.1, pp. 25–26](https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf) — HGX B200 dense/sparse compute and alternate rounded specification values.

## Reproduce

```sh
python -m pip install matplotlib==3.10.8
python plot.py
python plot-slide.py
```

The comparison appears on [Week 1, slide 6](https://zhiweixx.github.io/llm-systems-study-group/week-1/slides.html#slide-6), between the bandwidth explanation and the optimization roadmap. The embedded chart is self-contained and works offline; these files preserve its data and provenance.
