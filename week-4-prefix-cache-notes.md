# Week 4 - Prefix Caching and Cache Metrics

Teaching notes for the planned model-serving week. This material was moved out of the Week 1 cheatsheet; it is not a separate slide deck. It assumes familiarity with per-request KV storage and the prefill/decode distinction introduced in Week 1.

## 1. Three different meanings of "cache"

Define the numerator, denominator and measurement window before calculating.

| Concept | What is reused? | Metric / accounting |
| --- | --- | --- |
| Per-request KV cache | Past K/V within autoregressive generation | Payload from token count and architecture |
| Prefix cache | K/V for an identical token prefix across requests | Reused prompt tokens / queried prompt tokens |
| Hardware L1 / L2 cache | Memory data handled by GPU cache hardware | Cache accesses/hits, separate from prefix hits |

```text
Token hit rate = delta(prefix_cache_hits) / delta(prefix_cache_queries)
Request hit rate = requests with any prefix hit / queried requests
```

**Use counter deltas from the same time window.** vLLM prefix hit/query counters count tokens; verify the deployed version. Request hit rate uses requests. **KV pool occupancy** is used blocks/capacity, a third quantity. GPU type and model size alone cannot determine a prefix hit rate.

**Reuse requires** matching token IDs and model/cache identity, resident data and supported block granularity. Arrival order, warm/cold state, eviction and routing matter. Prefix caching saves prefill work; 60% hits does not mean 60% less end-to-end latency.

Sources: [vLLM cache metrics](https://docs.vllm.ai/en/stable/usage/metrics/#general-metrics), [vLLM prefix-cache design](https://docs.vllm.ai/en/stable/design/prefix_caching/), [vLLM prefix-cache limits](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/#limits).

---

## 2. Worked example: cold prefix cache, eight prompts

Count tokens for reuse; count stored positions for memory. State the physical sharing assumptions.

**Teaching model:** dense 8B GQA model with L = 32 layers, 32 query heads, h_KV = 8 KV heads, d_h = 128 values/head and BF16 KV at 2 bytes/value. It is a specified example, not an exact checkpoint. Across all layers, each cached position stores:

```text
m_token = 2 x L x h_KV x d_h x b_KV
        = 2 x 32 x 8 x 128 x 2
        = 131,072 bytes = 128 KiB
```

**Setup:** eight requests each have 4096 prompt tokens: an identical 3072-token prefix and a distinct 1024-token suffix (different from its first token). Prefills finish sequentially. The first lookup is cold; no eviction occurs. Retain all eight request states and exclude decode. Same model/cache identity; 16-token blocks; matching blocks are stored once and shared by reference.

| Quantity | Calculation | Result |
| --- | --- | --- |
| Token hits | 0 + 7 x 3072 | 21,504 tokens |
| Token queries | 8 x 4096 | 32,768 tokens |
| Token hit rate | 21,504 / 32,768 | 65.625% |
| Request hit rate | 7 / 8 | 87.5% |
| Unique cached positions | 3072 + 8 x 1024 | 11,264 |
| Shared KV payload | 11,264 x 128 KiB | 1.375 GiB |
| Without physical sharing | 8 x 4096 x 128 KiB | 4 GiB |

**Warm-cache variant:** if the prefix is resident before all eight lookups, the token hit rate is 3072 / 4096 = **75%**, and the request hit rate is **100%**. The final unique payload is still **1.375 GiB** under the same sharing assumptions.

The shared payload excludes metadata, unused reserved blocks and later decode growth. Identical prompt text alone is insufficient if tokenization or preceding tokens differ. Hit rate describes reuse events; physical KV sharing describes stored bytes. One does not uniquely determine the other.

Sources: [vLLM prefix-cache design](https://docs.vllm.ai/en/stable/design/prefix_caching/), [vLLM cache metrics](https://docs.vllm.ai/en/stable/usage/metrics/#general-metrics), [vLLM prefix-cache limits](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/#limits).

---

## 3. Serving interview checklist

**Metric:** reused prompt tokens / queried prompt tokens, requests with any reuse / queried requests, and used KV blocks / pool capacity answer different questions.

**Assumptions:** state token overlap, cache identity, arrival order, cache warmness, eviction, routing, block size and physical sharing. Model size and GPU type alone are insufficient.

**Memory effect:** prefix block sharing reduces the number of unique KV blocks when prefixes match and remain cached. Weight storage per replica is unchanged.

**Example to remember:** cold token hits **65.625%**, request hits **87.5%**, and shared KV payload **1.375 GiB** under the stated assumptions. A warm prefix changes the hit rates to **75%** and **100%** without changing that final unique payload.

**Interpretation:** a hit percentage does not directly predict end-to-end latency savings. Prefix caching avoids some prefill work; decode still attends to historical K/V. Measure the serving workload before making a latency or capacity promise.

**Provenance:** this worked exercise uses authored teaching numbers, not a verified verbatim frontier-lab interview question. Check implementation and metric details against the deployed software version.
