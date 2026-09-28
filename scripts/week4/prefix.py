"""Prefix reuse, KV lifetime and a latency-oriented routing exercise."""
from .common import *

DESIGN = ('vLLM prefix-cache design', 'https://docs.vllm.ai/en/stable/design/prefix_caching/')
APC = ('vLLM prefix caching', 'https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/')
PAGED = ('PagedAttention paper', 'https://arxiv.org/abs/2309.06180')
ROUTER = ('SGLang cache-aware routing', 'https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/policies/cache_aware.rs')


def tokens(x, y, values, fill=PALE, cell=68, height=58):
    out = ''
    for i, value in enumerate(values):
        out += rect(x+i*cell, y, cell, height, fill)
        out += text(x+(i+.5)*cell, y+38, value, 26, 600, anchor='middle')
    return out


def prefix_identity():
    b = text(75, 190, 'Cached KV represents a token together with its preceding context.', 30)
    b += text(75, 261, 'Cached request', 28, 700)
    b += tokens(390, 224, ['P1', 'P2', 'P3', 'P4'])
    b += tokens(672, 224, ['A1', 'A2', 'A3', 'A4'], '#f4f4f4')
    b += text(1010, 261, 'Prefix P is available.', 27, color=BLUE)
    b += text(75, 365, 'New request', 28, 700)
    b += tokens(390, 328, ['P1', 'P2', 'P3', 'P4'])
    b += tokens(672, 328, ['B1', 'B2', 'B3', 'B4'], '#f4f4f4')
    b += lines(1010, 356, ['Reuse P’s KV;', 'compute the new suffix.'], 27, gap=35, color=BLUE)
    b += arrow(526, 288, 526, 320)
    b += line(75, 426, 1525, 426)
    b += text(75, 493, 'Changed context', 28, 700)
    b += tokens(390, 456, ['Q1', 'P2', 'P3', 'P4'], '#faf1e9')
    b += tokens(672, 456, ['A1', 'A2', 'A3', 'A4'], '#f4f4f4')
    b += lines(1010, 484, ['Same suffix text;', 'its cached KV is not valid.'], 27, gap=35, color=WARM)
    b += lines(75, 592, [
        'Match token IDs from the beginning, under the same model and position context.',
        'A matching phrase in the middle of a different prompt is insufficient.',
    ], 28, gap=47)
    b += takeaway('Prefix caching reuses an earlier computation, not an earlier answer.')
    return slide('Reuse KV when the causal prefix matches', b,
        'Use P and A/B as symbolic token IDs, not words. For the ordinary causal Transformer in our case, reusing a cached block requires the matching preceding token context. The second request can reuse P, then computes B while attending to P. Changing Q1 invalidates this causal-prefix match even though the later text A is identical. Model weights, adapters and relevant input/position settings must also agree. The serving engine validates cache identity; semantic similarity alone is not sufficient. The source describes including the parent-prefix identity and relevant extra inputs in each block key.',
        [DESIGN], 'Prefix caching')


def block_sharing():
    scenes = []
    suffix = ['A', 'B', 'C']
    captions = [
        '1. A arrives cold: compute P and A; store two KV blocks.',
        '2. B arrives: reuse P; compute and allocate only B.',
        '3. C arrives: reuse P; compute and allocate only C.',
    ]
    for step in range(3):
        s = text(75, 190, 'Toy prompts: 4 shared prefix tokens + 4 unique suffix tokens. Block size = 4.', 28)
        s += text(75, 249, 'Request block tables', 30, 700)
        s += text(945, 249, 'Physical KV blocks on one GPU', 30, 700)
        for i, name in enumerate(suffix):
            yy = 292 + 92*i
            active = i <= step
            c = INK if active else '#a6adb1'
            s += text(75, yy+37, f'Request {name}', 28, 600, color=c)
            s += label_box(270, yy, 135, 56, 'P' if active else '—', PALE if active else '#fafafa')
            s += label_box(424, yy, 135, 56, name if active else '—', '#f4f4f4' if active else '#fafafa')
            if active:
                s += line(337, yy+56, 337, yy+70, BLUE, 2)
                s += line(337, yy+70, 730, yy+70, BLUE, 2)
                s += arrow(730, yy+70, 919, 355)
        s += rect(920, 289, 225, 134, PALE)
        s += text(1032, 336, 'P', 31, 700, anchor='middle', color=BLUE)
        s += text(1032, 385, 'P1 P2 P3 P4', 24, anchor='middle')
        for i, name in enumerate(suffix):
            x = 1172
            yy = 285 + i*104
            active = i <= step
            s += rect(x, yy, 325, 86, '#f4f4f4' if active else 'white', LINE if active else '#dfe3e6')
            s += text(x+162, yy+34, f'{name}: {name}1 {name}2 {name}3 {name}4' if active else 'Unallocated', 25, 600 if active else 400, anchor='middle', color=INK if active else '#a6adb1')
            if active: s += text(x+162, yy+67, 'private suffix', 22, anchor='middle', color=MUTED)
        s += text(75, 585, captions[step], 29, 600, color=BLUE)
        s += text(75, 630, f'{step+1} request(s) retain prompt KV; {step+2} physical blocks × 4 token positions.', 26, color=MUTED)
        scenes.append(s)
    b = steps('prefix', scenes)
    b += takeaway('Several requests can reference one computed, read-only prefix block.')
    return slide('Share prefix blocks; allocate each unique suffix', b,
        'Step through the arrival of A, B, then C at the same replica. All three prompt KV states remain retained; no decoding or eviction occurs in this toy. Each suffix differs at its first token, so only the first block matches. The request block tables refer to physical block IDs. The three arrows terminate at the same physical P: there are not three copies of its KV. At the last step we have four physical blocks: P, A, B and C. Each represents four token positions across the model’s KV state. The block size of four is a teaching choice. Real cache managers handle partial blocks and boundaries; this example aligns all blocks to avoid that complication.',
        [PAGED, DESIGN], 'Prefix caching')


def kv_lifetime():
    b = text(75, 190, 'Follow one computed prefix block P inside a replica’s KV pool.', 30)
    states = [
        (75, 'Active', '2 requests reference P', 'Keep P available to those requests.', PALE),
        (583, 'Idle but cached', '0 requests reference P', 'Retain its contents for a future hit.', '#f4f4f4'),
        (1091, 'Reassigned', 'P’s cache entry is removed', 'Overwrite this slot for other KV.', '#faf1e9'),
    ]
    for x, title, subtitle, desc, fill in states:
        b += rect(x, 260, 434, 262, fill)
        b += text(x+24, 308, title, 32, 700, color=BLUE)
        b += text(x+24, 359, subtitle, 25, 600)
        b += label_box(x+24, 384, 385, 61, 'P' if title != 'Reassigned' else 'New KV', 'white', 28)
        b += text(x+24, 490, desc, 22)
    b += arrow(509, 390, 574, 390)
    b += arrow(1017, 390, 1082, 390)
    b += lines(525, 237, ['users finish'], 21, color=MUTED, anchor='middle')
    b += lines(1056, 237, ['slot needed'], 21, color=MUTED, anchor='middle')
    b += lines(75, 590, ['Cache hit while idle: attach P to the new request; P becomes active again.',
                         'Eviction: reclaim idle cached blocks when space is needed.'], 29, gap=47)
    b += text(75, 698, 'Reclaiming a block need not reduce the GPU memory allocation of the preallocated KV pool.', 24, color=MUTED)
    b += takeaway('Idle cached KV is reusable and evictable; active KV is still needed.')
    return slide('A finished request does not require deleting its KV', b,
        'The reference count records how many current requests depend on a block. Finishing a request removes its reference. When the count reaches zero, the manager may retain the computed contents as a cache candidate while making the slot reclaimable. A hit reactivates the block. Reassigning the slot invalidates the old cached mapping before new data overwrites it. This is a logical lifecycle within a preallocated memory pool, not necessarily a GPU allocation/free event. In vLLM V1, an idle cached block can already belong to the free-block queue: free for allocation does not mean its old contents have been erased. Avoid interpreting a high allocated-HBM number as entirely non-reclaimable active KV.',
        [DESIGN], 'Prefix caching')


def cache_metrics():
    b = text(75, 190, 'Same toy: 3 prompts × 8 tokens; shared prefix length 4. A is cold; B and C hit.', 28)
    b += table(75, 247, 1450,
        ['Measure', 'Counting rule in this toy', 'Result'],
        [('Token hit fraction', '(0 + 4 + 4) / (8 + 8 + 8)', '33.3%'),
         ('Requests with a hit', '2 requests / 3 requests', '66.7%'),
         ('Physical prompt KV', 'P + A + B + C = 4 blocks', '16 token positions')],
        ratios=[.29, .47, .24], row_h=95, size=28)
    b += text(75, 671, 'Without sharing: 24 positions. Snapshot assumes all three prompts remain retained.', 27)
    b += takeaway('Hit rates count reuse over arrivals; memory occupancy counts stored blocks.',
                   'A real dashboard must name its denominator and measurement window.')
    return slide('Cache hit rate and KV occupancy answer different questions', b,
        'The token-hit fraction answers what portion of the arriving prompt tokens used existing KV in this explicitly defined three-request window. The request fraction answers whether a request reused anything, even a short prefix. Physical occupancy is a snapshot of unique retained KV blocks, rather than a traffic ratio. Our arithmetic follows directly from the preceding diagram: the cold request needs eight token positions, and each of the next two contributes four new ones. The cache saves eight recomputed positions across these arrivals. This toy counts prompt KV only, with no output growth, eviction, duplication or partial-block waste. Production counters can use different eligibility and boundary conventions; verify the engine metric before comparing systems.',
        [PAGED, DESIGN], 'Prefix caching')


def remaining_work():
    b = text(75, 190, 'For a new request [shared prefix P | new suffix A], a prefix hit supplies P’s KV.', 29)
    b += line(785, 243, 785, 676)
    b += text(75, 271, 'Prefill after a cache hit', 33, 700, color=BLUE)
    b += label_box(75, 315, 295, 74, 'Cached KV for P')
    b += label_box(425, 315, 280, 74, 'New suffix A', '#f4f4f4')
    b += arrow(224, 397, 350, 457)
    b += arrow(565, 397, 442, 457)
    b += label_box(170, 470, 455, 79, 'Compute suffix activations', 'white', 28)
    b += lines(75, 606, ['Skip recomputing prefix activations.',
                          'Suffix attention still uses P’s KV.'], 28, gap=42)
    b += text(842, 271, 'Each subsequent decode step', 33, 700, color=BLUE)
    b += label_box(842, 315, 283, 74, 'New token query')
    b += label_box(1180, 315, 345, 74, 'History KV: P + A + …', '#f4f4f4', 25)
    b += arrow(984, 397, 1106, 457)
    b += arrow(1352, 397, 1210, 457)
    b += label_box(936, 470, 487, 79, 'Attention over the prior context', 'white', 26)
    b += lines(842, 606, ['Still read history KV for attention.',
                           'Still run the model to generate tokens.'], 28, gap=42)
    b += takeaway('A cache hit reduces repeated prefill work; it does not remove the history.',
                   'Queueing can still dominate TTFT, and output length still drives decode work.')
    return slide('What a prefix hit saves—and what remains', b,
        'The direct algorithmic saving is the repeated prefill computation for the reused prefix. The uncached suffix still traverses the model, including attention to previous keys and values. During autoregressive generation, each new query also attends to its valid history, including the cached prefix. Cache sharing changes where the history is stored and how many copies exist; it does not make the attention dependencies disappear. Less prefill demand can indirectly alter queueing and available capacity, so observed end-to-end effects depend on workload and scheduling. Do not predict that a 50% token-hit fraction will halve TTFT, per-token latency, or total request latency.',
        [APC], 'Prefix caching')


def question1():
    b = text(75, 190, 'Four unchanged GPU replicas; same model and arrival trace. Hypothetical measurements.', 27, color=MUTED)
    b += table(75, 230, 1450,
        ['Observed across the service', 'Load-balanced routing', 'Prefer cached prefixes'],
        [('Prompt-token hit fraction', '40%', '75%'), ('p95 TTFT', '180 ms', '420 ms')],
        ratios=[.43, .29, .28], row_h=67, size=28)
    b += text(75, 478, 'For the next request, compare two eligible replicas:', 29, 600)
    b += rect(75, 508, 686, 124, PALE)
    b += lines(100, 547, ['Replica A: prefix cached', 'Estimated queue 320 ms + prefill 40 ms'], 28, gap=43)
    b += rect(809, 508, 716, 124, '#f4f4f4')
    b += lines(834, 547, ['Replica B: prefix absent', 'Estimated queue 30 ms + prefill 140 ms'], 28, gap=43)
    b += text(75, 687, 'Why can this happen? Where would you send this request? What would you test?', 29, 600)
    b += takeaway('Question 1: does a higher cache hit rate imply a faster response?')
    return slide('Question 1: more cache hits, worse p95 TTFT', b,
        'Give the group about one minute to reason before showing the solution. These values are authored illustrative observations, not an experiment from SGLang or vLLM, and not an attributed company interview question. A and B are two eligible replicas in our four-GPU deployment; compare them for this next request while holding other overhead equal. The per-request queue and prefill estimates should not be added to global p95 values: quantiles do not add in general. The exercise asks for a causal hypothesis, a routing choice and a controlled experiment, rather than merely reciting that cache is good. The token-hit counter rose, but it measures one kind of saved work rather than user waiting time.',
        [ROUTER], 'Question 1')


def solution1():
    b = text(75, 190, 'For this request: A ≈ 320 + 40 = 360 ms; B ≈ 30 + 140 = 170 ms.', 31, 600, color=BLUE)
    b += text(75, 233, 'Compare these estimates with common network / sampling overhead omitted.', 25, color=MUTED)
    b += line(75, 269, 1525, 269)
    rows = [
        ('Explanation', ['Prefix affinity can concentrate arrivals on a few replicas.',
                         'The extra queue wait can exceed the saved prefill time.']),
        ('Routing decision', ['Choose B here. Use both expected wait and remaining prefill work.',
                              'A cache match is useful only if its saved work outweighs the extra wait.']),
        ('Controlled test', ['Replay the same trace with load-aware cache routing versus strict affinity.',
                             'Record per-replica queue delay, hit fraction, TTFT and completed requests.']),
    ]
    for i, (title, desc) in enumerate(rows):
        yy = 320 + i*133
        b += text(75, yy, title, 29, 700, color=BLUE)
        b += lines(390, yy, desc, 27, gap=39)
        if i < 2: b += line(75, yy+77, 1525, yy+77)
    b += text(75, 707, 'If queue delay is unchanged, investigate prefill timing, evictions and the request mix instead.', 25, color=MUTED)
    b += takeaway('Optimize response time under load; treat cache hits as one input to that decision.')
    return slide('Solution 1: saved work must outweigh additional waiting', b,
        'B is preferred under the stated estimates: a cache miss with a short wait can finish prefill earlier than a cache hit behind a long queue. The service-wide p95 observation alone would not prove this mechanism; request-level traces and per-replica queue measurements supply the missing evidence. Replay the identical offered workload, retain the same GPU budget and model, and compare policies. Account for cache warmness consistently and include failures and completed throughput. A load-aware policy could stop following prefix affinity when a replica becomes too busy, or distribute hot prefixes across additional replicas at the cost of duplicated KV. The exact scoring or threshold should be measured rather than asserted as universally optimal.',
        [ROUTER], 'Question 1 solution')


def get_slides():
    slides = [prefix_identity(), block_sharing(), kv_lifetime(), cache_metrics(), remaining_work(), question1(), solution1()]
    assert len(slides) == 7
    return slides
