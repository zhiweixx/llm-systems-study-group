from .common import *
SPEC=('Leviathan et al., 2023','https://proceedings.mlr.press/v202/leviathan23a.html')
CHEN=('Chen et al., 2023','https://arxiv.org/abs/2302.01318')

def get_slides():
    body=text(75,195,'Optional appendix · a greedy-decoding example',27,color=MUTED)
    body+=text(75,252,'A cheap draft proposes several tokens; the target verifies them together.',32,700)
    body+=text(75,335,'Draft',30,700,color=BLUE)
    for i,label in enumerate(['The','cat','sat','quietly']):body+=label_box(350+i*265,290,225,78,label,size=32)
    body+=text(75,463,'Target argmax',28,700,color=BLUE)
    for i,label in enumerate(['The','cat','slept','…']):body+=label_box(350+i*265,417,225,78,label,PALE if i<2 else '#f6eee6',size=32)
    for i in range(4):body+=arrow(463+i*265,370,463+i*265,413)
    body+=text(350,557,'Accept the matching prefix',29,700,color=TEAL)+text(970,557,'First mismatch',28,700,color=WARM)
    body+=text(75,629,'Commit: “The cat slept”. Discard the remaining draft continuation.',31,700)
    body+=text(75,686,'Later target predictions were conditioned on the rejected draft prefix.',28)
    body+=takeaway('One target verification can advance several output tokens.', 'Each verification position uses a causal mask; the draft supplies the candidate context.')
    out=[slide('Speculative decoding: draft, verify, and commit',body,
        'This authored example uses deterministic greedy decoding and the same tokenizer. Verification computes target logits for each proposed position in one multi-position forward call. The target prediction for the fourth position is based on the draft context ending in “sat”, so it cannot be used after we replace that token with “slept”. Keep the longest matching prefix, use the target choice at the first mismatch, and discard invalid future KV entries. If all guesses match, the verification also permits a bonus target token. This preserves greedy output with consistent numerical behavior and tie-breaking; stochastic sampling needs a different acceptance rule, shown next.',[SPEC], 'Optional · speculative decoding')]
    body=text(75,191,'Sampling requires an acceptance rule, not an argmax comparison.',32,700)
    body+=text(75,241,'One position: target distribution p; draft distribution q; candidate x sampled from q.',27)
    body+=label_box(75,281,1450,79,'Accept x with probability min(1, p(x) / q(x)).',size=34)
    body+=text(75,407,'If rejected: sample from the normalized positive part of p − q.',31,700)
    body+=table(75,446,810,['Token','Target p','Draft q','Accepted mass'],[
        ['A','0.6','0.8','0.8 × 0.75 = 0.6'],['B','0.4','0.2','0.2 × 1 = 0.2'],
    ],ratios=[.16,.22,.22,.40],row_h=66,size=25)
    body+=lines(955,478,['Rejection probability = 0.2', 'Residual mass: A = 0, B = 0.2', 'Rejected draws become B.', 'Final: A = 0.6; B = 0.4.'],26,gap=48)
    body+=takeaway('The correction recovers the target distribution.', 'For a sequence, stop at the first rejection; later draft-conditioned predictions are discarded.')
    out.append(slide('Why exact speculative sampling stays correct',body,
        'The toy two-token distribution makes the correction explicit. The draft overproposes A: accept 75% of its proposals, yielding 0.6 total probability. B proposals are always accepted, yielding 0.2. The remaining 0.2 rejection probability is assigned to B by the residual distribution, recovering the target probabilities. The rule is applied at each position conditioned on the already accepted prefix. At a rejection, sample the correction and restart from the committed prefix; if all proposals are accepted, sample a bonus token from the next target distribution. p and q must reflect the intended sampling transformations. Exactness means the same output distribution, not identical sampled text for every random seed.',[CHEN,SPEC], 'Optional · speculative decoding'))
    body=text(75,203,'A useful break-even test',33,700,color=BLUE)
    body+=text(75,277,'Draft time + verify time + extra overhead < emitted tokens × baseline step time',30,700)
    body+=text(75,333,'Count all committed tokens, including the correction or bonus token.',27,color=MUTED)
    body+=table(75,388,1450,['Illustrative round','Baseline: 10 ms/token','Speculative: 6 + 14 + 0 ms'],[
        ['4 committed tokens','40 ms','20 ms → 2× faster'],['1 committed token','10 ms','20 ms → 2× slower'],
    ],ratios=[.30,.32,.38],row_h=76,size=28)
    body+=lines(75,665,['Measure acceptance, verification cost, draft memory, and serving latency under load.'],28)
    body+=takeaway('Higher acceptance helps only if its extra work is cheap enough.', 'Single-request speedup does not guarantee better throughput or tail latency at high concurrency.')
    out.append(slide('When does speculation improve the service?',body,
        'These round times are invented to illustrate a break-even inequality, not a benchmark. Compare the same number of emitted tokens. The draft cost includes all draft steps, and verification includes target work for multiple positions; neither is free. Extra overhead is set to zero only in this toy, and must be measured in a real engine. For many rounds, divide total elapsed time by total committed tokens rather than averaging per-round ratios. A larger draft or longer lookahead may increase acceptance but also latency and memory. At high concurrency the target may already use its compute effectively, and draft/verification work can compete with other requests. Use the same fixed-budget load test as the main lesson.',[('vLLM speculative decoding','https://docs.vllm.ai/en/latest/features/speculative_decoding/')], 'Optional · speculative decoding'))
    return out
