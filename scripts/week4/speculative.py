"""Classical draft-model speculative decoding: algorithm and implementation."""
from .common import *
from .mathml import *
SPEC=('Leviathan et al., §2–3','https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf')
CHEN=('Chen et al., Algorithm 2','https://arxiv.org/html/2302.01318v1')
HF=('Transformers assisted decoding','https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py')
VLLM=('vLLM speculative decoding','https://docs.vllm.ai/en/latest/features/speculative_decoding/')
ACCEPT='#e5f1ed'
REJECT='#fae9df'


def overview():
    b=text(75,190,'Target p defines the desired distribution; draft q is cheaper. h is the current prefix.',29)
    b+=text(75,252,'1. Draft γ tokens sequentially',31,700,color=BLUE)
    b+=mathbox(105,267,1390,75,sub('d',mi('i'))+mo('∼')+mi('q')+par(mo('·')+mo('|')+mi('h')+mo(',')+sub('d',mo('<')+mi('i')))+'<mspace width="1.5em"/>'+mi('i')+mo('=')+mn(1)+mo(',')+mo('…')+mo(',')+mi('γ'),34)
    b+=line(75,352,1525,352)
    b+=text(75,397,'2. Evaluate the target on the known draft sequence in one causal forward pass',31,700,color=BLUE)
    b+=mathbox(105,414,1390,76,sub('p',mi('i'))+par(mo('·'))+mo('=')+mi('p')+par(mo('·')+mo('|')+mi('h')+mo(',')+sub('d',mo('<')+mi('i')))+'<mspace width="1.5em"/>'+mi('i')+mo('=')+mn(1)+mo(',')+mo('…')+mo(',')+mi('γ')+mo('+')+mn(1),34)
    b+=text(75,522,'Known input positions run together within each layer; the causal mask prevents look-ahead.',27)
    b+=line(75,545,1525,545)
    b+=text(75,589,'3. Commit an accepted prefix, then continue from the corrected history',31,700,color=BLUE)
    b+=text(75,637,'First rejection: replace that token and discard later proposals. All accepted: append a bonus.',27)
    b+=text(75,684,'One round emits 1 to γ + 1 tokens. The acceptance and correction rule is on the next slide.',27)
    b+=takeaway('Potential gain: fewer serial target calls and more reuse of target weights.', 'd<i denotes earlier draft tokens. Verification costs extra work; speedup depends on acceptance and workload.')
    return slide('Speculative decoding: draft, verify, commit',b,
        'h is the committed history and d_<i is the draft prefix before position i. The classical draft model samples gamma proposals autoregressively. Since those candidates are now known, the target can score all candidate-conditioned contexts together using causal attention, as in a short prefill. This computes probability distributions, not independently sampled target tokens or argmax matches. In a common KV convention, the final committed token has not yet been processed; feed it plus the draft tokens to obtain gamma candidate scores and a bonus distribution. Equivalently, an already computed first-position distribution can be reused. Each proposal and its target score condition on the same history. Verification is one model call, not one CUDA kernel, and layers remain sequential. At the first rejection the subsequent draft history is invalid, so discard that suffix. If every proposal is accepted, sample one bonus from the final target distribution. The exact stochastic acceptance and residual correction appear next. The range of emitted tokens ignores EOS and output limits. Lower-batch target decode can benefit because multi-position verification amortizes weight movement; extra draft and verification work can outweigh those savings.',[SPEC,CHEN], 'Speculative decoding · overview')


def sampling_algorithm():
    b=text(75,188,'At each position, p and q are the target and draft distributions at the same prefix.',29)
    b+=text(75,241,'1. Propose',30,700,color=BLUE)
    b+=mathbox(325,214,1175,89,mi('X')+mo('∼')+mi('q')+'<mspace width="2em"/>'+mi('U')+mo('∼')+'<mi mathvariant="normal">Uniform</mi>'+par(mn(0)+mo(',')+mn(1)),35)
    b+=line(75,305,1525,305)
    b+=text(75,356,'2. Accept',30,700,color=BLUE)
    b+=mathbox(325,319,1175,98,mi('U')+mo('≤')+'<mi mathvariant="normal">min</mi>'+par(mn(1)+mo(',')+frac(f('p','X'),f('q','X')))+mo('⇒')+mi('Y')+mo('=')+mi('X'),36)
    b+=line(75,420,1525,420)
    b+=text(75,475,'3. Otherwise',30,700,color=BLUE)
    b+=mathbox(325,435,1175,123,mi('Y')+mo('∼')+mi('r')+'<mspace width="1.2em"/>'+f('r')+mo('=')+frac(positive(),summand('v',positive('v'))),36)
    b+=text(325,588,'[z]₊ = max(z, 0). If p = q, rejection never occurs.',27,color=MUTED)
    b+=line(75,614,1525,614)
    b+=text(75,656,'Apply in order: stop at the first rejection and discard the remaining draft suffix.',28)
    b+=text(75,703,'If all γ proposals pass, sample a bonus from p(· | h, d₁, …, dγ).',28)
    b+=takeaway('The emitted sequence has exactly the target autoregressive distribution.', 'Use the actual sampling distributions after temperature / truncation. Equality is distributional, not seed-by-seed.')
    return slide('Exact speculative sampling',b,
        'This is the stochastic sampling algorithm in Leviathan et al. Section 2.3, written in native notation rather than using an argmax illustration. Fix one history, draw X from q and an independent U uniformly on (0,1), then accept if U is at most min(1,p(X)/q(X)). A proposal necessarily has q(X)>0, so the evaluated ratio is defined. On rejection draw a new token from the normalized positive residual p-q. The residual normalizer equals the overall rejection probability; if it is zero, p=q and the correction branch is never reached. Tokens outside the proposal support can still be generated by the residual. For a speculative sequence apply this rule in order to each reached candidate, using p_i and q_i conditioned on the identical prefix. Stop after the first rejection and discard later draft candidates and invalid scores. If all candidates pass, sample the bonus directly from the target distribution after the full accepted draft. Ordinary probabilities after temperature, top-p or other intended sampling transformations are required; raw logits cannot be used in the ratio. The exactness claim is about the ideal probability algorithm. It does not require the same sampled text for a fixed RNG seed or numerically identical floating-point implementations. The proof follows the multiple-choice quiz.',[('Leviathan et al., §2.3 / Algorithm 1',SPEC[1]),CHEN], 'Speculative decoding · algorithm')


def cache():
    b=text(75,192,'Verification creates provisional target KV; h is the committed prefix.',31,700)
    b+=line(790,254,790,718)
    for x,label in [(75,'First rejection'),(845,'All four proposals accepted')]:b+=text(x,276,label,32,700,color=BLUE)
    b+=text(75,327,'Draft: The · cat · sat · quietly',26)
    b+=text(845,327,'Alternative draft: The · cat · slept · soundly',26)
    for x,ws in [(75,['h','The','cat','sat','quietly']),(845,['h','The','cat','slept','soundly'])]:
        b+=text(x,390,'Target KV after verification',26,700)
        for j,w in enumerate(ws):
            xx=x+j*134
            invalid=x==75 and j>=3
            b+=label_box(xx,421,121,64,w,REJECT if invalid else ACCEPT,24)
            if invalid:b+=line(xx+8,430,xx+113,476,WARM,2)
        b+=text(x,536,'Keep h + The + cat' if x==75 else 'Keep h + all four draft tokens',28,700,color=TEAL)
    b+=lines(75,593,['Emit correction: “slept”', 'Its target KV still needs to be computed.', 'Discard KV for “sat” and “quietly”.'],27,gap=42)
    b+=lines(845,593,['Emit a bonus from p(next | h + draft).', 'Example: “.” — its target KV is pending.', 'Four accepted + one bonus = five outputs.'],27,gap=42)
    b+=takeaway('Emitting a token and computing that token’s KV are different events.', 'h is now fully cached. The correction/bonus is an input next; draft and target have separate KV caches.')
    return slide('After verification: rollback, continuation, and the bonus',b,
        'Before verification the target cache covers h except its last token u. Verification evaluates u and all four draft input positions, so h is now fully cached. When sat is rejected, the target KV for The and cat is valid because their histories are unchanged. KV for sat and quietly belongs to an invalid branch and must no longer be visible to attention. The replacement slept is selected from a distribution already computed at the previous position; slept itself has not been fed through the target, so its KV is pending until continuation. In the all-accepted alternative, the final draft position provides the distribution for one bonus token. That bonus is also not yet an evaluated target input. In practice engines track valid lengths or block mappings; rollback does not necessarily erase bytes. The draft model has its own weights and KV state, which must be synchronized with the committed sequence; accepted draft KV cannot be substituted for target KV. We omit EOS and length-limit termination, which can truncate the emitted block.',[SPEC,HF], 'Speculative decoding')


def cost():
    b=text(75,191,'Toy timings: four draft steps × 1 ms; baseline target decoding = 10 ms/token.',28,color=MUTED)
    b+=text(75,242,'Our example accepts 2 proposals + 1 correction: compare time for 3 output tokens.',30,700,attrs='data-spec-cost="description"')
    b+=text(75,339,'Target only',29,700)+text(75,441,'Speculative',29,700)
    b+='<rect x="390" y="295" width="600" height="65" fill="'+PALE+'" stroke="'+LINE+'" data-spec-cost="baseline-bar"/>'
    b+=text(1015,338,'30 ms',29,700,attrs='data-spec-cost="baseline-label"')
    b+=rect(390,397,80,65,ACCEPT)+text(430,438,'4 ms',22,600,anchor='middle')
    b+='<rect x="470" y="397" width="280" height="65" fill="'+PALE+'" stroke="'+LINE+'" data-spec-cost="verify-bar"/>'
    b+=text(610,438,'14 ms',26,600,anchor='middle',attrs='data-spec-cost="verify-label"')
    b+='<rect x="750" y="397" width="40" height="65" fill="#dce1e4" stroke="'+LINE+'" data-spec-cost="overhead-bar"/>'
    b+=text(815,438,'20 ms total',28,700,attrs='data-spec-cost="total-label"')
    b+=text(390,501,'Draft',23,color=TEAL)+text(565,501,'Target verification',23,color=BLUE)+text(970,501,'+ 2 ms bookkeeping',23,color=MUTED)
    b+=text(75,559,'3 × 10 / (4 + 14 + 2) = 1.50× speedup',32,700,color=BLUE,attrs='data-spec-cost="ratio"')
    b+='''<foreignObject x="75" y="584" width="1450" height="142"><div xmlns="http://www.w3.org/1999/xhtml" class="spec-cost-controls" data-deck-ignore-keys="true"><label><span>Accepted prefix: <output id="spec-accepted-value">2</output> / 4</span><input id="spec-accepted" type="range" min="0" max="4" value="2" step="1" aria-label="Number of accepted draft tokens"/></label><label><span>Verification: <output id="spec-verify-value">14</output> ms</span><input id="spec-verify" type="range" min="10" max="34" value="14" step="2" aria-label="Target verification milliseconds"/></label><div id="spec-cost-explanation" aria-live="polite">2 accepted + 1 correction = 3 emitted tokens. More guesses are not automatically faster.</div></div></foreignObject>'''
    b+=takeaway('Benefit depends on accepted progress per unit time, not acceptance alone.', 'Higher load can increase verification cost and memory pressure. Measure serving latency and throughput again.')
    return slide('When does the extra work pay off?',b,
        'Start at two accepted proposals, matching the rollback example. Four sequential draft steps cost 4 ms, verification costs 14 ms and bookkeeping costs 2 ms, for 20 ms to emit three tokens; ordinary target decoding would take 30 ms. Move accepted-prefix length to zero: the draft round still costs 20 ms but emits only the correction, versus 10 ms normally. Move to four: all four proposals plus a bonus yield five outputs. Then increase verification time: good acceptance can still fail to pay off. For this illustrative single round, emitted tokens = accepted prefix length + one correction or bonus; EOS and output limits are excluded. The sliders vary independent assumptions and are not hardware predictions. Across actual rounds use total elapsed time divided by total emitted tokens, rather than averaging speedup ratios. A separate draft also uses weight memory and its own KV cache; a configuration that speeds one stream can reduce capacity or throughput under concurrency. Verification time need not equal one-token target latency.',[SPEC,VLLM], 'Speculative decoding')


def get_slides():
    return [overview(),sampling_algorithm(),cache(),cost()]
