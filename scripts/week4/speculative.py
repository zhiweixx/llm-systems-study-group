"""Six visual lessons on classical draft-model speculative decoding."""
from .common import *
SPEC=('Leviathan et al., §2–3','https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf')
CHEN=('Chen et al., Algorithm 1','https://arxiv.org/html/2302.01318v1')
HF=('Transformers assisted decoding','https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py')
VLLM=('vLLM speculative decoding','https://docs.vllm.ai/en/latest/features/speculative_decoding/')
ACCEPT='#e5f1ed'
REJECT='#fae9df'
GREY='#f1f3f4'


def token(x,y,w,label,fill=PALE,small=None,cross=False):
    b=label_box(x,y,w,57,label,fill,27)
    if small:b+=text(x+w/2,y+84,small,21,color=MUTED,anchor='middle')
    if cross:b+=line(x+8,y+8,x+w-8,y+49,WARM,2)
    return b


def motivation():
    b=text(75,191,'Target = the model whose outputs we want. Draft = a cheaper proposal model.',29)
    b+=text(75,252,'Ordinary decoding: three output tokens require three serial target calls.',31,700)
    for j in range(3):
        x=360+j*370
        b+=label_box(x,289,295,77,f'Target call {j+1}',PALE,29)
        b+=text(x+147,407,f'emit token {j+1}',25,anchor='middle',color=BLUE)
        if j<2:b+=arrow(x+300,327,x+360,327)
    b+=text(75,490,'Speculation: draft several tokens cheaply, then check them in one target call.',30,700)
    for j in range(4):
        x=360+j*91
        b+=label_box(x,530,74,58,f'd{j+1}',ACCEPT,26)
        if j<3:b+=arrow(x+75,559,x+87,559)
    b+=arrow(727,559,800,559)+label_box(807,521,350,78,'One target verification',PALE,27)
    b+=arrow(1163,559,1224,559)+lines(1250,548,['2 accepted','+ 1 correction'],26,gap=39,color=BLUE)
    b+=text(75,668,'A short multi-token check can reuse target weights across several positions.',29,700,color=BLUE)
    b+=takeaway('Spend cheap draft work to reduce serial calls to the expensive model.', 'Potential benefit: low-batch, bandwidth-limited decode. Verification still costs time; widths are schematic.')
    return slide('Speculative decoding: fewer serial target calls',b,
        'Begin with the bottleneck from Week 2: ordinary autoregressive decoding must choose a token before the next unknown input can be processed. At low batch sizes, reading target-model weights can dominate, leaving arithmetic capacity for checking several known positions together. A smaller proposal model can exploit that headroom. The lower diagram previews our later four-proposal example: two proposals survive and a target correction provides the third output token. The target is not replaced by the draft. Verification is one model forward call, not one CUDA kernel. The target still reads context KV and performs extra arithmetic. This schematic makes no measured timing claim. We teach the classical separate-draft-model algorithm; other proposal mechanisms exist but are outside this six-slide lesson.',[SPEC,VLLM], 'Speculative decoding · 12–15 min')


def verification():
    b=text(75,193,'Verification inputs: u · The · cat · sat · quietly',34,700,color=BLUE)
    b+=text(75,239,'u is the last committed token. Earlier prefix KV is already cached.',29)
    labels=['u','The','cat','sat','quietly']
    b+=text(337,298,'New inputs visible at each position',24,700,color=BLUE)
    for c,label in enumerate(labels):b+=text(380+c*98,337,label,23,600,anchor='middle')
    for r in range(5):
        y=358+r*54
        b+=text(75,y+35,'Check “'+labels[r+1]+'”' if r<4 else 'Bonus distribution',26,700 if r==4 else 400)
        for c in range(5):
            x=337+c*98
            b+=rect(x,y,86,45,PALE if c<=r else 'white')
            if c<=r:b+=text(x+43,y+31,'✓',25,600,color=BLUE,anchor='middle')
    b+=line(864,305,864,632)
    b+=lines(920,345,['Each position predicts the NEXT token:', 'u → distribution for “The”', 'The → distribution for “cat”', '…', 'quietly → bonus distribution'],27,gap=47)
    b+=text(920,620,'All rows also attend to the cached history.',25,color=MUTED)
    b+=text(75,693,'The candidates are known inputs: process their positions together within each layer.',29,700)
    b+=takeaway('One causal call scores five contexts; it does not generate five independent guesses.', 'As in prefill, future tokens are masked. Scores after a rejected proposal use a history we will discard.')
    return slide('Why can the target verify several tokens at once?',b,
        'Let the committed prefix P end in u. The target cache covers P except u, which has been emitted but not yet processed. This is the common generation-loop convention. One target forward processes u and all four draft inputs. Its five output positions score d1, d2, d3, d4 and a bonus. Each table row shows visible new inputs; all also attend to cached history before u. Logits from u predict d1; logits from The predict d2, and so forth. The causal mask prevents future input leakage. Matrix operations across known input positions run together within each layer, like a short prefill. Layers still depend on previous layers. Candidate positions are not independent future generations: the fourth score assumes the draft prefix ending in sat. The draft itself produces its proposals sequentially in this classical scheme. If the first next-token distribution is already stored at a prefix boundary, an equivalent implementation can reuse it; the context semantics are unchanged.',[SPEC,HF], 'Speculative decoding')



def greedy():
    captions=[
        '1. The draft proposes four tokens, one after another.',
        '2. One target call scores the draft-conditioned contexts.',
        '3. The first two proposals match the target’s greedy choices.',
        '4. Reject “sat”. Even a later matching token cannot be kept.',
        '5. Commit “The cat slept”; continue from this corrected prefix.',
    ]
    frames=[]
    for f in range(5):
        b=text(75,188,'Greedy example · each word is one toy token; prefix P is already committed.',27,color=MUTED)
        b+=text(75,246,captions[f],32,700,color=BLUE)
        b+=text(75,331,'Draft proposes',29,700)
        b+=text(75,452,'Target argmax',29,700)
        b+=text(75,593,'Committed output',29,700)
        draft=['The','cat','sat','quietly'];target=['The','cat','slept','quietly']
        for j in range(4):
            x=405+j*273
            fill=ACCEPT if f>=2 and j<2 else REJECT if f>=3 and j==2 else GREY if f>=3 and j==3 else PALE
            b+=token(x,293,231,draft[j],fill,cross=f>=3 and j>=2)
            value=target[j] if f>=1 else '?'
            b+=token(x,414,231,value,ACCEPT if f>=2 and j<2 else PALE if j<3 else GREY)
            if f>=1:b+=arrow(x+115,359,x+115,407)
            if f>=3 and j==3:b+=text(x+115,498,'conditioned on “sat”',22,color=WARM,anchor='middle')
        if f<4:
            b+=text(405,593,'P  ·  no new tokens committed yet',29,color=MUTED)
        else:
            for j,w in enumerate(['The','cat','slept']):b+=token(405+j*273,554,231,w,ACCEPT if j<2 else PALE)
            b+=text(1224,592,'3 new tokens',28,700,color=BLUE)
        if f==0:b+=text(75,645,'Draft suggestions are provisional, not streamed to the user.',25,color=MUTED)
        elif f==1:b+=text(75,645,'We now have target predictions; deciding what survives is the next step.',25,color=MUTED)
        elif f==2:b+=text(75,645,'Keep only a consecutive accepted prefix. The next proposal must be checked.',25,color=MUTED)
        else:b+=text(75,645,'After replacing “sat”, the score for “quietly” used the wrong history.',25,color=WARM)
        frames.append(b)
    b=steps('spec-greedy',frames)
    b+=takeaway('Greedy rule: accept matches until the first mismatch, then use the target’s token.', 'The stochastic sampling rule is different; we will derive it using a probability example.')
    return slide('A complete round: draft → verify → accept or reject',b,
        'Click through all five stages. The fourth target prediction deliberately matches quietly: this demonstrates that agreement alone is insufficient after an earlier rejection. That prediction assumed The cat sat, whereas the committed result is The cat slept. Therefore both the rejected token and its continuation must be discarded. The two accepted tokens plus the correction produce three output tokens from one target verification round. Greedy equivalence assumes the same target computation and tie-breaking; floating-point variations between kernel shapes can matter in implementations. Here the target remains the authority for every committed position. The sentence is an authored toy vocabulary, not a measured model output. In streaming, provisional guesses must not be presented as accepted output.',[SPEC], 'Speculative decoding')


def cache():
    b=text(75,192,'Verification creates provisional target KV. Keep only the accepted path.',31,700)
    b+=line(790,254,790,718)
    for x,label in [(75,'First mismatch'),(845,'All four proposals accepted')]:b+=text(x,276,label,32,700,color=BLUE)
    b+=text(75,327,'Draft: The · cat · sat · quietly',26)
    b+=text(845,327,'Alternative draft: The · cat · slept · soundly',26)
    for x,ws in [(75,['P','The','cat','sat','quietly']),(845,['P','The','cat','slept','soundly'])]:
        b+=text(x,390,'Target KV after verification',26,700)
        for j,w in enumerate(ws):
            xx=x+j*134
            invalid=x==75 and j>=3
            b+=label_box(xx,421,121,64,w,REJECT if invalid else ACCEPT,24)
            if invalid:b+=line(xx+8,430,xx+113,476,WARM,2)
        b+=text(x,536,'Keep P + The + cat' if x==75 else 'Keep P + all four draft tokens',28,700,color=TEAL)
    b+=lines(75,593,['Emit correction: “slept”', 'Its target KV still needs to be computed.', 'Discard KV for “sat” and “quietly”.'],27,gap=42)
    b+=lines(845,593,['Emit a bonus from p(next | P + draft).', 'Example: “.” — its target KV is pending.', 'Four accepted + one bonus = five outputs.'],27,gap=42)
    b+=takeaway('Emitting a token and computing that token’s KV are different events.', 'P is now fully cached. The correction/bonus is an input next; draft and target have separate KV caches.')
    return slide('After verification: rollback, continuation, and the bonus',b,
        'Before verification the target cache covers P except its last token u. Verification evaluates u and all four draft input positions, so P is now fully cached. When sat is rejected, the target KV for The and cat is valid because their histories are unchanged. KV for sat and quietly belongs to an invalid branch and must no longer be visible to attention. The replacement slept is selected from a distribution already computed at the previous position; slept itself has not been fed through the target, so its KV is pending until continuation. In the all-accepted alternative, the final draft position provides the distribution for one bonus token. That bonus is also not yet an evaluated target input. In practice engines track valid lengths or block mappings; rollback does not necessarily erase bytes. The draft model has its own weights and KV state, which must be synchronized with the committed sequence; accepted draft KV cannot be substituted for target KV. We omit EOS and length-limit termination, which can truncate the emitted block.',[SPEC,HF], 'Speculative decoding')


def mass_bar(x,y,segments,w=1040):
    b=''
    for fraction,label,color in segments:
        width=w*fraction;b+=rect(x,y,width,58,color)
        b+=text(x+width/2,y+38,label,27,600,anchor='middle')
        x+=width
    return b


def sampling():
    frames=[]
    for f in range(3):
        b=text(75,190,'At the same context: draft q(A)=0.8, q(B)=0.2; target p(A)=0.6, p(B)=0.4.',28)
        b+=text(75,239,['The draft proposes A too often. Copying its draws would bias the result.',
                             'Accept only 75% of proposed A; always accept proposed B.',
                             'Resample the rejected mass as B. The final probabilities match the target.'][f],29,700,color=BLUE)
        b+=text(75,308,'Draft probability',27,700)
        b+=mass_bar(420,269,[(.8,'A: 0.8',PALE),(.2,'B: 0.2',ACCEPT)])
        b+=text(75,431,'Accept / reject',27,700)
        if f>=1:b+=mass_bar(420,392,[(.6,'Keep A: 0.6',PALE),(.2,'Reject: 0.2',REJECT),(.2,'Keep B: 0.2',ACCEPT)])
        else:b+=rect(420,392,1040,58,GREY)+text(940,431,'Decide which draft samples to keep',27,color=MUTED,anchor='middle')
        b+=text(75,551,'Final probability',27,700)
        if f>=2:
            b+=mass_bar(420,512,[(.6,'A: 0.6',PALE),(.4,'B: 0.4',ACCEPT)])
            b+=arrow(1148,456,1148,506,WARM)+text(1170,490,'Replace with B',22,color=WARM)
        else:b+=rect(420,512,1040,58,GREY)+text(940,551,'After correction',27,color=MUTED,anchor='middle')
        b+=text(75,613,'Accept proposed x with min(1, p(x) / q(x)). Here: A → 0.6/0.8; B → 1.',26)
        b+=text(75,653,'If rejected, resample in proportion to max(p − q, 0): here A → 0, B → 0.2.',25)
        frames.append(b)
    b=steps('spec-sampling',frames,y=674)
    b+=takeaway('Exact speculative sampling preserves the target distribution.', 'Apply the rule along the accepted prefix; after the first rejection, discard the remaining draft.')
    return slide('Sampling: correct the draft’s probability bias',b,
        'The bars represent probability mass, not a measured set of exactly 100 draws. q proposes A with probability 0.8, and we keep each A with probability 0.75, giving accepted A mass 0.6. Proposed B is always accepted, giving mass 0.2. Rejection occurs with total probability 0.2. The positive residual p−q is 0 for A and 0.2 for B, so its normalized distribution puts probability 1 on B; adding that correction gives final mass 0.4 for B. For a proposed x, q(x)>0, so the ratio is defined. If p=q, acceptance is certain and the zero residual is never used. For a sequence, each distribution conditions on the same accepted history, and accept/reject decisions stop at the first rejection; later precomputed target scores are discarded. If all candidates pass, draw the bonus directly from the next target distribution. Use distributions after the intended temperature or other sampling transformations. Distributional equivalence does not imply identical text for every random seed. This is classical exact speculative sampling, not an approximation that silently changes the target distribution.',[CHEN,SPEC], 'Speculative decoding')


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
        'Start at two accepted proposals, matching the greedy toy. Four sequential draft steps cost 4 ms, verification costs 14 ms and bookkeeping costs 2 ms, for 20 ms to emit three tokens; ordinary target decoding would take 30 ms. Move accepted-prefix length to zero: the draft round still costs 20 ms but emits only the correction, versus 10 ms normally. Move to four: all four proposals plus a bonus yield five outputs. Then increase verification time: good acceptance can still fail to pay off. For this illustrative single round, emitted tokens = accepted prefix length + one correction or bonus; EOS and output limits are excluded. The sliders vary independent assumptions and are not hardware predictions. Across actual rounds use total elapsed time divided by total emitted tokens, rather than averaging speedup ratios. A separate draft also uses weight memory and its own KV cache; a configuration that speeds one stream can reduce capacity or throughput under concurrency. Verification time need not equal one-token target latency.',[SPEC,VLLM], 'Speculative decoding')


def get_slides():
    return [motivation(),verification(),greedy(),cache(),sampling(),cost()]
