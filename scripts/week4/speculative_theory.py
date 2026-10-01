"""Correctness, acceptance, and performance of speculative decoding."""
from .common import *
from .speculative import ACCEPT, REJECT
from .mathml import *

EXACT=('Leviathan et al., App. A.1','https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf')
CHEN_PROOF=('Chen et al., §4.2','https://arxiv.org/html/2302.01318v1#S4.SS2')
ANALYSIS=('Leviathan et al., §3','https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf')
SAMPLING=('Adapted from Leviathan et al., §2.3','https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf')


def expected_tokens(alpha,gamma):
    return sum(alpha**i for i in range(gamma+1))

def illustrative_speedup(gamma):
    return expected_tokens(.8,gamma)*10/(gamma+(10+gamma)+2)


def sampling_quiz():
    b=text(75,190,'Fix one context: p is the target distribution and q is the proposal distribution.',29)
    b+=code(75,222,1450,155,'Propose X ~ q; accept with probability min(1, p(X)/q(X)).\nIf accepted: output Y = X.\nIf rejected: independently draw Y ~ p.',27)
    b+=text(75,415,'Let Z = P(reject). Which expression equals P(Y = x) for every p and q?',30,700)
    b+=text(75,464,'Choose one answer.',25,color=MUTED)
    choices=[
        ('A',75,505,f('p')),
        ('B',75,621,par(mn(1)+mo('−')+mi('Z'))+f('q')+mo('+')+mi('Z')+f('p')),
        ('C',835,505,minimum()+mo('+')+mi('Z')+f('p')),
        ('D',835,621,minimum()+mo('+')+mi('Z')+f('q')),
    ]
    for label,x,y,formula in choices:
        b+=text(x,y+47,label+'.',33,700,color=BLUE)
        b+=mathbox(x+58,y,620,90,formula,34)
    b+=line(785,495,785,716)
    b+=takeaway('Select the general identity, even if some options coincide in special cases.', 'Both distributions are normalized over the same vocabulary. Y denotes the final emitted token.')
    return slide('Quiz: Resampling from the target after rejection',b,
        'Allow approximately 2–3 minutes for discussion and the immediate solution. There is one universally valid expression, C: Pr(Y=x)=min(p(x),q(x))+Z*p(x), where Z=Pr(reject)=1-sum_x min(p(x),q(x)). The quiz deliberately changes the correct algorithm by sampling the replacement from p. A is the desired target law but not the law of this altered procedure in general. B mistakenly uses q as the conditional law of accepted proposals; accepting with token-dependent probabilities changes that law. C adds the accepted joint mass and the replacement joint mass. D uses q for the replacement, contrary to the stated procedure. Some options can agree for special distributions such as p=q, but C is the identity valid for all p,q. The acceptance ratio is evaluated only for sampled X with q(X)>0. This is an authored assessment based on Leviathan Section 2.3, not a reported company interview question.',[SAMPLING], 'Speculative decoding theory · multiple-choice quiz')


def sampling_quiz_solution():
    b=text(75,190,'Correct answer: C',32,700,color=BLUE)
    b+=text(75,255,'Accepted proposal',29,700)
    b+=mathbox(425,212,1090,100,'<mi mathvariant="normal">Pr</mi>'+par(mi('Y')+mo('=')+mi('x')+mo(',')+'<mtext>accept</mtext>')+mo('=')+f('q')+'<mi mathvariant="normal">min</mi>'+par(mn(1)+mo(',')+frac(f('p'),f('q')))+mo('=')+minimum(),33)
    b+=line(75,323,1525,323)
    b+=text(75,379,'Replacement',29,700)
    b+=mathbox(425,338,1090,94,'<mi mathvariant="normal">Pr</mi>'+par(mi('Y')+mo('=')+mi('x')+mo(',')+'<mtext>reject</mtext>')+mo('=')+mi('Z')+f('p'),33)
    b+=line(75,444,1525,444)
    b+=mathbox(75,461,1450,100,'<mi mathvariant="normal">Pr</mi>'+par(mi('Y')+mo('=')+mi('x'))+mo('=')+minimum()+mo('+')+mi('Z')+f('p'),39)
    b+=text(75,601,'B incorrectly assumes that accepted proposals still follow q.',29,700,color=BLUE)
    b+=text(75,654,'Example: p = (0.20, 0.50, 0.20, 0.10), q = (0.60, 0.20, 0.10, 0.10), Z = 0.40.',27)
    b+=text(75,705,'The altered algorithm outputs (0.28, 0.40, 0.18, 0.14), which differs from p.',28)
    b+=takeaway('The residual correction supplies the missing target mass after acceptance.', 'The next slide proves that replacing Zp(x) with [p(x) − q(x)]₊ restores the target distribution.')
    return slide('Quiz solution: account for both output paths',b,
        'C follows by summing the joint probabilities of two disjoint events: emitting x from an accepted proposal, and emitting x from the replacement branch. The former is q(x)*min(1,p(x)/q(x))=min(p(x),q(x)); the latter is Z*p(x) because the replacement is an independent draw from p and rejection occurs with probability Z. B is a common misuse of a mixture: the accepted component is not q. For 0<Z<1, its conditional law is min(p,q)/(1-Z). If Z=0 the branch weights are degenerate and the same unconditional expression still applies. The displayed four-token counterexample has accepted mass (0.20,0.20,0.10,0.10), total acceptance 0.60 and rejection 0.40. Adding Z*p gives (0.28,0.40,0.18,0.14), not p. A therefore fails in general; D is the law of a different rule that resamples from q. Correct residual resampling replaces Z*p by max(p-q,0), yielding min(p,q)+max(p-q,0)=p. The following proof derives the residual normalizer. Equal distributions and disjoint-support cases can make several distractors coincide numerically with C; the question explicitly asks for the identity valid for arbitrary distributions.',[SAMPLING,EXACT], 'Speculative decoding theory · quiz solution')


def exactness():
    b=text(75,187,'Fix one context. p(x) is the target probability; q(x) is the draft probability.',29)
    b+=text(75,224,'Both distributions use the same vocabulary. [z]₊ means max(z, 0).',26,color=MUTED)
    b+=text(75,277,'1. Probability mass from accepting a draft token',30,700,color=BLUE)
    b+=mathbox(105,288,1390,100,f('q')+mo('·')+'<mi mathvariant="normal">min</mi>'+par(mn(1)+mo(',')+frac(f('p'),f('q')))+mo('=')+minimum(),36)
    b+=line(75,381,1525,381)
    b+=text(75,423,'2. On rejection, sample from the missing target mass',30,700,color=BLUE)
    b+=mathbox(105,435,1390,115,mi('Z')+mo('=')+'<mi mathvariant="normal">Pr</mi>'+par('<mtext>reject</mtext>')+mo('=')+summand('y',positive('y'))+'<mspace width="1.4em"/>'+f('r')+mo('=')+frac(positive(),mi('Z')),35)
    b+=line(75,546,1525,546)
    b+=text(75,588,'3. Add the accepted and corrected paths',30,700,color=BLUE)
    b+=mathbox(105,608,1390,73,'<mi mathvariant="normal">Pr</mi>'+par(mi('Y')+mo('=')+mi('x'))+mo('=')+minimum()+mo('+')+positive()+mo('=')+f('p'),37)
    b+=text(75,716,'Y is the emitted token. If Z = 0, every proposal passes and no correction is needed.',26,color=MUTED)
    b+=takeaway('Repeating this rule at each context preserves the target sequence distribution.', 'This is distributional equality. The same random seed need not produce the same sample sequence.')
    return slide('Why speculative sampling is exact',b,
        'This proof completes the correction motivated by the multiple-choice quiz. Fix a history and let p and q be normalized target and proposal distributions over a common vocabulary, after any intended sampling transformations. Propose X from q and accept with probability min(1,p(X)/q(X)). For any vocabulary item x, the joint probability of proposing and accepting x is min(p(x),q(x)). Its sum is the total acceptance probability. Since p and q each sum to one, the omitted target mass Z=sum_x max(p(x)-q(x),0) equals the rejection probability 1-sum_x min(p(x),q(x)). On rejection draw Y from r(x)=max(p(x)-q(x),0)/Z. This contributes unconditional mass Z*r(x), exactly filling the missing target mass. Thus min(p,q)+max(p-q,0)=p pointwise. When q(x)=0 the algorithm cannot propose x, so it never evaluates that ratio for such an x; positive target mass there comes through the residual. When Z=0, the rejection branch has zero probability. At successive positions, use the distributions for the actual accepted history and discard the invalid suffix after a rejection. This yields the target autoregressive joint distribution by the chain rule. The proof concerns the ideal probability algorithm; floating-point implementations can differ numerically. Chen uses the opposite p/q naming; this deck consistently uses p for target and q for draft.',[EXACT,CHEN_PROOF], 'Speculative decoding theory · 8–10 min')


def overlap():
    b=text(75,190,'α is the probability that one draft proposal is accepted, at a fixed context.',29)
    b+=text(75,254,'A two-token example',30,700,color=BLUE)
    b+=table(75,280,650,['Token','Draft q','Target p','Kept mass'],[
        ['A','0.8','0.6','0.6'],['B','0.2','0.4','0.2'],['Total','1.0','1.0','0.8'],
    ],ratios=[.2,.25,.25,.3],row_h=67,size=27)
    b+=line(770,249,770,558)
    b+=mathbox(820,269,690,90,mi('α')+mo('=')+summand('x',minimum()),37)
    b+=mathbox(820,380,690,110,mo('=')+mn(1)+mo('−')+frac(mn(1),mn(2))+summand('x',row('<mo fence="true" stretchy="false">|</mo>'+f('p')+mo('−')+f('q')+'<mo fence="true" stretchy="false">|</mo>')),35)
    b+=text(835,528,'α = 1 − TV(p, q)',35,700,color=BLUE)
    b+=text(75,612,'Here: TV = 0.2, so α = 0.8.',30,700)
    b+=rect(75,640,1160,65,ACCEPT)+rect(1235,640,290,65,REJECT)
    b+=text(655,684,'Accepted: 80%',29,700,anchor='middle')+text(1380,684,'Rejected: 20%',27,700,anchor='middle')
    b+=takeaway('More probability overlap means more draft tokens survive verification.', 'TV is total variation distance. Distribution overlap measures acceptance, not the draft model’s runtime.')
    return slide('Acceptance measures distribution overlap',b,
        'For a fixed context, sum the accepted mass min(p(x),q(x)) over the vocabulary. Using min(p,q)=(p+q-|p-q|)/2 and the fact that both distributions sum to one, alpha=1-(1/2)sum|p-q|. The subtracted term is total variation distance. In our example TV=(|0.6-0.8|+|0.4-0.2|)/2=0.2 and total acceptance is 0.8. Distinguish this 80% overall acceptance from the 75% acceptance conditional on having proposed A. Perfectly equal distributions give alpha=1; disjoint supports give alpha=0. This is a fixed-context result for stochastic speculative sampling, not simply the fraction of matching argmax predictions. Real histories give different acceptance probabilities. A draft with greater overlap may also be more expensive, so overlap alone does not determine speedup. For one-token exact couplings, this overlap is also the largest possible probability that a q-distributed proposal and p-distributed output agree; no mass at x can be shared beyond min(p(x),q(x)).',[ANALYSIS], 'Speculative decoding theory')


def progress():
    b=text(75,185,'γ = draft length. N = emitted tokens per round, including one correction or bonus.',28)
    exact=expectation()+mo('=')+mn(1)+mo('+')+finite_sum('i',1,mi('γ'),'<mi mathvariant="normal">Pr</mi>'+par('<mtext>first</mtext><mspace width="0.25em"/>'+mi('i')+'<mspace width="0.25em"/><mtext>proposals all accepted</mtext>'))
    b+=mathbox(75,210,1450,109,exact,34)
    b+=text(75,344,'Model assumption: each next proposal passes with probability α, given earlier ones passed.',28,700,color=BLUE)
    b+=text(75,392,'Example: α = 0.8, γ = 4',27,700)
    base=622
    for i in range(5):
        xx=113+i*135;prob=.8**i;h=160*prob
        b+=rect(xx,base-h,86,h,PALE)
        b+=text(xx+43,base-h-14,format(prob,'.4f').rstrip('0').rstrip('.'),25,600,anchor='middle')
        b+=text(xx+43,657,str(i+1),26,anchor='middle')
    b+=line(88,base,792,base)+text(433,702,'Output position (bar = probability it exists)',24,color=MUTED,anchor='middle')
    b+=line(835,398,835,704)
    geometric=expectation()+mo('=')+finite_sum('i',0,mi('γ'),sup(mi('α'),mi('i')))+mo('=')+frac(mn(1)+mo('−')+sup(mi('α'),mi('γ')+mo('+')+mn(1)),mn(1)+mo('−')+mi('α'))
    b+=mathbox(886,405,630,112,geometric,34)
    b+=text(886,552,'1 + 0.8 + 0.64 + 0.512 + 0.4096',27)
    b+=text(886,608,'E[N] = 3.3616 tokens',35,700,color=BLUE)
    b+=text(886,660,'If α = 1, every round emits γ + 1 tokens.',26)
    b+=takeaway('The expected progress is the sum of the probabilities of reaching each output.', 'Both formulas exclude EOS and output limits. The geometric form assumes constant conditional acceptance.')
    return slide('Expected tokens per verification round',b,
        'Let A be the consecutive accepted-prefix length among gamma proposals. N=A+1 includes one correction on rejection or one bonus when all proposals pass. This counts emitted tokens before EOS or a maximum-output cutoff. The tail-sum identity for a positive integer N gives E[N]=sum_{k=1}^{gamma+1}P(N>=k)=1+sum_{i=1}^{gamma}P(A>=i). No independence assumption is needed for this first formula. If P(proposal i accepted | proposals 1 through i-1 accepted) equals the same alpha at every reached depth, P(A>=i)=alpha^i and the finite geometric series follows. This is a model assumption, not a consequence of knowing one global average acceptance rate. At alpha=1 use the continuous limit gamma+1; alpha=0 gives one output per round. The chart displays the probability that each output position exists: 1,0.8,0.64,0.512,0.4096. Their sum is 3.3616, not 1+4*0.8=4.2, because a later accepted draft token is useful only if all earlier proposals survive. Actual context- and depth-dependent survival rates can be inserted directly into the general identity. A single round emits an integer; this fractional result is the average over many rounds. The paper derives the same geometric model using an i.i.d. acceptance assumption.',[ANALYSIS], 'Speculative decoding theory')


def speedup():
    b=text(75,188,'Compare average time per emitted token at a fixed workload.',29)
    td=mi('D');tv=mi('V')+par(mi('γ'));to=mi('O')
    b+=mathbox(75,207,1450,103,mi('S')+par(mi('γ'))+mo('≈')+frac(expectation()+mi('T'),mi('γ')+td+mo('+')+tv+mo('+')+to),38)
    b+=text(75,332,'T: one target step. D: one draft step. V(γ): verification. O: other round overhead.',27)
    # An analytic curve from the explicitly stated toy cost model, not measured performance.
    x0,x1,y0,y1=146,833,646,430
    def xy(g,s):return x0+(g-1)/11*(x1-x0),y0-(s-.8)/1.0*(y0-y1)
    for val in [1.0,1.4,1.8]:
        yy=xy(1,val)[1];b+=line(x0,yy,x1,yy,'#d9dfe3',1, '5 5' if val==1 else '')
        b+=text(x0-18,yy+8,f'{val:.1f}×',24,anchor='end',color=MUTED)
    b+=line(x0,y1-5,x0,y0)+line(x0,y0,x1,y0)
    points=[xy(g,illustrative_speedup(g)) for g in range(1,13)]
    b+='<polyline points="'+' '.join(f'{x:.3f},{y:.3f}' for x,y in points)+'" fill="none" stroke="'+BLUE+'" stroke-width="3"/>'
    for j,(x,y) in enumerate(points,1):
        b+=f'<circle cx="{x}" cy="{y}" r="{7 if j==4 else 4}" fill="{TEAL if j==4 else BLUE}"/>'
        if j in [1,2,4,6,8,10,12]:b+=text(x,678,j,24,anchor='middle')
    bx,by=xy(4,illustrative_speedup(4))
    b+=text(bx,by-22,'1.68×',27,700,color=TEAL,anchor='middle')
    b+=text(150,398,'Expected speedup',27,700,color=BLUE)+text(490,721,'Draft length γ',25,anchor='middle')
    b+=line(885,383,885,709)
    b+=text(935,400,'Illustrative cost model',29,700,color=BLUE)
    b+=lines(935,451,['α = 0.8, T = 10 ms, D = 1 ms', 'V(γ) = (10 + γ) ms; O = 2 ms'],26,gap=41)
    b+=text(935,563,'Best here: γ = 4',32,700,color=TEAL)
    b+=lines(935,609,['E[N] ≈ 3.36, round time = 20 ms', 'More proposals have diminishing returns.'],26,gap=43)
    b+=takeaway('For α < 1, expected progress saturates while drafting keeps adding cost.', 'Analytic illustration, not a GPU benchmark. Measure verification time at the intended context and concurrency.')
    return slide('Expected speedup and draft length',b,
        'Use a stationary fixed-workload approximation and enough generation rounds for average costs to be meaningful. In the displayed cost model a round costs C(gamma)=gamma*D+V(gamma)+O. Its average time per output token is C/E[N], versus T for target-only decoding, so speedup is T*E[N]/C. If round cost also varies, use E[C]/E[N] for the long-run average time per token, not E[C/N] or the average of per-round speedups. For variable contexts and system load, measure aggregate elapsed time and emitted tokens against a matched baseline. The illustrative curve assumes alpha=0.8, T=10 ms, D=1 ms, V(gamma)=10+gamma ms and O=2 ms. At gamma=4, E[N]=3.3616 and C=20 ms, giving speedup 1.6808. Integer gamma=4 maximizes this example over 1 through 12. The earlier slider chose exactly two accepted tokens in one round, hence 1.5x; this theory uses an average over random acceptance, hence 1.68x. The classic simplified formula follows if verification takes exactly one target step and overhead is zero: S=(1-alpha^(gamma+1))/((1-alpha)*(1+gamma*c)), c=D/T. For gamma=1 this is (1+alpha)/(1+c), so alpha>c is the break-even condition under those simplified assumptions only. For alpha<1, E[N] approaches 1/(1-alpha), but positive draft cost grows with gamma. The marginal benefit of one more proposal is alpha^(gamma+1); increasing gamma helps only when this extra progress per extra millisecond exceeds current E[N]/C. Context length, KV traffic, concurrency, draft memory, synchronization and sampling implementation affect the measured costs. Faster generation for one stream does not by itself imply higher throughput or better latency under load.',[ANALYSIS], 'Speculative decoding theory')


def get_slides():
    return [sampling_quiz(),sampling_quiz_solution(),exactness(),overlap(),progress(),speedup()]
