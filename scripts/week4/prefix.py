"""Agent prefix reuse: successive calls, cache metrics, and compaction policy."""
from .common import *
from .mathml import mathbox, mi, mo, sub, frac

MANUS = ('Manus: agent context', 'https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus')
APC = ('vLLM: prefix reuse', 'https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/')
DESIGN = ('vLLM: cache identity', 'https://docs.vllm.ai/en/latest/design/prefix_caching/')
METRICS = ('vLLM: cache metrics', 'https://docs.vllm.ai/en/latest/design/metrics/#prefix-cache-metrics')
EDIT = ('Anthropic: context editing', 'https://platform.claude.com/docs/en/build-with-claude/context-editing#context-editing-and-prompt-caching')
BERKELEY = ('Berkeley L21, pp. 35–39', 'https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_21.pdf#page=35')
CACHED = '#dceaf2'
FRESH = '#f6ece2'


def segment(x, y, w, label, count, fill=CACHED, height=76):
    return rect(x,y,w,height,fill)+text(x+w/2,y+31,label,25,600,anchor='middle')+text(x+w/2,y+61,count,23,anchor='middle',color=MUTED)


def legend(x=75, y=699):
    return rect(x,y-20,25,25,CACHED)+text(x+38,y,'Reused KV',23)+rect(x+232,y-20,25,25,FRESH)+text(x+270,y,'Newly computed KV',23)


def agent_calls():
    b = text(75,192,'A coding agent is fixing one failing test. Each tool interaction triggers another model call.',29)
    b += text(75,232,'The next input includes the earlier transcript, followed by the new action and observation.',27,color=MUTED)
    b += text(75,301,'Model input',26,700,color=BLUE)
    b += text(1155,301,'What happens next',26,700,color=BLUE)
    b += text(75,369,'Call 1',28,700)
    b += segment(245,324,195,'Task + tools','4k tokens',FRESH)
    b += lines(1155,348,['Run tests;', 'receive a long log.'],26,gap=35)
    b += text(75,493,'Call 2',28,700)
    b += segment(245,448,195,'Same input','4k')
    b += segment(440,448,330,'Action + test log','20k',FRESH)
    b += lines(1155,472,['Read source files;', 'receive code excerpts.'],26,gap=35)
    b += text(75,617,'Call 3',28,700)
    b += segment(245,572,195,'Same input','4k')
    b += segment(440,572,330,'Same interaction','20k')
    b += segment(770,572,330,'Action + files','16k',FRESH)
    b += lines(1155,596,['Rerun tests;', 'append another 2k.'],26,gap=35)
    b += legend(245,705)
    b += text(1155,703,'k = 1,000 tokens',22,color=MUTED)
    b += takeaway('One agent session naturally reuses long prefixes across separate model calls.',
                   'Illustrative transcript. Blue: prior input retained in cache. Tan: new input positions.')
    return slide('An agent session contains many model calls',b,
        'Start with one coding task, not several unrelated users. Each model call returns an action or response; the harness executes tools and sends a new input containing the previous transcript plus new content. Blue marks an earlier input assumed retained by the same model/cache; tan marks material newly supplied in this diagram. The 20k includes the action serialization and test log; the next16k consist of the subsequent model action and returned file excerpts. Each row is one successive model call with the earlier input retained, so its blue region can be reused. At the end, assume the full 40k input is cached. The next slides use this exact snapshot. We conservatively count previous input reuse: whether generated output is immediately reusable as the next input depends on the serving implementation and exact token serialization. Bars show transcript structure, not proportional lengths. This scenario is explicitly supported by the vLLM multiround conversation example and Manus’s append-only agent loop; no prevalence ranking across industry workloads is implied.',
        [MANUS,APC], 'Agent cache reuse')


def hit_metrics():
    b = text(75,192,'At a model call: I = input tokens; C = tokens served from the prefix cache.',30)
    formula = mi('H')+mo('=')+frac(mi('C'),mi('I'))+mo(',')+'<mspace width="1.5em"/>'+mi('U')+mo('=')+mi('I')+mo('−')+mi('C')
    b += mathbox(75,218,590,100,formula,40)
    b += lines(690,252,['H: token hit rate', 'U: input positions that still require prefill'],28,gap=42)
    b += text(75,361,'Start from the same cached 40k history; vary only the new material.',29,600)
    b += table(75,389,1450,['New material','Input I','Reused C','Uncached U','Hit rate H'],[
        ['2k action + result','42k','40k','2k','95.2%'],
        ['20k action + result','60k','40k','20k','66.7%']],
        ratios=[.34,.15,.17,.19,.15],row_h=74,size=28)
    b += text(75,655,'Across these calls: token hit rate = (40k + 40k) / (42k + 60k) = 78.4%.',29)
    b += text(75,704,'Add token counts over the same window; averaging the two percentages gives the wrong weighting.',25,color=MUTED)
    b += takeaway('A lower hit rate can mean more new information—not lost cache.',
                   'Keep I, C and U alongside H. A request with any hit is a different metric.')
    return slide('Hit rate needs a denominator and an explanation',b,
        'Here H is cached input tokens divided by total input tokens for one model call, not a hardware L1/L2 hit rate and not a request-level Boolean. U counts fresh input positions, not bytes read from HBM or equal-cost units of GPU time. The two rows are alternative next calls after the same 40k snapshot; if those two calls are included in a reporting window, the aggregate ratio is 80/102=78.43%, whereas averaging their percentages gives about80.95%. Both requests have some cache reuse, so an any-hit request metric would report100% and conceal the difference. vLLM exposes queried/hit-token counters; for a specific engine verify which eligible tokens and boundaries its denominator covers. Provider usage fields can separate uncached tokens, cache creation and cache reads: reconstruct total input according to that provider rather than dividing by an uncached-only field. At fixed model/configuration, cache residency and complete-block alignment are assumed for these illustrative counts.',
        [METRICS,APC], 'Agent cache reuse')


def edit_boundary():
    b = text(75,190,'Replace the old 20k test interaction (action + log) with a 2k summary.',31)
    b += text(75,232,'The later 16k of messages stays unchanged. Can its KV still be reused?',29,600)
    x=325
    b += text(75,340,'Cached input',28,700)
    b += segment(x,288,210,'Fixed prefix','4k')
    b += segment(x+210,288,365,'Old test interaction','20k')
    b += segment(x+575,288,555,'Later history','16k')
    b += text(75,495,'Edited input',28,700)
    b += segment(x,443,210,'Same prefix','4k')
    b += segment(x+210,443,365,'Summary','2k',FRESH)
    b += segment(x+575,443,555,'Same text, new context','16k',FRESH)
    b += line(x+210,273,x+210,542,WARM,2,'6 5')
    b += text(x+232,407,'First changed token: reuse stops here',27,700,color=WARM)
    b += legend(325,574)
    b += lines(75,642,[
        'At deeper layers, a token’s K and V depend on earlier tokens through causal attention.',
        'Changing that history requires computing the suffix again—even if its text is identical.',
    ],28,gap=44)
    b += takeaway('Ordinary prefix caching requires the same preceding token sequence.',
                   'This invalidates reuse for the edited request; it need not erase the old cached branch.')
    return slide('Editing the past moves the cache-reuse boundary',b,
        'This is selective compaction of an old test interaction (the action and its tool result), not a requirement that all compaction systems use this exact format. Ordinary exact-prefix reuse is the assumption. The first summary token differs from the old action/log range, so only the fixed4k prefix matches. At the first Transformer layer, token embeddings may not yet mix all previous context, but the full multilayer KV state of the unchanged later text generally depends on the changed earlier context; positions may also shift when20k becomes2k. Thus neither textual identity of the later messages nor retaining their old cache blocks makes those blocks valid for this new request. Model weights, adapters, token serialization and relevant position/input settings must also agree. Engines use parent-prefix identities in cache keys; full-block boundaries may round the reusable prefix down. The old branch can remain valid for the original transcript. Figure widths are schematic; the next slide uses a proportional token scale. Tool-result clearing can use a placeholder rather than a generated summary; both change past tokens. Full-history compaction may replace a larger range and move the boundary differently.',
        [DESIGN,EDIT], 'Agent cache reuse')


def compaction_trace():
    scenes=[]
    cases=[('Keep history',42,40,2,'95.2%'),('Compact now',24,4,20,'16.7%'),('Next call',26,24,2,'92.3%')]
    captions=[
        ('Keep the history: reuse 40k; prefill only the new 2k.', 'The long history remains available to attention.'),
        ('Replace 20k with 2k: input shrinks by 18k, but fresh prefill rises by 18k.', '4k reused + 2k summary + 16k later history + 2k new material = 24k.'),
        ('After caching the edited 24k input, only the next 2k append needs prefill.', 'The hit-rate drop is temporary if the rebuilt prefix remains unchanged and available.'),
    ]
    for frame in range(3):
        s=text(75,186,'Same 40k cached history. Keep the old test interaction, or replace it with a summary.',28)
        s+=text(75,224,'Rows 1–2 are alternative next calls; row 3 follows the compacted call. k = 1,000 tokens.',24,color=MUTED)
        s+=text(315,279,'Input tokens (length to scale)',25,700,color=BLUE)
        s+=text(1310,279,'Reused / input',24,700,color=BLUE)
        for i,(name,total,cached,fresh,rate) in enumerate(cases):
            y=309+i*103
            if i<=frame:
                s+=text(75,y+30,name,27,700,color=BLUE if i==frame else INK)
                s+=rect(315,y,cached*21,35,CACHED)
                s+=rect(315+cached*21,y,fresh*21,35,FRESH)
                s+=text(315,y+68,f'{cached}k reused + {fresh}k uncached = {total}k input',25)
                s+=text(1310,y+28,rate,31,700,color=BLUE)
            else:
                s+=text(75,y+30,name,27,color='#a8b0b5')
                s+=line(315,y+18,1197,y+18,'#e1e5e8')
        s+=text(75,620,captions[frame][0],28,600,color=BLUE)
        s+=text(75,659,captions[frame][1],23,color=MUTED)
        scenes.append(s)
    b=steps('compaction',scenes,y=675)
    b+=takeaway('Shorter context can cause more prefill now and less attention work later.',
                 'Authored example, not measured latency. Cache available; block rounding ignored; summary cost separate.')
    return slide('Compaction causes a rebuild, then reuse recovers',b,
        'Advance through retain, compact, and the next append. Starting cached input is4k fixed prefix+20k old test interaction+16k later history=40k. Keeping it and appending2k requires42k total input with40k reused and2k newly prefilling. Replacing the old20k test interaction by a2k summary gives24k input: only4k match, so20k must be newly computed. After this edited input is processed and cached, another2k append gives26k input,24k reused and2k uncached. This last suffix includes the previous generated action and new observation under our conservative previous-input convention. The first two rows are alternatives at the same point, not two sequential executions; the third follows the second. Treat cache residency, eligibility, compatible model settings and stable exact serialization as assumptions. Counts idealize block rounding and any final-token recomputation. Summary generation is a separate cost and is not included in these token numbers. The bars show input positions, not GPU memory traffic, elapsed time or retained physical cache allocation. A rebuilt shorter active history can reduce subsequent attention and active KV needs, even though old unused cached blocks may remain in the pool until eviction.',
        [EDIT,APC], 'Agent cache reuse')


def amortization():
    b=text(75,190,'A prefix hit skips old token computation. New queries still attend to the cached history.',29)
    b+=line(785,238,785,458)
    b+=text(75,279,'Pay once at compaction',32,700,color=BLUE)
    b+=lines(75,337,['Generate or extract a faithful summary.',
                     'Recompute the edited input after the reuse boundary.'],27,gap=48)
    b+=text(835,279,'Potential savings on later calls',32,700,color=BLUE)
    b+=lines(835,337,['Fewer history K/V vectors for attention to read.',
                      'Less active KV state per ongoing request.',
                      'Fewer cached-input tokens billed, where applicable.'],27,gap=48)
    b+=text(75,490,'Attention to cached history: U new queries × C cached keys, per head and per layer.',26)
    b+=line(75,516,1525,516)
    b+=text(75,562,'Remaining-task latency: compaction minus keeping history',28,700,color=BLUE)
    # A deliberately small cost model, with units and assumptions kept visible.
    formula=sub('ΔT',mi('task'))+mo('≈')+mi('R')+mo('−')+mi('n')+mi('s')
    b+=mathbox(75,584,540,77,formula,41)
    b+=lines(670,606,['R: extra time now, including summary + rebuild',
                      'n: remaining calls; s: average time saved per later call'],26,gap=40)
    b+=text(75,698,'Break-even: n × s > R. Estimate from matched traces, then validate task success.',29,600)
    b+=takeaway('Optimize the remaining task—not the hit rate of the compaction call.',
                 'This approximation holds other work fixed. Lost context can change tool calls, output length and success.')
    return slide('Will future savings repay the cache rebuild?',b,
        'The additional latency R is the difference between compact-now and keep-history at this point, including any separate summary generation and extra main-model prompt processing. It is not all of the compacted call’s latency. If n subsequent comparable calls each save about s time, the net task-time change is approximately R−n*s. U new query positions each attend to C cached keys in ordinary full causal attention, giving U*C cross-prefix query-key pairs per head and layer; attention within the new suffix adds further work. This is a local planning model, not a universal latency law: future queries, output lengths, cache residency, batching and queueing can change s. Use the sum of per-call savings if it is not approximately constant. Even with a100% prefix hit, suffix and decode attention still read existing K/V. Shortening the history reduces that work and the number of active KV positions required, while the allocator may retain old evictable cache blocks. Provider billing can also charge cached-input reads, but use actual provider rates and cache-write charges for a money objective; no pricing is assumed here. The policy must also preserve correctness and task-relevant constraints. Use matched workload replay for system costs and real end-to-end task evaluation for quality and changed behavior. A hard context-window or capacity constraint can force compaction regardless of the latency break-even.',
        [APC,EDIT,BERKELEY], 'Agent cache reuse')


def context_policy():
    b=text(75,192,'For the coding agent, distinguish reducing new input from rewriting cached history.',30)
    b+=text(75,256,'Before adding a tool result',30,700,color=BLUE)
    b+=label_box(75,290,330,70,'Large test log',FRESH,27)
    b+=arrow(405,325,467,325)
    b+=label_box(467,290,455,70,'Save full log as an artifact',PALE,27)
    b+=arrow(922,325,984,325)
    b+=label_box(984,290,541,70,'Append failures + path + line ranges',PALE,26)
    b+=text(75,409,'Less new prefill; the existing prefix is unchanged. Retrieve omitted detail when needed.',28)
    b+=line(75,450,1525,450)
    b+=text(75,498,'When old history becomes too large',30,700,color=BLUE)
    b+=lines(75,548,['Compact at a checkpoint; keep that summary stable while new turns are appended.',
                     'Preserve the goal, constraints, decisions, unresolved failures and evidence locations.',
                     'Retain recent raw observations; keep older full outputs available outside the prompt.'],28,gap=45)
    b+=text(75,708,'Choose the trigger from context pressure, useful evidence and future work—not a universal turn count.',25,color=MUTED)
    b+=takeaway('Decide what enters context, what stays verbatim, and when to pay for an edit.',
                 'Clearing old tool results removes selected content; summary compaction also creates a replacement state.')
    return slide('Design the transcript to support both reuse and reasoning',b,
        'This is an authored policy sketch for a coding agent, informed by Berkeley’s long-context packing/checkpoint discussion and Manus’s restorable external memory. It is not a claim that every framework uses this policy. Before a new tool response first enters the model input, persist the full raw log and expose a useful bounded view with retrieval references. This reduces fresh input without changing old cached tokens. Selecting what to omit can lose essential evidence; record failures and exact paths or line ranges needed to recover it. Later compaction changes history and pays the previously explained rebuild cost. Preserve task constraints, current state, rejected hypotheses and unresolved errors so that the agent does not repeat failed work. A stable checkpoint followed by append-only updates permits reuse between checkpoints; repeatedly rewriting that checkpoint does not. Tool-result clearing often replaces selected old results by placeholders, whereas summarization synthesizes a replacement state and may cover much more history. Both are context policies rather than KV-cache eviction policies. A context-token threshold and retaining the most recent N tool calls are separate controls; neither establishes an industry-wide every-N-turns rule.',
        [BERKELEY,MANUS,EDIT], 'Agent cache reuse')


def diagnose():
    b=text(75,190,'Hit rate drops. Inspect how the prompt and the reused-token count changed.',29)
    b+=table(75,231,1450,['Observation in the trace','Likely mechanism','Check next'],[
        ['Input grows; reused tokens stay flat','New tool output was appended','New suffix length'],
        ['Reuse stops at an edited old message','History was cleared or summarized','First changed token'],
        ['Input prefix matches; fewer actual hits','Matching KV is not available','Routing / eviction / expiry'],
    ],ratios=[.41,.34,.25],row_h=85,size=25)
    b+=text(75,636,'Log per call: input tokens, reused tokens, first edit, replica, queue time and prefill time.',28,600)
    b+=text(75,694,'Plot these along the agent trajectory and mark compaction events; do not diagnose from H alone.',27)
    b+=takeaway('Separate new information, changed history and unavailable cache.',
                 'A cache match is useful only if the matching computation can actually be reused at that request.')
    return slide('Three different mechanisms can lower cache hit rate',b,
        'The first row is a denominator effect plus real extra fresh work, not a failure of old-prefix retention. The second is a cache-identity change; inspect serialized model input rather than only the user-visible chat, because tool definitions, timestamps, formatting or hidden context transformations can alter token prefixes. The third concerns availability: a different worker may have no local copy, memory pressure can evict it, or a provider cache can expire. Relevant model/configuration changes can also prevent reuse even if visible text matches, so hold those fixed before interpreting the third row. On a self-hosted engine record request IDs and replica/cache metrics; on a managed API use exposed cache-usage fields and accessible client timing without pretending to observe hidden server queue or prefill timings. Cache IDs/hashes can locate differences without logging sensitive raw content. For a changed agent policy, compare the same tasks and report both token-weighted hit rate and total uncached positions, queue delay, TTFT, end-to-end task time and success. Caching can interact with routing and load; the later serving section addresses those effects.',
        [DESIGN,METRICS,EDIT], 'Agent cache reuse')


def question1():
    b=text(75,189,'An engineer refreshes the agent’s summary after every tool call to keep it up to date.',29)
    b+=text(75,232,'The previous 22k input is cached. A new action + tool result adds 2k tokens.',28)
    b+=segment(75,280,295,'System + tools','4k')
    b+=segment(370,280,255,'Summary','2k')
    b+=segment(625,280,545,'Recent history','16k')
    b+=segment(1170,280,355,'New material','2k',FRESH)
    b+=table(75,391,1450,['Policy A: refresh now','Policy B: hold the checkpoint'],[
        ['Replace summary by a new 2k summary','Keep the existing 2k summary'],
        ['Its first token changes; other text is identical','Append only the new 2k material'],
    ],row_h=66,size=27)
    b+=lines(75,646,['1. Both inputs contain 24k tokens. Do they require the same prefill work?',
                     '2. How would you choose and evaluate the summary-refresh frequency?'],29,gap=48)
    b+=takeaway('Locate the reuse boundary, then compare cost over the remaining task.',
                 'Assume exact-prefix caching, matching model settings and available cached blocks. Ignore block rounding.')
    return slide('Question 1: refresh the summary after every tool call?',b,
        'Allow the audience to reason about a realistic harness design, not guess a cache formula. Both candidate prompts are24k because the summary remains2k and all other text is unchanged. The summary in policy A starts with a different first token; otherwise the reuse boundary could extend into the summary. The newly appended2k include the preceding model action and the latest tool observation, under the conservative previous-input convention. Policy B is a decision not to refresh on this call, not a policy never to compact. The cached22k input is4k system/tools+2k existing summary+16k recent history. Ask first for reused and uncached token counts, then for the tradeoff and measurement strategy. Do not infer latency directly from token counts. This is an authored systems-design question motivated by the documented cache interaction, not a verified interview question attributed to a company.',
        [DESIGN,EDIT], 'Question 1')


def solution1():
    b=table(75,187,1450,['Next-call input: 24k in both cases','Reused','Uncached'],[
        ['A: rewrite the early summary','4k','20k'],
        ['B: hold summary; append only','22k','2k'],
    ],ratios=[.62,.19,.19],row_h=72,size=29)
    b+=text(75,451,'Why',29,700,color=BLUE)
    b+=lines(295,451,['A changes the context of the later 16k; those KV states must be rebuilt.',
                      'It also pays for summary generation. 10× fresh tokens does not imply 10× TTFT.'],27,gap=41)
    b+=line(75,518,1525,518)
    b+=text(75,557,'Policy',29,700,color=BLUE)
    b+=lines(295,557,['Keep the checkpoint stable while it remains useful and fits the context budget.',
                      'Refresh when its expected future benefit or a hard limit justifies the rebuild.'],27,gap=41)
    b+=line(75,623,1525,623)
    b+=text(75,660,'Test',29,700,color=BLUE)
    b+=lines(295,660,['Matched trace: separate summary time, queue wait and main-model prefill.',
                      'Real task runs: compare completion time, success and repeated tool work.'],27,gap=41)
    b+=takeaway('The right checkpoint interval balances reuse, context size and retained information.')
    return slide('Solution 1: the summary’s location makes every edit costly',b,
        'Policy A’s reusable prefix ends at the first changed token after the initial4k; everything thereafter is uncached:2k updated summary+16k recent history+2k new material=20k. Policy B retains the full22k earlier input and processes only2k new. This comparison isolates prefix identity at equal prompt length. It does not prove that B is always faster or that summary updates are undesirable; an updated summary can improve reasoning, make future compaction effective, or be necessary to fit a context budget. Measure time spent generating the summary separately from main-model processing and queueing. Replay recorded inputs under identical model, hardware, cache warmness/residency, arrival conditions and output-length controls to isolate system costs. Then evaluate complete tasks with each policy: a replay fixes the trajectory and cannot reveal changed reasoning, forgotten constraints or repeated retrieval. Use confidence intervals across tasks rather than a single convenient prompt. The break-even approximation on the earlier slide is a guide to planning measurements, not a universal cadence. Transition back to service-level scheduling: all these calls now compete with other sessions for the same replicas.',
        [EDIT,BERKELEY], 'Question 1 solution')


def get_slides():
    slides=[agent_calls(),hit_metrics(),edit_boundary(),compaction_trace(),amortization(),context_policy(),diagnose(),question1(),solution1()]
    assert len(slides)==9
    return slides
