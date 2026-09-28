"""Week 4: scheduling, routing, and the second discussion problem."""
from .common import *

OPT = ('vLLM: scheduler tuning', 'https://docs.vllm.ai/en/latest/configuration/optimization/')
SARATHI = ('Sarathi-Serve, OSDI 2024', 'https://arxiv.org/abs/2403.02310')
ORCA = ('Orca, OSDI 2022', 'https://www.usenix.org/conference/osdi22/presentation/yu')
PD = ('vLLM: disaggregated prefill', 'https://docs.vllm.ai/en/latest/features/disagg_prefill/')
DIST = ('DistServe, OSDI 2024', 'https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin')
ROUTER = ('SGLang: cache-aware routing', 'https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/policies/cache_aware.rs')


def queue_slide():
    b = text(75,195,'A fast individual request does not imply a stable service under load.',32,600)
    b += text(75,241,'Illustrative queue for one replica: 6 arrivals/s, at most 4 completions/s.',26,color=MUTED)
    b += table(75,290,840,['One-second interval','Arrivals','Completions','Queue at end'],[
        ['1','6','4','2'],['2','6','4','4'],['3','6','4','6']],
        ratios=[.34,.18,.22,.26],row_h=78,size=27)
    b += line(970,280,970,615)
    b += text(1010,326,'Backlog grows each second',30,700,color=BLUE)
    for row,n in enumerate([2,4,6]):
        y=370+row*72
        for j in range(n): b += rect(1010+j*64,y,49,39,PALE)
    b += lines(75,665,['Real request costs vary with prompt/output length, cache hits and batching.',
                          'Even below average capacity, bursts can produce a long tail of waiting times.'],26,gap=37)
    b += takeaway('Measure arrival rate, completion rate and queue age together.')
    notes = '''The numbers are a teaching construction, not a GPU benchmark. Count complete requests here, not generated tokens. The recurrence is backlog at the end = backlog at the start + arrivals - completions, with a lower bound of zero. Assuming a sustained maximum of four completions per second, six arrivals per second cannot be served indefinitely without accumulating requests. A larger queue postpones rejection but does not add processing capacity. Real LLM service capacity is workload-dependent: different prompt lengths, output lengths, prefix reuse and batch composition change the work per request. A finite burst can also make tail latency poor even when a long-run average arrival rate is below capacity. Observe the slope of the backlog and the age of the oldest request, not only a snapshot of GPU utilization. When benchmarking, distinguish how many requests the client tries to send from how many it actually sends and how many complete.'''
    return slide('Queues grow when arrivals outrun completions',b,notes,[ORCA],section='Scheduling')


def budgets_slide():
    b = text(75,196,'A scheduler must satisfy three different limits before launching a round.',31,600)
    cols=[(75,450,'Running sequences','How many requests can progress?',[
        'A sequence is one active generation.',
        'More sequences can improve batching,',
        'but retain more KV state.'], 'Example control: max_num_seqs'),
        (580,450,'New tokens this round','How much fresh work is scheduled?',[
        'One decode position per live request;',
        'possibly many prompt positions.',
        'The budget limits their total.'], 'Example: max_num_batched_tokens'),
        (1085,440,'KV capacity','How much history can remain?',[
        'Cached keys and values persist',
        'across rounds and grow as',
        'generation continues.'], 'Limited by available KV blocks')]
    for x,w,title,question,items,knob in cols:
        b += rect(x,262,w,335,PALE)
        b += text(x+22,307,title,30,700,color=BLUE)
        b += text(x+22,353,question,24,600)
        b += lines(x+22,417,items,25,gap=40)
        b += text(x+22,568,knob,21,color=MUTED)
    b += lines(75,657,['Four decode tokens can still require reading a large existing KV cache.',
                       'The new-token budget neither counts those history reads nor guarantees a round duration.'],27,gap=39)
    b += takeaway('Request count, scheduled-token count and cached-token count are different quantities.')
    notes='''Use the same four-GPU service, but zoom into the scheduler on one replica. The first limit constrains sequences admitted to execution; the second constrains the sum of newly processed positions in an iteration; the third is a memory resource constraint. The controls shown are examples from vLLM, not an assertion that every framework uses identical admission policies. A request with 10,000 cached positions still contributes one newly processed position to an ordinary decode round. It retains the old KV state and attention may read that history again. Therefore lowering a new-token budget can reduce added prefill work without making decode KV reads disappear. Sequence count also does not predict memory by itself: ten long-context requests can retain more KV than many short-context requests. Implementation versions and optional features can alter how the scheduler accounts for speculative or multimodal positions; this example is ordinary text generation with no speculation.'''
    return slide('Three budgets constrain each scheduling round',b,notes,[OPT,ORCA],section='Scheduling')


def scheduling_slide():
    scenes=[]
    for step in range(4):
        s=text(75,198,'Toy example: 4 chat streams + one 12-token prompt; new-token budget = 8.',30,600)
        s+=text(75,240,'Reserve 4 decode positions, then use the remaining 4 positions for prefill.',26,color=MUTED)
        s+=text(75,295,'New prompt',29,700,color=BLUE)
        for j in range(12):
            x=75+j*75
            fill=PALE if j<step*4 else 'white'
            s+=rect(x,322,64,64,fill)
            s+=text(x+32,363,str(j+1),25,600,anchor='middle',color=BLUE if j<step*4 else MUTED)
        s+=text(1030,347,f'{step*4} / 12 processed',31,700,color=BLUE)
        s+=text(1030,387,f'{12-step*4} prompt tokens remain',25,color=MUTED)
        s+=line(75,420,1525,420)
        s+=text(75,468,'One row = one completed scheduling round',27,600)
        s+=text(1080,468,'No duration scale',24,color=MUTED)
        for r in range(3):
            y=496+r*47
            active=r<step
            s+=text(75,y+29,f'Round {r+1}',24,600,color=INK if active else MUTED)
            s+=label_box(230,y,385,37,'4 decode positions',PALE if active else '#f7f8f9',23)
            s+=label_box(635,y,445,37,f'4 prompt positions ({r*4+1}–{r*4+4})', '#e9f0ed' if active else '#f7f8f9',23)
            s+=text(1110,y+29,f'Each chat: +1 token' if active else 'Not scheduled yet',23,color=BLUE if active else MUTED)
        scenes.append(s)
    b=steps('schedule',scenes)
    b+=takeaway('Chunking gives ongoing decodes another turn before the whole prompt finishes.',
                'Assume no chats finish, enough sequence slots and KV capacity; token counts are illustrative.')
    notes='''Click Next step three times. At the start, four existing chat requests are decoding. A fifth request arrives with twelve uncached prompt positions. An eight-position budget admits four existing decode positions plus a four-position prompt chunk. After round 1, each existing chat has received one more output token and the new prompt has four processed positions. After round 2, the prompt has eight. Round 3 processes its final four positions; the prompt is complete and the new request can produce its first output token. It joins decoding subsequently if it continues. The toy freezes the other four chats to expose the scheduler rule. This is a teaching example of decode-prioritized chunking, not a performance simulation: equal-size boxes and rounds do not imply equal execution times. Without chunking, a long prefill admitted into a round can stretch the interval before existing chats get their next output. Smaller chunks limit that added work, while potentially delaying completion of the new prompt and adding overhead. The new prompt retains KV for its earlier chunks, which later chunks attend to as needed. Chunked prefill is distinct from continuous batching: the latter changes which requests participate; chunking additionally changes how much of a new prompt is processed in a round.'''
    return slide('Chunking controls how much a new prompt interrupts chat',b,notes,[OPT,SARATHI],section='Scheduling')


def admission_slide():
    b=text(75,196,'A request that fits now may need additional KV blocks before it finishes.',31,600)
    b+=text(75,240,'Illustrative pool: 100 equally sized KV blocks on one replica.',26,color=MUTED)
    x0,y,w=75,289,1450
    for fraction,label,fill in [(.7,'70 active blocks',PALE),(.2,'20 retained', '#e9f0ed'),(.1,'10 empty','white')]:
        bw=w*fraction
        b+=rect(x0,y,bw,79,fill)
        b+=text(x0+bw/2,y+49,label,28,600,anchor='middle')
        x0+=bw
    b+=text(75,409,'Active: still referenced by running requests',26)
    b+=text(825,409,'Retained: reusable, but eligible for eviction',26)
    b+=line(75,443,1525,443)
    rows=[('Before admission','Account for prompt KV and expected generation growth.'),
          ('If memory gets tight','Evict unreferenced cached blocks; pause admission if necessary.'),
          ('If active state must be freed','Preempt a request; rebuilding its KV later adds work and latency.')]
    for i,(left,right) in enumerate(rows):
        yy=493+i*79
        b+=text(75,yy,left,28,700,color=BLUE)
        b+=text(530,yy,right,27)
        if i<2:b+=line(75,yy+30,1525,yy+30)
    b+=takeaway('Admission policy trades immediate concurrency against later queueing and recomputation.',
                'Output length is uncertain: conservative reservation and optimistic admission have different costs.')
    notes='''Active, retained and empty describe block-content states; retained blocks may already be in the allocator’s free queue. A retained block is not referenced by an active request, but its content may still be reused by a future prefix hit; reclaiming it loses that opportunity. In this toy, a new request requiring twenty blocks initially can be admitted by using the ten free blocks and evicting ten retained blocks. That leaves no free blocks and only ten more evictable blocks. Additional generation by all active requests can soon consume the remainder. This is why checking current prompt fit is not a complete admission policy. A conservative policy can account for maximum allowed future output, at the cost of admitting fewer requests; an optimistic policy can admit more and recover from pressure with preemption or other mechanisms. In a recomputation-based policy, freeing an active request's KV means its history must be processed again before it resumes. Repeated preemption can turn memory pressure into a throughput and latency problem. Observe preemption counts, cache-block state and queue age together. Admission may also reject or defer work according to an application's latency limit; a waiting queue alone cannot make sustained overload disappear.'''
    return slide('Admission must consider future KV growth',b,notes,[OPT],section='Scheduling')


def routing_slide():
    b=text(75,196,'The router chooses a replica. That replica chooses the next GPU batch.',32,600)
    b+=label_box(565,234,470,63,'Request router',PALE,29)
    b+=line(245,334,1355,334)
    b+=arrow(800,297,800,334)
    descriptions=[('Replica 0','Matching prefix','Long queue'),('Replica 1','No matching prefix','Short queue'),('Replica 2','Partial prefix match','Moderate queue'),('Replica 3','Matching prefix','KV pool under pressure')]
    for i,(name,a,c) in enumerate(descriptions):
        x=75+i*370
        b+=arrow(x+170,334,x+170,365)
        b+=rect(x,365,340,207,'white')
        b+=text(x+20,407,name,29,700,color=BLUE)
        b+=lines(x+20,451,[a,c],25,gap=36)
        b+=label_box(x+20,510,300,42,'Local scheduler → GPU',PALE,22)
    b+=lines(75,632,['Choose using both reuse and load: avoid sending every popular prefix to one busy replica.',
                       'Useful signals: queued work, running load, reusable prefix length and KV headroom.'],27,gap=42)
    b+=takeaway('Cache locality saves prefill work; queueing can erase that saving.')
    notes='''These four boxes are the four one-GPU replicas from the case study. All hold the same model weights, but their queues and cached prefixes differ. The router makes a placement decision once for an incoming request in this simple architecture. Each local scheduler then repeatedly decides which requests and prompt chunks enter its next iteration. Routing and iteration scheduling solve different problems. A round-robin router ignores prefix locality, while always choosing the longest matching prefix can overload a hot replica. Prefix length also does not capture the work remaining in the local queue: a few very long requests can be more expensive than many short ones. A useful routing estimate compares expected waiting plus uncached prefill work, and considers memory pressure; implementations can use heuristics rather than accurate time predictions. SGLang's cache-aware policy is a concrete implementation that incorporates load-balancing behavior. The categorical states here are an authored diagram, not telemetry from that router. Moving already-active requests is a separate mechanism and can require KV transfer; it is not implied by this diagram.'''
    return slide('Routing balances prefix reuse against local load',b,notes,[ROUTER],section='Deployment')


def pd_slide():
    b=text(75,197,'Hold the hardware budget fixed: four GPUs, the same 8B model and request trace.',30,600)
    b+=line(790,252,790,679)
    b+=text(75,279,'Four shared-phase replicas',31,700,color=BLUE)
    b+=text(845,279,'Separate prefill and decode pools',31,700,color=BLUE)
    for j in range(4):
        x=75+j*169
        b+=label_box(x,326,151,75,f'GPU {j}',PALE,27)
        b+=text(x+75,443,'P + D',29,700,anchor='middle')
    b+=lines(75,517,['Every replica can process either phase.',
                      'Chunking limits prefill added to a round.',
                      'No P→D handoff across GPUs.'],27,gap=46)
    b+=label_box(1050,321,240,67,'GPU 0: prefill',PALE,27)
    b+=arrow(1170,388,1170,442)
    b+=text(1240,432,'KV transfer',23,color=MUTED)
    b+=line(955,442,1393,442)
    for j in range(3):
        x=845+j*219
        b+=arrow(x+100,442,x+100,472)
        b+=label_box(x,472,200,64,f'GPU {j+1}: decode','#e9f0ed',24)
    b+=lines(845,586,['Example split: 1 P + 3 D; not an optimum.',
                       'Either pool or the link can become a bottleneck.'],26,gap=40)
    b+=takeaway('Compare latency and completed useful work under the same four-GPU budget.',
                'Both designs store model weights on all four GPUs; PD also needs KV transfer and pool sizing.')
    notes='''Week 2 introduced prefill–decode disaggregation. Here the new issue is resource allocation and comparison fairness. The left design allows all four one-GPU replicas to switch between the two phases, applying chunked prefill to control interference. The right illustration assigns one GPU to prefill and three to decode. Each GPU still has a full copy of the model in this case; this is not pipeline parallelism. The prefill worker computes prompt KV and the state needed to continue generation is handed off to a selected decode worker. KV transfer, destination capacity and handoff scheduling can delay progress; transfer can sometimes overlap other work, so raw transfer duration is not necessarily wholly exposed. Separation can protect decode iterations from new prefill work and allows phase-specific tuning, but a busy prefill pool can leave decoders underused or vice versa. The illustrated 1:3 split is arbitrary. Select a split based on the workload and latency targets, and compare against a well-tuned shared-phase baseline at the same GPU count. Published DistServe results are workload- and configuration-specific; this slide claims no numerical speedup.'''
    return slide('Compare shared and separate prefill/decode pools',b,notes,[DIST,PD],section='Deployment')


def question_slide():
    b=lines(75,190,['Four one-GPU replicas serve the same 8B model.',
                     'Some short prompts are replaced by long documents; request rate and output lengths stay fixed.'],29,gap=43)
    b+=text(75,293,'Hypothetical observations — not a benchmark result',24,color=MUTED)
    b+=table(75,318,1450,['Observation','Short prompts','Mixed prompt lengths'],[
        ['Chat inter-token latency, p95','35 ms','180 ms'],
        ['Decode-only kernels, matched shapes','Similar duration','Similar duration'],
        ['Timeline during chat stalls','Short GPU rounds','Long mixed P + D rounds']],
        ratios=[.49,.21,.30],row_h=72,size=26)
    b+=text(75,661,'What would you test first, and what evidence would justify moving to PD separation?',30,600,color=BLUE)
    b+=takeaway('Explain the stall, propose a controlled comparison and name the tradeoff.',
                'Keep all four GPUs; do not assume that a slower output stream means a slower decode kernel.')
    notes='''This is an authored interview-style problem, not a question attributed to a particular company. Give participants around one minute before advancing to the solution. ITL is measured on the client-visible output stream. The table says that an isolated or matched-shape decode-only kernel interval remains similar; it does not say that the entire mixed iteration has unchanged runtime. The trace contains more prompt work in the intervals where chats wait. Thus the evidence points toward scheduling interference as a first hypothesis, while not proving that it is the only cause. Ask for a concrete experiment rather than a list of optimizations. The request trace, model, GPU count, cache condition and output lengths should be held fixed when comparing scheduler configurations. Good answers mention the effect on new requests' time to first token and total served work, as well as chat ITL. They also state what measurement would falsify the hypothesis and avoid jumping to a resource-unequal PD comparison.'''
    return slide('Question 2: Why do chat streams stall on long prompts?',b,notes,[SARATHI,DIST],section='Questions')


def solution_slide():
    b=text(75,193,'The trace suggests long mixed rounds, not intrinsically slower decode kernels.',30,600)
    rows=[('1  Test the scheduling hypothesis',[
        'Replay the same request trace; vary only the prefill chunk / new-token budget.',
        'Check chat ITL alongside prompt TTFT, completed work and preemptions.']),
        ('2  Check what changed',[
        'If mixed rounds shorten and ITL improves, the evidence supports interference.',
        'If stalls remain elsewhere, inspect host gaps, KV pressure and client delivery.']),
        ('3  Evaluate PD only with a fair baseline',[
        'Compare tuned shared replicas against P/D splits of the same four GPUs.',
        'Include KV transfer, both pool queues and both latency objectives.'])]
    for i,(heading,items) in enumerate(rows):
        yy=264+i*145
        b+=text(75,yy,heading,30,700,color=BLUE)
        b+=lines(75,yy+45,items,27,gap=40)
        if i<2:b+=line(75,yy+115,1525,yy+115)
    b+=takeaway('Success means better service under its latency targets, not only shorter chat gaps.',
                'A smaller chunk can delay first tokens; a separate prefill pool can itself become overloaded.')
    notes='''First separate the duration of GPU kernels from the time between outputs seen by a request. A chat may wait while a mixed iteration performs additional prefill work. The primary controlled experiment replays the same mixed workload and varies the chunk size or iteration token budget while keeping other controls fixed. Collect per-round scheduled prompt/decode counts and duration, client ITL, TTFT, completions and preemption events. The expected tradeoff is less prefill added to each round but potentially more rounds and overhead before a new prompt completes. If ITL remains poor after mixed rounds become short, the original explanation is incomplete: inspect CPU scheduling or launch gaps, network buffering, cache misses, KV preemption and any changed kernel shapes. Use admission limits if offered load or KV growth exceeds sustainable capacity. PD is a further deployment comparison, not an automatic solution. Test resource splits under the same four-GPU budget and include transfer and destination waiting. A design that makes chats smoother but causes new requests to miss their first-token objective may be unacceptable. The final choice depends on the specified service objectives and workload distribution.'''
    return slide('Solution 2: Attribute the delay before changing deployment',b,notes,[SARATHI,DIST],section='Questions')


def get_slides():
    return [queue_slide(),budgets_slide(),scheduling_slide(),admission_slide(),routing_slide(),pd_slide(),question_slide(),solution_slide()]
