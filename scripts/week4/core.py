from .common import *
METRICS=('vLLM metrics', 'https://docs.vllm.ai/en/latest/design/metrics/')
BERKELEY=('Berkeley L18', 'https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture_18.pdf')

def get_slides():
    body=text(75,214,'Speculative decoding and LLM serving',59,700,color=BLUE)
    body+=text(75,280,'From faster token generation to a responsive service',36)
    body+=lines(75,368,['First: can we generate several tokens with one target-model call?', 'Then: what happens when many requests keep arriving?'],32,gap=52)
    stages=[('Generate faster','Speculative decoding'),('Reuse and schedule','Prefixes + queues'),('Serve under load','Routing + measurement')]
    for i,(a,b) in enumerate(stages):
        x=75+i*500
        body+=line(x,486,x+440,486)+text(x,536,a,32,700,color=BLUE)+text(x,579,b,29)
    body+=text(75,682,'Speculative decoding: visual walkthrough, then correctness and performance theory.',28,color=MUTED)
    body+=takeaway('Goal: reduce generation time while meeting explicit service latency targets.', '34 slides · about 55–60 min of teaching material, followed by discussion.')
    out=[slide('Week 4 · From model execution to an online service',body,
        'Open with the six-slide speculative-decoding lesson: motivation, parallel verification, greedy acceptance, KV rollback and the bonus, exact sampling, then a timing experiment. Allow 12–15 minutes including the interactive steps, then 8–10 minutes for the four theory slides: exact sampling, acceptance as distribution overlap, expected tokens per round, and expected speedup. Slides 12–13 establish the four-GPU serving case; 14–20 teach prefix reuse; 21–28 cover scheduling, routing and deployment; 29–33 cover measurement; 34 begins discussion. The expanded full sequence is approximately 55–60 minutes plus 15 minutes of discussion, not a rehearsed duration. Choose later sections if retaining a 50-minute meeting. Numerical cases and diagrams are authored teaching examples, not GPU measurements.',section='Opening')]
    body=text(75,202,'One case throughout: an 8B chat service on four GPUs',34,700)
    body+=text(75,246,'Assume the model fits on one GPU; start with four independent replicas.',28)
    body+=label_box(75,328,240,92,'Requests',size=32)+arrow(315,374,405,374)+label_box(405,328,260,92,'Router',size=32)
    for i in range(4):
        y=285+i*91
        body+=label_box(900,y,625,69,f'GPU {i}  ·  queue + engine + local KV cache',size=25)
        body+=arrow(665,374,835,374) if i==0 else ''
        body+=line(835,320,835,594) if i==0 else ''
        body+=arrow(835,y+35,900,y+35)
    body+=lines(75,495,['Workload: short chats + long documents', 'Repeated system prompts; varied user suffixes'],27,gap=42)
    body+=line(75,640,1525,640)+text(75,684,'Example targets: ≥95% pass both TTFT ≤1 s and per-request TPOT ≤50 ms.',29,700,color=BLUE)
    body+=takeaway('The resource budget stays fixed when we compare serving policies.', 'Targets are illustrative. We also inspect ITL tails so a mean cannot hide streaming stalls.')
    out.append(slide('The case: four GPUs, variable prompts, latency targets',body,
        'Use one model and precision across all experiments. A replica is a complete model instance here, not a tensor-parallel shard. Each replica owns its own queue and KV cache; the baseline does not share cache storage across GPUs. Workload lengths are not fixed in the motivating case; controlled experiments will isolate them. TTFT is client time to first output token. TPOT is the per-request mean time for subsequent output tokens. Our service target means at least 95% of offered requests successfully satisfy BOTH bounds; errors and timeouts count as failures. ITL measures individual gaps and can reveal stalls that a request average hides. The specific thresholds are teaching assumptions, not measured limits or recommendations for a particular product.',[METRICS], 'Setup · 4 min'))
    body=text(75,199,'A request can wait before and between GPU operations.',32,700)
    boxes=[(75,270,240,'Route + queue'),(350,270,300,'Prefill uncached input'),(685,270,220,'First token'),(940,270,585,'Decode → stream → repeat')]
    for x,y,w,label in boxes:body+=label_box(x,y,w,84,label,size=26)
    for x1,x2 in [(315,350),(650,685),(905,940)]:body+=arrow(x1,312,x2,312)
    body+=line(75,401,905,401,BLUE,3)+text(490,443,'TTFT: arrival → first token',30,700,color=BLUE,anchor='middle')
    body+=line(940,401,1525,401,TEAL,3)+text(1230,443,'ITL: gap between tokens',29,700,color=TEAL,anchor='middle')
    body+=lines(75,533,['Router: chooses the replica.', 'Scheduler: chooses the work in each engine iteration.', 'KV manager: tracks active, reusable, and reclaimable blocks.'],29,gap=47)
    body+=text(75,696,'Completion: release the request’s references; retain reusable prefix blocks if policy allows.',27)
    body+=takeaway('Measure the user timeline and the engine timeline together.', 'TPOT is a request’s mean token gap; ITL measures individual gaps. Both include client-visible waits.')
    out.append(slide('Follow one request from arrival to completion',body,
        'This is a logical sequence, not a scale drawing or a promise of one GPU kernel per box. Prefill may be split across engine iterations and interleaved with other requests. Prefix caching can skip a portion of input computation, but routing, queuing, uncached work, and the output-generation path remain. The first token is produced using the prompt processing result; subsequent tokens are generated by decode steps. For n>1 output tokens, client mean TPOT=(last-token time−first-token time)/(n−1), with streaming event/token conventions stated by the benchmark. Individual ITL samples need not equal the mean. Network buffering can change observed streaming gaps. Define timestamp boundaries before comparing server and client metrics.',[METRICS,BERKELEY], 'Setup · 4 min'))
    return out
