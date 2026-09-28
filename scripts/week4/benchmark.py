from .common import *
BENCH=('vLLM benchmark CLI','https://docs.vllm.ai/en/latest/cli/bench/serve/')
METRICS=('vLLM metrics','https://docs.vllm.ai/en/latest/design/metrics/')
DIST=('DistServe §2–3','https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf')

def get_slides():
    body=text(75,198,'Keep the workload fixed while changing one policy.',33,700)
    body+=table(75,237,1450,['Control','Record'],[
        ['Requests','Prompt / output lengths, repeated prefixes, arrival times'],
        ['Model + hardware','Checkpoint, dtype, GPU count, placement, network'],
        ['Engine','Version, token budget, sequence limit, cache policy'],
        ['Cache condition','Cold-prefix trial vs warmed-prefix trial'],
        ['Measurement','Client timings + engine queue / KV / iteration metrics'],
    ],ratios=[.25,.75],row_h=75,size=27)
    body+=text(75,724,'Kernel warmup and prefix-cache warmup are different operations.',28,700,color=BLUE)
    body+=takeaway('Report cold and warm results separately, using the same request trace.', 'A benchmark with random prompts does not establish the benefit of repeated-prefix caching.')
    out=[slide('Design the experiment before choosing the winner',body,
        'Replay identical request content and arrival patterns, ideally in several randomized trial orders. Warm kernels and compilation before timed runs, then explicitly establish the intended prefix-cache state. Turning off benchmark warmup requests does not clear a server cache. A cold-prefix test needs cache reset/restart or guaranteed unseen prefixes, with startup/compilation handled separately. Record cache configuration, exact model revision, engine/client versions and relevant server flags. Use per-workload breakdowns so short chats do not disappear inside an aggregate average. Production traces may have bursts and correlated shared prefixes; a random fixed-length benchmark is only a controlled baseline.',[BENCH], 'Benchmarking · 7 min')]
    body=text(75,199,'Two experiments answer different questions.',33,700)
    body+=text(75,279,'Fixed concurrency',32,700,color=BLUE)+text(840,279,'Fixed arrival rate',32,700,color=BLUE)
    body+=line(795,250,795,640)
    body+=lines(75,339,['Keep C requests in flight.', 'When one finishes, send another.'],29,gap=45)
    body+=label_box(75,435,270,78,'Send C requests',size=26)+arrow(345,474,410,474)+label_box(410,435,305,78,'Wait for completion',size=24)
    body+=lines(75,571,['Slower service → fewer new arrivals.', 'Useful for saturation / batch-efficiency tests.'],25,gap=40)
    body+=lines(840,339,['Send requests at a chosen rate λ.', 'Arrivals continue while earlier work waits.'],29,gap=45)
    for j in range(5):body+=arrow(866+j*135,428,866+j*135,475)
    body+=label_box(840,489,650,72,'Queue grows if the service falls behind',size=26)
    body+=text(840,618,'Useful for latency and overload tests.',27)
    body+=text(75,700,'A concurrency cap can throttle the intended arrival stream: record achieved rate too.',27)
    body+=takeaway('To test a service under demand, sweep the offered request rate.', 'Run long enough to see queue growth, errors, timeouts, and the drain period.')
    out.append(slide('Request rate and concurrency are different controls',body,
        'The left is a closed-loop client. Its achieved request rate drops automatically when requests take longer; it can mask an overloaded open-loop service. The right schedules arrivals independently of completion, for example with Poisson inter-arrival times. Real tools may limit concurrency or be client-CPU bound, so verify actual dispatch timestamps and report client-side queuing separately. The number of in-flight requests is an outcome of arrival rate and request duration in an open-loop test; it is not synonymous with engine decode batch size. Both tests are useful but should not be compared as if they had identical demand.',[BENCH], 'Benchmarking · 7 min'))
    body=text(75,194,'Illustrative load sweep — invented numbers, not a GPU measurement',27,color=MUTED)
    body+=text(75,240,'Target: ≥95% succeed with TTFT ≤1 s AND per-request TPOT ≤50 ms.',27,700)
    body+=table(75,274,1450,['Offered req/s','Completed req/s','p95 TTFT','p95 TPOT','Joint pass rate'],[
        ['2','2','0.25 s','24 ms','99%'],['4','4','0.45 s','31 ms','98%'],
        ['6','6','0.85 s','44 ms','96%'],['8','8','1.80 s','70 ms','75%'],
    ],ratios=[.19,.23,.19,.18,.21],row_h=67,size=26)
    body+=text(75,660,'At 6 req/s: 6 × 0.96 = 5.76 requests/s meet both bounds.',30,700,color=BLUE)
    body+=text(75,705,'At 8 req/s: 8 × 0.75 = 6.00 pass/s, but the 95% service target fails.',29)
    body+=takeaway('Choose capacity using latency attainment, not the largest throughput number.', 'Here: 6 req/s is the highest tested rate that meets the target; inspect stability before deployment.')
    out.append(slide('More throughput can still violate the service target',body,
        'These values are authored for interpretation, not fitted or measured. Assume stable intervals, no errors and matched arrivals/completions so the rates can be multiplied directly. The pass fraction counts requests satisfying BOTH latency conditions, not the product of two independent percentile pass rates. Goodput here means passing completed requests per second. Even a higher absolute goodput can violate a required 95% attainment fraction, so both matter. The largest tested passing rate only brackets capacity; finer sweeps, longer runs, failures and bursts may change the choice. In a finite run with queue drain, report offered rate, achieved dispatch rate, measurement duration and completion window rather than assuming they coincide. Inspect per-request ITL tails in addition to TPOT.',[DIST,BENCH], 'Benchmarking · 7 min'))
    body=text(75,200,'Use the symptom to choose the next measurement.',33,700)
    body+=table(75,253,1450,['Observed change','Next measurement / controlled experiment'],[
        ['High hit rate; high TTFT','Per-replica queue delay; balance load with the same trace'],
        ['Chat token gaps spike','Mixed-round durations; sweep prefill token budget'],
        ['KV pool fills; stalls rise','Active vs evictable blocks; preemption / recompute counts'],
        ['PD decode pool is idle','Prefill queue and KV transfer; vary the P:D split'],
        ['Throughput plateaus','Client dispatch rate, GPU timeline, bandwidth / compute'],
    ],ratios=[.38,.62],row_h=77,size=26)
    body+=takeaway('Change one variable, and state what result would disprove the diagnosis.', 'A busy GPU, a high hit rate, or a full KV pool is not a diagnosis by itself.')
    out.append(slide('Connect a bad user metric to an engine cause',body,
        'Each row is a hypothesis test, not a lookup table guaranteeing the cause. For example, long mixed rounds plus unchanged isolated decode-kernel times supports scheduler interference; if token stalls persist when long prompts are removed, investigate host stalls or network buffering. For a full KV pool, separate active references from idle retained cache blocks: evictable cached data may be healthy use of memory. Recomputations and queue growth are stronger evidence of pressure. In PD, destination idleness can result from a slow prefill stage or transfer starvation; examine stage timelines. Keep the four-GPU budget, model and input trace fixed.',[METRICS,DIST], 'Benchmarking · 7 min'))
    body=text(75,196,'Take-home experiment: locate the highest passing request rate',32,700)
    command_lines=['python run_benchmark.py --model YOUR_MODEL ' + chr(92), '  --base-url http://YOUR_SERVER:8000 ' + chr(92), '  --rates 1 2 4 8 --num-prompts 200 ' + chr(92), '  --server-record server-record.txt --run']
    body+=code(75,246,1450,190,'\n'.join(command_lines),26)
    body+=text(75,459,'The companion script prints commands by default; --run sends requests.',27,color=MUTED)
    body+=lines(75,524,['1. Start with a controlled random-prompt workload; save per-request results.', '2. Plot offered rate against p95 TTFT, p95 TPOT, errors, and passing req/s.', '3. Repeat one policy change with the same trace and hardware budget.'],29,gap=52)
    body+=text(75,706,'Prefix-cache extension: use a repeated-prefix trace and explicit cold / warm trials.',27)
    body+=takeaway('Deliver one measured plot, a reproducible command, and a supported explanation.', 'The lab includes setup assumptions and CLI sources; no GPU measurements are supplied.')
    out.append(slide('A reproducible serving experiment',body,
        'Optional work after the meeting. The script requires a working endpoint and a compatible vLLM benchmark client; it does not start a server or allocate GPUs. Inspect the printed commands before using --run. The default fixed-length random requests isolate load and do not test prefix reuse. The README explains a separate prefix-trace extension and how to distinguish warm kernels from warmed KV prefixes. Use enough samples and repetitions for stable tail estimates: 200 requests is an inexpensive starting run, not enough to claim robust p99 under arbitrary variability. Record offered and achieved rate, prompt/output lengths, server config and software versions. Full commands and per-request data are more useful than a single tokens/s number.',[('Companion lab','https://zhiweixx.github.io/llm-systems-study-group/week-4/lab.html'),BENCH], 'Benchmarking · 7 min'))
    body=text(75,211,'Design the next experiment for our four-GPU chat service.',36,700)
    rows=[('Choose the workload','Short chats, long documents, repeated prefixes, bursts.'),('Choose the target','TTFT, streaming gaps, and an explicit passing fraction.'),('Choose one change','Speculation, cache policy, scheduling, or routing.'),('Choose the evidence','Which timeline or metric would overturn your explanation?')]
    for j,(a,b) in enumerate(rows):
        y=300+j*96
        body+=line(75,y-24,1525,y-24)+text(75,y+18,a,30,700,color=BLUE)+text(480,y+18,b,28)
    body+=text(75,716,'Discuss for 15 minutes. Hold the hardware budget and workload constant.',28,color=MUTED)
    body+=takeaway('Connect the chain: kernels → engine → multi-GPU layout → user latency.')
    out.append(slide('Discussion: defend a serving decision with evidence',body,
        'Close the main route here. Divide the group into workload, policy and measurement roles. Have each group state a baseline, one intervention, an expected observation and a result that would falsify it. A useful answer must say what is held constant; “add more GPUs” avoids the fixed-budget question. Discuss fairness: optimizing overall throughput may leave a class of long requests with unacceptable waits, while aggressively favoring chat may starve document requests. The two worked question solutions provide candidate experiments but no universal winner. Speculative decoding is another candidate intervention: measure both per-stream latency and capacity under concurrent load.',section='Discussion · 15 min'))
    return out
