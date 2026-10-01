"""Build the Week 4 serving deck and its speaker notes."""
from pathlib import Path
from html import escape as esc
import json
from week4.common import *
ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'site/slide-assets/week4'

def all_slides():
    from week4.core import get_slides as core
    from week4.prefix import get_slides as prefix
    from week4.llama405b import get_slides as llama405b
    from week4.speculative import get_slides as speculative
    from week4.speculative_theory import get_slides as theory
    from week4.bandwidth import get_slide as bandwidth
    opening = core()
    slides = opening[:1]+speculative()+theory()+[bandwidth()]+llama405b()+opening[1:]+prefix()[:4]
    assert len(slides) == 30
    return slides

def build():
    slides=all_slides()
    css=(ASSETS/'base.css').read_text()+'''
    svg,svg text{font-family:Arial,Helvetica,sans-serif}
    .interaction{font:25px Arial,Helvetica,sans-serif;display:flex;gap:16px;align-items:center;color:#245675}
    .interaction button{font:24px Arial,Helvetica,sans-serif;border:1px solid #aec3d1;color:#245675;background:white;padding:10px 17px}
    .interaction button:hover:not(:disabled){background:#edf3f7}.interaction button:focus-visible{outline:3px solid #397b71;outline-offset:3px}
    .spec-cost-controls{font:25px Arial,Helvetica,sans-serif;color:#245675;display:grid;grid-template-columns:1fr 1fr;gap:10px 65px;padding-top:7px}
    .spec-cost-controls label{display:grid;grid-template-columns:auto 1fr;column-gap:8px;align-items:center}
    .spec-cost-controls label>span{white-space:nowrap}
    .spec-cost-controls input{width:100%;accent-color:#245675;margin:12px 0;height:26px;cursor:pointer}
    .spec-cost-controls input:focus-visible{outline:3px solid #397b71;outline-offset:2px}
    #spec-cost-explanation{grid-column:1/-1;font-size:24px;color:#58656d;padding-top:8px}
    .theory-math{color:#172329;padding:0;margin:0;line-height:1.1}
    .theory-math math{margin:0;text-align:left;font-family:"STIX Two Math","Cambria Math",serif}
    .prose{font-family:Arial,Helvetica,sans-serif;line-height:1.45;color:#172329}.prose p{margin:0 0 .7em}
    .code-block{margin:0;padding:18px 22px;line-height:1.35;background:#f4f6f8;border-left:3px solid #245675;font-family:ui-monospace,Menlo,Consolas,monospace;white-space:pre}
    @media print{.interaction{font-size:23px}.interaction button{display:none}.code-block{break-inside:avoid}}
    '''
    parts=[]
    notes=['# Week 4 — Speculative decoding and LLM serving','',
      '30 slides. Slides 2–11 cover speculative decoding. Slide 12 compares hardware bandwidths immediately before the Llama 3.1 405B serving case on slides 13–25. Slides 26–30 cover the request lifecycle and agent caching. Hardware assumption for the 405B case: eight servers, each with eight H100 SXM 80 GB GPUs, for 64 GPUs total.','',
      'Route: 2–3 speculative decoding and exact sampling; 4–5 KV continuation and cost; 6–7 sampling quiz and solution; 8–11 correctness and performance theory; 12 hardware bandwidth. The 405B case covers setup (13), fit (14), replica placement (15), TP ownership (16), KV budget (17), TP16 (18), PP timing (19), decode/prefill costs (20–21), context parallelism (22), FP8 (23), configuration (24), and deployment choice (25). The final section covers the request lifecycle (26), agent context and cache reuse (27–29), and interactive compaction (30).','',
      'This is an expanded teaching deck. Select a route for the meeting rather than treating the page count as a rehearsed duration. Speculative decoding, the 405B case, and agent caching can also be presented as separate sections.','',
      'The slide 12 pyramid and table distinguish cited hardware specifications, published cache measurements, and architectural estimates. Other diagrams and numerical estimates are authored teaching examples, not GPU measurements. The applied-inference chapter supplies a worked-problem structure; all 405B calculations are rederived from primary model/hardware sources and distinguish one-request latency from pipelined throughput. Versioned vLLM documentation is 0.19.1; verify the selected backend and installed version.','',
      'The optional [serving lab](https://zhiweixx.github.io/llm-systems-study-group/week-4/lab.html) remains available separately. No GPU benchmark or cluster deployment was run to produce these slides. The HTML works offline; reference links need internet.','']
    for i,s in enumerate(slides,1):
        footer=line(75,838,1525,838,'#bdc7cc')
        xx=75; links=[]
        for label,url in s['sources']:
            footer+=f'<a href="{esc(url,quote=True)}" target="_blank" rel="noopener">'+text(xx,873,label,20,color=MUTED,attrs='text-decoration="underline"')+'</a>'
            xx+=len(label)*10+32
            links.append(f'<a href="{esc(url,quote=True)}" target="_blank" rel="noopener">{esc(label)}</a>')
        footer+=text(1525,873,f'{i} / {len(slides)}',20,color=MUTED,anchor='end')
        defs=f'<defs><marker id="arrow-{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE}"/></marker></defs>'
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" width="1600" height="900" role="img" aria-labelledby="title-{i}"><title id="title-{i}">{esc(s["title"])}</title>{defs}{rect(0,0,1600,900,"white","none")}{text(75,98,s["title"],46,700,color=BLUE)}{line(75,132,1525,132)}{s["body"].replace("url(#arrow)",f"url(#arrow-{i})")}{footer}</svg>'
        aside=f'<aside class="speaker-notes"><p class="notes-meta">{esc(s.get("section",""))}</p><p>{esc(s["notes"])}</p><p>{"<br>".join(links)}</p></aside>'
        parts.append(f'<section class="slide" id="slide-{i}" data-title="{esc(s["title"],quote=True)}" data-section="{esc(s.get("section",""),quote=True)}" {"hidden" if i>1 else ""}>{svg}{aside}</section>')
        notes += [f'## {i}. {s["title"]}','',f'**Section: {s.get("section", "")}**','',s['notes'],'']
        notes += [f'- [{label}]({url})' for label,url in s['sources']]+['']
    chrome=f'''<nav class="deck-chrome" aria-label="Presentation controls"><div class="chrome-left"><button id="prev" type="button" aria-label="Previous slide">←</button><span id="counter" aria-live="polite">1 / {len(slides)}</span><button id="next" type="button" aria-label="Next slide">→</button><span id="slide-title"></span></div><div class="chrome-right"><button id="overview-toggle" type="button">Slides</button><button id="notes-toggle" type="button">Notes</button><button id="fullscreen" type="button">Full screen</button><button id="print" type="button">Print / PDF</button><button id="help-toggle" type="button" aria-label="Keyboard help">?</button></div></nav>
<section id="notes-panel" class="deck-overlay" hidden><div class="panel-header"><h2>Speaker notes</h2><button data-close-overlay="true" type="button">Close</button></div><div id="notes-content"></div></section>
<section id="overview-panel" class="deck-overlay" hidden><div class="panel-header"><h2>{len(slides)} slides</h2><button data-close-overlay="true" type="button">Close</button></div><div id="overview-list"></div></section>
<section id="help-panel" class="deck-overlay" hidden><div class="panel-header"><h2>Presentation controls</h2><button data-close-overlay="true" type="button">Close</button></div><p>Arrow keys: change slides. Home / End: first / last slide. Escape: close a panel.</p><p>Next step advances an example inside a slide. Notes contains explanations, assumptions, and sources. Slides 2–3 introduce speculative decoding and exact sampling, 4–5 cover KV continuation and cost, 6–7 give a multiple-choice quiz and its solution, and 8–11 develop the theory. Slide 12 compares hardware bandwidths before the Llama 3.1 405B deployment case on 64 H100s (13–25). The request lifecycle (26) and agent caching (27–30) close the deck.</p><p>Print / PDF shows completed step examples and the default illustrative timing case. Use the Slides menu to choose a topic.</p></section>'''
    js=(ASSETS/'navigation.js').read_text()+'\n'+(ASSETS/'interactions.js').read_text()
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LLM serving · Week 4</title><style>'+css+'</style></head><body><main id="viewport" aria-label="Presentation"><div id="stage">'+''.join(parts)+'</div></main>'+chrome+'<script>'+js+'</script></body></html>'
    (ROOT/'week-4-serving.html').write_text(html)
    (ROOT/'week-4-speaker-notes.md').write_text('\n'.join(notes).rstrip()+'\n')
    (ROOT/'.build').mkdir(exist_ok=True)
    (ROOT/'.build/week4-manifest.json').write_text(json.dumps([dict(number=i,title=s['title'],section=s.get('section','')) for i,s in enumerate(slides,1)],indent=2))
    print(f'Built {len(slides)} Week 4 slides; speculative decoding 2–11, bandwidth 12, 405B case 13–25, lifecycle and agent caching 26–30.')

if __name__=='__main__':build()
