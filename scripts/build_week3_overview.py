"""Build a separate 21-slide visual overview; never write the detailed deck."""
from pathlib import Path
from html import escape as esc
import json
from week3.common import *
ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'site/slide-assets/week3'

def all_slides():
    from week3.overview_core import foundations, tensor_pipeline
    from week3.overview_data import get_slides as data
    from week3.overview_context_experts import get_slides as context_experts
    from week3.overview_synthesis import get_slides as synthesis
    slides = foundations()+data()+tensor_pipeline()+context_experts()+synthesis()
    assert len(slides) == 21
    return slides

def build():
    slides=all_slides()
    css=(ASSETS/'base.css').read_text()+'''
    svg,svg text{font-family:Arial,Helvetica,sans-serif}
    .interaction{font:25px Arial,Helvetica,sans-serif;display:flex;gap:16px;align-items:center;color:#245675}
    .interaction button{font:24px Arial,Helvetica,sans-serif;border:1px solid #aec3d1;color:#245675;background:white;padding:10px 17px}
    .interaction button:hover:not(:disabled){background:#edf3f7}.interaction button:focus-visible{outline:3px solid #397b71;outline-offset:3px}
    .prose{font-family:Arial,Helvetica,sans-serif;line-height:1.45;color:#172329}.prose p{margin:0 0 .7em}
    .code-block{margin:0;padding:18px 22px;line-height:1.35;background:#f4f6f8;border-left:3px solid #245675;font-family:ui-monospace,Menlo,Consolas,monospace;white-space:pre}
    @media print{.interaction{font-size:23px}.interaction button{display:none}.code-block{break-inside:avoid}}
    '''
    parts=[]
    notes=['# Week 3 visual overview — speaker notes','', 'A visual introduction to data, tensor, pipeline, context, and expert parallelism, plus ZeRO/FSDP. The [50-slide reference](https://zhiweixx.github.io/llm-systems-study-group/week-3/slides.html) retains the detailed derivations and exercises.','',f'{len(slides)} slides. Use the Slides menu to navigate by section. The standalone HTML includes all diagrams and works offline. External reference links require internet access.','', 'Diagrams and toy examples show tensor ownership and communication. Sizes and timings are schematic unless stated otherwise.','']
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
<section id="help-panel" class="deck-overlay" hidden><div class="panel-header"><h2>Presentation controls</h2><button data-close-overlay="true" type="button">Close</button></div><p>Arrow keys: change slides. Home / End: first / last slide. Escape: close a panel.</p><p>Next step advances an example inside a slide. Notes contains explanations, assumptions, and sources. The separate 50-slide reference contains the detailed exercises.</p><p>Print / PDF shows the completed state of each interactive example. Use the Slides menu to choose a topic.</p></section>'''
    js=(ASSETS/'navigation.js').read_text()+'\n'+(ASSETS/'interactions.js').read_text()
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Parallelism and sharding · Week 3 visual overview</title><style>'+css+'</style></head><body><main id="viewport" aria-label="Presentation"><div id="stage">'+''.join(parts)+'</div></main>'+chrome+'<script>'+js+'</script></body></html>'
    (ROOT/'week-3-parallelism-overview.html').write_text(html)
    (ROOT/'week-3-overview-speaker-notes.md').write_text('\n'.join(notes))
    (ROOT/'.build').mkdir(exist_ok=True)
    (ROOT/'.build/week3-overview-manifest.json').write_text(json.dumps([dict(number=i,title=s['title'],section=s.get('section','')) for i,s in enumerate(slides,1)],indent=2))
    print(f'Built {len(slides)} Week 3 overview slides.')

if __name__=='__main__':build()
