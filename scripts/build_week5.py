"""Convert Gaotang Li's source PDF to a self-contained Week 5 HTML deck.

Requires pypdf and Poppler's pdftocairo. PDF pages stay as vector artwork,
including their embedded figures, rather than being re-typeset or summarized.
GitHub Pages publishes the checked-in HTML; it need not run this converter.
"""
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sources/week-5/Linear_Attention_gaotang_li.pdf"
OUT = ROOT / "week-5-linear-attention.html"
BUILD = ROOT / ".build/week5"
ASSETS = ROOT / "site/slide-assets/week3"


def section_name(number):
    if number == 1:
        return "Linear Attention · Gaotang Li"
    if number <= 5:
        return "Linear attention"
    if number <= 11:
        return "DeltaNet, GDN and KDA"
    if number <= 33:
        return "Chunkwise DeltaNet"
    if number <= 38:
        return "Quality and efficiency"
    if number <= 48:
        return "Appendix"
    return "References"


def convert_page(number, executable):
    target = BUILD / "svg" / f"page-{number:02}.svg"
    subprocess.run(
        [executable, "-svg", "-noshrink", "-nocenter", "-f", str(number), "-l", str(number), str(SOURCE), str(target)],
        check=True, capture_output=True,
    )
    svg = target.read_text()
    svg = svg[svg.index("<svg"):]
    # Cairo reuses glyph/clip IDs on every page. Make every definition unique
    # before embedding all pages in the same HTML document.
    prefix = f"p{number}-"
    svg = re.sub(r'id="([^"]+)"', lambda m: f'id="{prefix}{m[1]}"', svg)
    svg = re.sub(r'(?:xlink:)?href="#([^"]+)"', lambda m: f'href="#{prefix}{m[1]}"', svg)
    svg = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{prefix}{m[1]})', svg)
    # Use ordinary HTML-compatible href for embedded images as well.
    svg = svg.replace("xlink:href=", "href=")
    svg = re.sub(r'width="[^"]+"', 'width="1600"', svg, count=1)
    svg = re.sub(r'height="[^"]+"', 'height="900"', svg, count=1)
    # Accessible content is supplied by the section transcript below.
    svg = svg.replace("<svg ", '<svg aria-hidden="true" focusable="false" ', 1)
    if number == 1:
        credit = '''<g font-family="Arial,Helvetica,sans-serif" fill="#142b4e" text-anchor="middle">
<text id="author-credit" x="360" y="211" font-size="16" font-weight="600">Original slides by Gaotang Li</text>
<text x="360" y="231" font-size="11.5">LLM Systems Study Group · Week 5</text>
</g>'''
        svg = svg.replace("</svg>", credit + "</svg>")
    return svg


def build():
    executable = shutil.which("pdftocairo")
    if not executable:
        raise SystemExit("Install Poppler (pdftocairo) to regenerate the Week 5 deck.")
    reader = PdfReader(SOURCE)
    count = len(reader.pages)
    assert count == 49, "Review source changes before rebuilding the converted deck."
    (BUILD / "svg").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        artwork = list(pool.map(lambda n: convert_page(n, executable), range(1, count + 1)))
    manifest, slides = [], []
    for number, (page, svg) in enumerate(zip(reader.pages, artwork), 1):
        transcript = page.extract_text() or ""
        title = "Linear Attention: Gated DeltaNet & Kimi Delta Attention" if number == 1 else transcript.splitlines()[0]
        if number == 1:
            transcript += "\nOriginal slides by Gaotang Li\nLLM Systems Study Group · Week 5"
        section = section_name(number)
        slides.append(
            f'<section class="slide" id="slide-{number}" data-title="{escape(title, quote=True)}" '
            f'data-section="{escape(section, quote=True)}" {"hidden" if number > 1 else ""}>'
            + svg + f'<div class="slide-transcript">{escape(transcript)}</div></section>'
        )
        manifest.append({"number": number, "title": title, "section": section})
    css = (ASSETS / "base.css").read_text() + '''
    .slide-transcript{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip-path:inset(50%);white-space:pre-wrap;border:0}
    @media print{.slide-transcript{display:none!important}}
    '''
    chrome = f'''<nav class="deck-chrome" aria-label="Presentation controls">
<div class="chrome-left"><button id="prev" type="button" aria-label="Previous slide">←</button><span id="counter" aria-live="polite">1 / {count}</span><button id="next" type="button" aria-label="Next slide">→</button><span id="slide-title"></span></div>
<div class="chrome-right"><button id="overview-toggle" type="button">Slides</button><button id="fullscreen" type="button">Full screen</button><button id="print" type="button">Print / PDF</button><button id="help-toggle" type="button" aria-label="Keyboard help">?</button></div></nav>
<section id="overview-panel" class="deck-overlay" hidden><div class="panel-header"><h2>{count} slides</h2><button data-close-overlay="true" type="button">Close</button></div><div id="overview-list"></div></section>
<section id="help-panel" class="deck-overlay" hidden><div class="panel-header"><h2>Presentation controls</h2><button data-close-overlay="true" type="button">Close</button></div><p>Arrow keys or Space: advance. Home / End: first / last slide. Escape: close a panel.</p><p>Use Slides to jump to a topic, Full screen to present, or Print / PDF to export all pages.</p><p>Original slides by Gaotang Li. Linear Attention: Gated DeltaNet &amp; Kimi Delta Attention.</p></section>'''
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="author" content="Gaotang Li">'
            '<meta name="description" content="Week 5: Linear Attention, Gated DeltaNet and Kimi Delta Attention. Original slides by Gaotang Li.">'
            '<title>Linear Attention · Week 5 · Gaotang Li</title><style>' + css
            + '</style></head><body><main id="viewport" aria-label="Week 5 presentation"><div id="stage">'
            + ''.join(slides) + '</div></main>' + chrome + '<script>'
            + (ASSETS / "navigation.js").read_text() + '</script></body></html>')
    OUT.write_text(html)
    (BUILD / "manifest.json").write_text(json.dumps({
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "author": "Gaotang Li", "slides": manifest,
    }, indent=2))
    print(f"Built {count} Week 5 slides ({OUT.stat().st_size / 1e6:.1f} MB), with Gaotang Li credited on page 1.")


if __name__ == "__main__":
    build()
