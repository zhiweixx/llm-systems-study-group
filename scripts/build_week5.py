"""Restyle Gaotang Li's PDF in the study group's existing HTML slide template.

Requires pypdf, pdfplumber and Poppler's pdftocairo. Preserve the source body
artwork, including equations and figures, inside the Week 4 slide frame.
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
from xml.etree import ElementTree as ET

import pdfplumber
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sources/week-5/Linear_Attention_gaotang_li.pdf"
OUT = ROOT / "week-5-linear-attention.html"
BUILD = ROOT / ".build/week5"
ASSETS = ROOT / "site/slide-assets/week4"
BLUE, INK, MUTED, LINE = "#245675", "#172329", "#58656d", "#aec3d1"
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def text(x, y, value, size=30, color=INK, weight=400, attrs=""):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" {attrs}>{escape(value)}</text>')


def line(y, color=LINE):
    return f'<line x1="75" y1="{y}" x2="1525" y2="{y}" stroke="{color}" stroke-width="1.5"/>'


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


def source_body(number, svg):
    """Remove the original template, retaining only the PDF's body artwork.

    In this source, glyph runs are positioned directly in page coordinates,
    while image uses carry a matrix transform. No body object touches the
    title/logo area (above 58pt) or the citation banner (below 360pt).
    """
    root = ET.fromstring(svg)
    surface = list(root)[-1]
    for element in list(surface):
        glyphs = [e for e in element.iter() if e.tag.endswith("}use") and "y" in e.attrib]
        if glyphs:
            ys = [float(e.attrib["y"]) for e in glyphs]
            if max(ys) < 58 or min(ys) > 360:
                surface.remove(element)
                continue
            # The PDF refers readers to notes that were not included in it.
            # Keep the actual references; omit this unavailable-notes pointer.
            if number == 49 and min(ys) > 325:
                surface.remove(element)
                continue
        if element.tag.endswith("}use") and "transform" in element.attrib:
            matrix = re.fullmatch(r"matrix\(([^)]+)\)", element.attrib["transform"])
            if matrix:
                components = [float(v) for v in re.split(r"[,\s]+", matrix[1]) if v]
                if components[5] < 58 or components[5] > 360:
                    # Removes the full-page background and institutional logo.
                    surface.remove(element)

    # Drop unreachable definitions, including the removed template images.
    definitions = {e.attrib["id"]: e for e in root.find(f"{{{SVG_NS}}}defs").iter() if "id" in e.attrib}
    def references(element):
        refs = set()
        for e in element.iter():
            for key, value in e.attrib.items():
                if key.endswith("href") and value.startswith("#"):
                    refs.add(value[1:])
                refs.update(re.findall(r"url\(#([^)]+)\)", value))
        return refs
    reachable = references(surface)
    pending = list(reachable)
    while pending:
        key = pending.pop()
        if key in definitions:
            for ref in references(definitions[key]) - reachable:
                reachable.add(ref)
                pending.append(ref)
    defs = root.find(f"{{{SVG_NS}}}defs")
    for parent in list(defs.iter()):
        for child in list(parent):
            if "id" in child.attrib and child.attrib["id"] not in reachable:
                parent.remove(child)

    svg = ET.tostring(root, encoding="unicode")
    for original, replacement in {
        "rgb(7.058716%,16.078186%,29.411316%)": BLUE,
        "rgb(77.645874%,41.960144%,16.078186%)": BLUE,
        "rgb(8.235168%,8.235168%,8.235168%)": INK,
        "rgb(12.548828%,12.548828%,12.548828%)": INK,
        "rgb(47.058105%,50.587463%,54.901123%)": MUTED,
        "rgb(92.939758%,94.900513%,97.253418%)": "#edf3f7",
    }.items():
        svg = svg.replace(original, replacement)
    return svg


def convert_page(number, executable):
    if number == 1:
        return ""
    target = BUILD / "svg" / f"page-{number:02}.svg"
    subprocess.run(
        [executable, "-svg", "-noshrink", "-nocenter", "-f", str(number), "-l", str(number), str(SOURCE), str(target)],
        check=True, capture_output=True,
    )
    svg = source_body(number, target.read_text())
    # Cairo reuses glyph/clip IDs on every page. Make every definition unique
    # before embedding all pages in the same HTML document.
    prefix = f"p{number}-"
    svg = re.sub(r'id="([^"]+)"', lambda m: f'id="{prefix}{m[1]}"', svg)
    svg = re.sub(r'(?:xlink:)?href="#([^"]+)"', lambda m: f'href="#{prefix}{m[1]}"', svg)
    svg = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{prefix}{m[1]})', svg)
    # Use ordinary HTML-compatible href for embedded images as well.
    svg = svg.replace("xlink:href=", "href=")
    svg = re.sub(r'width="[^"]+"', 'width="1450"', svg, count=1)
    svg = re.sub(r'height="[^"]+"', f'height="{302 * 1450 / 664:.6f}"', svg, count=1)
    svg = re.sub(r'viewBox="[^"]+"', 'viewBox="28 58 664 302"', svg, count=1)
    # Accessible content is supplied by the section transcript below.
    svg = svg.replace("<svg ", '<svg class="source-body" x="75" y="171" overflow="hidden" aria-hidden="true" focusable="false" ', 1)
    return svg


def frame(number, title, body, citations, count):
    if number == 1:
        title = "Linear Attention"
        body = (text(75, 244, "LLM Systems Study Group · Week 5", 29, MUTED)
                + text(75, 350, "Gated DeltaNet &", 62, BLUE, 700)
                + text(75, 426, "Kimi Delta Attention", 62, BLUE, 700)
                + line(494)
                + text(75, 565, "Original slides by Gaotang Li", 36, INK, 700, 'id="author-credit"'))
        citations = ["Linear attention, recurrent memory, and chunkwise parallelism"]
    # The source references slide incorrectly points to unavailable PDF notes.
    if number == 49:
        citations = ["Original slides by Gaotang Li"]
    footer = line(838, "#bdc7cc")
    for index, citation in enumerate(citations):
        footer += text(75, 858 + index * 20, citation, 18, MUTED, attrs='class="source-reference"')
    footer += text(1525, 873, f"{number} / {count}", 20, MUTED, attrs='text-anchor="end" class="slide-number"')
    return (f'<svg xmlns="{SVG_NS}" viewBox="0 0 1600 900" width="1600" height="900" '
            f'role="img" aria-labelledby="title-{number}"><title id="title-{number}">{escape(title)}</title>'
            '<rect width="1600" height="900" fill="white"/>'
            + text(75, 98, title, 46, BLUE, 700, 'class="slide-heading"')
            + line(132) + body + footer + '</svg>')


def build():
    executable = shutil.which("pdftocairo")
    if not executable:
        raise SystemExit("Install Poppler (pdftocairo) to regenerate the Week 5 deck.")
    reader = PdfReader(SOURCE)
    with pdfplumber.open(SOURCE) as pdf:
        source_citations = [[line["text"] for line in page.extract_text_lines()
                            if line["top"] >= 365 and line["x0"] < 690]
                           for page in pdf.pages]
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
            transcript = transcript.replace("IDEA–ISAIL Reading Group", "").strip()
            transcript += "\nOriginal slides by Gaotang Li\nLLM Systems Study Group · Week 5"
        if number == 49:
            transcript = transcript.replace("Additional references and source URLs appear in the slide notes.", "")
            transcript = transcript.replace("Full source links and supplementary references are included in the notes.", "")
            transcript += "\nOriginal slides by Gaotang Li"
        section = section_name(number)
        svg = frame(number, title, svg, source_citations[number-1], count)
        slides.append(
            f'<section class="slide" id="slide-{number}" data-title="{escape(title, quote=True)}" '
            f'data-section="{escape(section, quote=True)}" {"hidden" if number > 1 else ""}>'
            + svg + f'<div class="slide-transcript">{escape(transcript)}</div></section>'
        )
        manifest.append({"number": number, "title": title, "section": section})
    css = (ASSETS / "base.css").read_text() + '''
    svg text{font-family:Arial,Helvetica,sans-serif}
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
    print(f"Built {count} Week 5 slides in the study-group template ({OUT.stat().st_size / 1e6:.1f} MB), with Gaotang Li credited on page 1.")


if __name__ == "__main__":
    build()
