"""Validate generated routes, anchors, source integrity, and publication boundaries."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.path, self.ids, self.links, self.title = path, set(), [], False
        self.resources = []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            assert attrs["id"] not in self.ids, f"Duplicate id in {self.path.name}"
            self.ids.add(attrs["id"])
        if tag == "title":
            self.title = True
        for key in ("href", "src"):
            if key in attrs:
                self.links.append(attrs[key])
                if key == "src" or tag == "link" or (tag == "image" and key == "href"):
                    self.resources.append(attrs[key])


def main():
    pages = {p.resolve(): Page(p) for p in OUT.rglob("*.html")}
    expected_routes = {
        "index.html", "curriculum.html", "references.html",
        "week-1/slides.html", "week-1/cheatsheet.html", "week-1/speaker-notes.html",
        "week-2/slides.html", "week-2/speaker-notes.html", "week-2/lab.html",
        "week-3/slides.html", "week-3/speaker-notes.html",
        "week-4/prefix-cache.html",
    }
    assert {str(path.relative_to(OUT.resolve())) for path in pages} == expected_routes, "Unexpected or missing published HTML routes"
    checked = 0
    for path, page in pages.items():
        assert page.title, f"Missing page title: {path.name}"
        text = path.read_text()
        assert not re.search(r"file://|/Users/|/var/folders/|localhost:\d|127\.0\.0\.1:\d", text), f"Local-only reference in {path.name}"
        assert not re.search(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}", text), f"Unexpected credential-shaped content in {path.name}"
        assert "{{" not in text or path.name == "slides.html", f"Unrendered template: {path.name}"
        for link in page.links:
            url = urlsplit(link)
            if url.scheme in {"https", "http", "mailto", "data"} or url.netloc:
                continue
            assert not url.scheme, f"Unexpected link scheme in {path.name}"
            dest = (path.parent / unquote(url.path)).resolve() if url.path else path
            assert dest.is_relative_to(OUT.resolve()), f"Link escapes published output: {path.name}"
            if dest.is_dir():
                dest /= "index.html"
            assert dest.is_file(), f"Missing target: {path.name} -> {link}"
            if url.fragment and dest in pages:
                assert unquote(url.fragment) in pages[dest].ids, f"Missing anchor: {path.name} -> {link}"
            checked += 1
    for source_name, route, slide_count in (
            ("week-1-gpu-memory-short.html", "week-1/slides.html", 24),
            ("week-2-inference.html", "week-2/slides.html", 31),
            ("week-3-parallelism.html", "week-3/slides.html", 45)):
        source = (ROOT / source_name).read_bytes()
        assert (OUT / route).read_bytes() == source, f"Slide output differs from source: {route}"
        slide_page = pages[(OUT / route).resolve()]
        slide_ids = {i for i in slide_page.ids if re.fullmatch(r"slide-\d+", i)}
        assert slide_ids == {f"slide-{n}" for n in range(1, slide_count + 1)}, f"Unexpected slide IDs: {route}"
        assert all(resource.startswith(("data:", "#")) for resource in slide_page.resources), f"Slides require external resources: {route}"
    assert "09/10/26" in (OUT / "week-1/slides.html").read_text()
    for name in ("model.py", "benchmark.py", "plot_results.py", "week2-inference-lab.ipynb"):
        assert (OUT / "week-2/lab" / name).read_bytes() == (ROOT / "week-2-lab" / name).read_bytes(), f"Lab output differs from source: {name}"
    assert (OUT / "week-3/tp-exercise.py").read_bytes() == (ROOT / "week-3-tp-exercise.py").read_bytes(), "Week 3 exercise output differs from source"
    assert (OUT / ".nojekyll").exists()
    print(f"Validated {len(pages)} HTML pages, {checked} local links/anchors, 24 Week 1 slides, 31 Week 2 slides, 45 Week 3 slides, exercise downloads, and publication boundaries.")


if __name__ == "__main__":
    main()
