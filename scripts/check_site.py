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


def main():
    pages = {p.resolve(): Page(p) for p in OUT.rglob("*.html")}
    assert len(pages) == 7, "Expected course pages, Week 1 materials, and Week 4 prefix-cache notes"
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
    source = (ROOT / "week-1-gpu-memory-short.html").read_bytes()
    assert (OUT / "week-1/slides.html").read_bytes() == source, "Slide output differs from source"
    slide_page = pages[(OUT / "week-1/slides.html").resolve()]
    slide_ids = {i for i in slide_page.ids if re.fullmatch(r"slide-\d+", i)}
    assert slide_ids == {f"slide-{n}" for n in range(1, 24)}
    assert "09/10/26" in source.decode()
    assert (OUT / ".nojekyll").exists()
    print(f"Validated {len(pages)} HTML pages, {checked} local links/anchors, 23 slides, and publication boundaries.")


if __name__ == "__main__":
    main()
