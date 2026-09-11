"""Build a static GitHub Pages site from the selected teaching materials."""
from pathlib import Path
from html import escape
import shutil
import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
TEMPLATE = (ROOT / "site/page.html").read_text()
LAB_FILES = ("model.py", "benchmark.py", "plot_results.py", "week2-inference-lab.ipynb")
LAB_OPTIONAL_FILES = ("test_lab.py",)


def page(title, content, target, prefix="", toc=""):
    html = (TEMPLATE.replace("{{TITLE}}", escape(title))
            .replace("{{ROOT}}", prefix)
            .replace("{{TOC}}", toc)
            .replace("{{CONTENT}}", content))
    dest = OUT / target
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html)


def document(source, title, target, prefix="", link_targets=None):
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"],
                           extension_configs={"toc": {"toc_depth": "2-2"}})
    content = md.convert((ROOT / source).read_text())
    for original, published in (link_targets or {}).items():
        content = content.replace(f'href="{escape(original, quote=True)}"',
                                  f'href="{escape(published, quote=True)}"')
    content = content.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    download = OUT / "downloads" / source
    download.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / source, download)
    actions = (f'<p class="document-actions"><a href="{prefix}downloads/{source}" download>Download Markdown</a>'
               ' · Use your browser’s Print command to save a PDF.</p>')
    first_heading = content.find("</h1>") + len("</h1>")
    content = content[:first_heading] + actions + content[first_heading:]
    toc = f'<div class="toc-container"><p class="nav-label">On this page</p>{md.toc}</div>' if md.toc_tokens else ""
    page(title, content, target, prefix, toc)


def main():
    # Only the generated output directory is replaced.
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "assets").mkdir(parents=True)
    shutil.copy2(ROOT / "site/styles.css", OUT / "assets/styles.css")
    page("Overview", (ROOT / "site/index.html").read_text(), "index.html")
    document("curriculum.md", "Curriculum", "curriculum.html")
    document("resources.md", "References", "references.html")
    document("week-1-llm-systems-cheatsheet.md", "Week 1 cheatsheet", "week-1/cheatsheet.html", "../")
    document("week-1-short-speaker-notes.md", "Week 1 speaker notes", "week-1/speaker-notes.html", "../")
    document("week-2-speaker-notes.md", "Week 2 speaker notes", "week-2/speaker-notes.html", "../")
    lab_files = [*LAB_FILES, *(name for name in LAB_OPTIONAL_FILES
                             if (ROOT / "week-2-lab" / name).is_file())]
    lab_out = OUT / "week-2/lab"
    lab_out.mkdir(parents=True)
    for name in lab_files:
        shutil.copy2(ROOT / "week-2-lab" / name, lab_out / name)
    document("week-2-lab/README.md", "Week 2 benchmark lab", "week-2/lab.html", "../",
             link_targets={name: f"lab/{name}" for name in lab_files})
    document("week-4-prefix-cache-notes.md", "Week 4 prefix-cache notes", "week-4/prefix-cache.html", "../")
    shutil.copy2(ROOT / "week-1-gpu-memory-short.html", OUT / "week-1/slides.html")
    shutil.copy2(ROOT / "week-2-inference.html", OUT / "week-2/slides.html")
    (OUT / ".nojekyll").touch()
    print(f"Built {len(list(OUT.rglob('*.html')))} HTML pages in _site/")


if __name__ == "__main__":
    main()
