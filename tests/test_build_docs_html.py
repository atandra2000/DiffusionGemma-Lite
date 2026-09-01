"""HTML docs build gate: every DOC_FILES page + index + assets materialize."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_docs_html  # noqa: E402


def test_build_produces_all_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(build_docs_html, "OUTPUT_DIR", tmp_path)
    build_docs_html.main()
    out = tmp_path
    # every doc page + the portal + the shell assets
    for rel, _, _ in build_docs_html.DOC_FILES:
        assert (tmp_path / rel.replace(".md", ".html")).exists(), rel
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "assets" / "style.css").exists()
    assert (tmp_path / "assets" / "portal.js").exists()
    assert (tmp_path / ".nojekyll").exists()
    # pages are non-trivial and carry the shared shell
    idx = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "DiffusionGemma-Lite" in idx and "portal-card" in idx
    for rel, _, _ in build_docs_html.DOC_FILES:
        page = (tmp_path / rel.replace(".md", ".html")).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in page and "sidebar" in page
        assert len(page) > 1000


def test_build_html_portal_on_disk():
    """The committed workflow builds into docs_html/ — a real run must succeed."""
    subprocess = __import__("subprocess")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_docs_html.py")],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (ROOT / "docs_html" / "index.html").exists()