"""Doc↔code alignment gate under pytest (logic + CLI live in scripts/check_docs.py)."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_docs  # noqa: E402


def test_doc_refs_all_anchors_resolve():
    failures = check_docs.check_resolution()
    assert not failures, "\n".join(failures)


def test_doc_refs_no_line_anchors():
    hits = check_docs.check_line_anchors()
    assert not hits, "\n".join(hits)


def test_doc_refs_coverage():
    missing = check_docs.check_coverage()
    assert not missing, "uncited public symbols:\n" + "\n".join(missing)


def test_check_docs_cli_clean():
    """The plan's Phase-5 gate: `python scripts/check_docs.py` exits 0."""
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_docs.py"), "--coverage"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
