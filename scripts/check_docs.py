#!/usr/bin/env python3
"""Doc↔code alignment checker for DiffusionGemma-Lite.

Parses `file.py:Symbol` / `file.py:Class.method` anchors in the markdown
docs, resolves each against the working tree, and fails on unknown files
or symbols. Flags line-number citations (they rot) and broken intra-repo
markdown links. A `--coverage` mode additionally requires every public
symbol in `models/`, `training/`, `data/`, `inference/`, `utils/` to be
cited at least once.

Usage:
    python3 scripts/check_docs.py                # resolve all anchors
    python3 scripts/check_docs.py --coverage --links
    python3 -m pytest tests/test_doc_refs.py     # same gate under pytest
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOC_PATHS = [
    ROOT / "docs",
    ROOT / "README.md",
    ROOT / "DIFFUSION.md",
    ROOT / "AGENTS.md",
    ROOT / "SKILLS.md",
]
COVERAGE_MODULES = [
    "models/mask.py",
    "models/diffusion.py",
    "models/attention.py",
    "models/selfcond.py",
    "models/time_embed.py",
    "models/block.py",
    "models/transformer.py",
    "training/losses.py",
    "training/pretrain.py",
    "data/prepare_data.py",
    "data/dataset.py",
    "inference/generate.py",
    "inference/evaluate.py",
    "utils/checkpoint.py",
    "utils/logging.py",
    "utils/memory.py",
]
ANCHOR_RE = re.compile(r"([\w./-]+\.py):([A-Za-z_][A-Za-z0-9_.]*)")
# Math terms that legitimately contain "L<digits>" (L1 norm, L2 loss, …).
LINE_ANCHOR_ALLOW = re.compile(r"\bL[12](?:[-–]?[0-9])?\b")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

_MODULE_CACHE: dict[str, object | None] = {}


def _doc_files() -> list[Path]:
    files: list[Path] = []
    for p in DOC_PATHS:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.md")))
        elif p.exists():
            files.append(p)
    return files


def collect_anchors() -> list[tuple[Path, str, str]]:
    """Return (doc_path, file.py, symbol) triples from all scanned docs."""
    anchors: list[tuple[Path, str, str]] = []
    for doc in _doc_files():
        for m in ANCHOR_RE.finditer(doc.read_text(encoding="utf-8")):
            anchors.append((doc, m.group(1), m.group(2)))
    return anchors


def load_module(rel_path: str):
    """Import a repo module by relative path; returns (module, error)."""
    if rel_path in _MODULE_CACHE:
        mod = _MODULE_CACHE[rel_path]
        return (mod, None) if mod is not None else (None, f"previous import failure: {rel_path}")
    path = ROOT / rel_path
    if not path.exists():
        _MODULE_CACHE[rel_path] = None
        return None, f"unknown file: {rel_path}"
    dotted = ".".join(Path(rel_path).with_suffix("").parts)
    try:
        mod = importlib.import_module(dotted)
    except Exception as exc:  # noqa: BLE001 — report any import failure
        _MODULE_CACHE[rel_path] = None
        return None, f"import failed for {rel_path}: {type(exc).__name__}: {exc}"
    _MODULE_CACHE[rel_path] = mod
    return mod, None


def _has_instance_attr(cls, name: str) -> bool:
    """Instance attrs assigned via self.name = … are invisible to hasattr on the class."""
    try:
        import inspect
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return False
    return re.search(rf"self\.{re.escape(name)}\s*=", src) is not None


def resolve_anchor(rel_path: str, symbol: str) -> tuple[bool, str | None]:
    mod, err = load_module(rel_path)
    if err:
        return False, err
    obj = mod
    for part in symbol.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
            continue
        if isinstance(obj, type) and _has_instance_attr(obj, part):
            continue
        return False, f"{rel_path}:{symbol} — '{part}' not found"
    return True, None


def check_resolution() -> list[str]:
    failures = []
    for doc, rel_path, symbol in collect_anchors():
        if rel_path == "file.py":
            continue  # literal metasyntax placeholder in the writing contract
        ok, err = resolve_anchor(rel_path, symbol)
        if not ok:
            failures.append(f"{doc.relative_to(ROOT)}: {err}")
    return failures


def public_symbols(rel_path: str) -> list[str]:
    mod, err = load_module(rel_path)
    if err:
        return []
    return sorted(
        n for n in dir(mod)
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", n)
        and not n.startswith("_")
        and callable(getattr(mod, n))
        and getattr(mod, n).__module__ == mod.__name__
    )


def check_coverage() -> list[str]:
    """Every public module-level symbol in COVERAGE_MODULES must be cited once."""
    cited = {f"{rel}:{sym.split('.')[0]}" for _, rel, sym in collect_anchors()}
    missing = []
    for rel_path in COVERAGE_MODULES:
        for sym in public_symbols(rel_path):
            if f"{rel_path}:{sym}" not in cited:
                missing.append(f"{rel_path}:{sym}")
    return missing


def check_line_anchors() -> list[str]:
    """Flag line-number citations (file.py:123, L15–32) — they rot after refactors."""
    line_re = re.compile(
        r"(?<![A-Za-z_])L[0-9]{2,}(?:\s*[-–]\s*[0-9]{2,})?"
        r"|(?<![A-Za-z_])[A-Za-z_][A-Za-z0-9_./-]*\.py\s*:\s*[0-9]{2,}")
    hits = []
    for doc in _doc_files():
        text = doc.read_text(encoding="utf-8")
        for m in line_re.finditer(text):
            if LINE_ANCHOR_ALLOW.search(m.group(0)):
                continue
            hits.append(f"{doc.relative_to(ROOT)}: {m.group(0)!r}")
    return hits


def check_links() -> list[str]:
    """Validate intra-repo markdown links; code fences are stripped first."""
    broken = []
    for doc in _doc_files():
        text = FENCE_RE.sub("", doc.read_text(encoding="utf-8"))
        for m in LINK_RE.finditer(text):
            target = m.group(1).strip()
            if not target or target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path_part = target.split("#", 1)[0]
            if not path_part:
                continue
            candidates = [(doc.parent / path_part).resolve(), (ROOT / path_part).resolve()]
            if not any(c.exists() for c in candidates):
                broken.append(f"{doc.relative_to(ROOT)}: broken link -> {target}")
    return broken


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Doc↔code alignment checker")
    ap.add_argument("--coverage", action="store_true", help="also enforce public-symbol coverage")
    ap.add_argument("--links", action="store_true", help="also validate intra-repo markdown links")
    args = ap.parse_args()

    failures = check_resolution()
    print(f"[doc-refs] scanned {len(_doc_files())} docs, {len(collect_anchors())} anchors")
    for f in failures:
        print(f"  FAIL {f}")
    print(f"[doc-refs] resolution: {'PASS' if not failures else f'{len(failures)} FAILURES'}")

    missing = check_coverage() if args.coverage else []
    if args.coverage:
        for m in missing:
            print(f"  UNCOVERED {m}")
        print(f"[doc-refs] coverage: {'PASS' if not missing else f'{len(missing)} UNCOVERED'}")

    line_hits = check_line_anchors()
    for h in line_hits:
        print(f"  LINE-ANCHOR {h}")
    print(f"[doc-refs] line anchors: {'PASS' if not line_hits else f'{len(line_hits)} FOUND'}")

    link_broken = check_links() if args.links else []
    if args.links:
        for b in link_broken:
            print(f"  BROKEN-LINK {b}")
        print(f"[doc-refs] links: {'PASS' if not link_broken else f'{len(link_broken)} BROKEN'}")

    return 1 if (failures or (args.coverage and missing) or line_hits
                 or (args.links and link_broken)) else 0


if __name__ == "__main__":
    sys.exit(main())