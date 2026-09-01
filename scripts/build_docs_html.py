#!/usr/bin/env python3
"""
DiffusionGemma-Lite Documentation Generator
Converts project markdown files into a responsive, beautifully-styled HTML documentation portal
with full LaTeX math (KaTeX) and syntax highlighting support.
Output directory: docs_html/ (ignored by git).

Design system: the "dark bench notebook" — espresso-graphite paper, warm-bone
ink, terracotta + olive marks, one mono voice. Shares its lineage with the
sibling LLaMA/DeepSeek/Mamba-3 Lite portals. See `assets/style.css`.
"""
import os
import re
import html
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE_DIR / "docs_html"

DOC_FILES = [
    # (relative_path_from_root, category, display_title)
    ("README.md", "Core", "Project Overview (README)"),
    ("AGENTS.md", "Core", "AGENTS & System Architecture"),
    ("SKILLS.md", "Core", "Skills Reference"),
    ("DIFFUSION.md", "Core", "DIFFUSION — the authoritative technical doc"),
    ("docs/README.md", "Core", "Documentation Index"),

    # Concepts
    ("docs/concepts/diffusion-core.md", "Concepts", "Diffusion Core — Uniform State, x0, Chunked Loss"),
    ("docs/concepts/block-causal-attention.md", "Concepts", "Block-Causal Attention — the load-bearing mask"),
    ("docs/concepts/self-conditioning.md", "Concepts", "Self-Conditioning — zero-init posterior feedback"),

    # Guides
    ("docs/guides/quickstart.md", "Guides", "Quickstart — data, train, sample, evaluate"),
    ("docs/guides/debugging-playbook.md", "Guides", "Debugging Playbook — symptom-first recipes"),

    # References
    ("docs/references/config.md", "References", "Config Reference — the production YAML"),
    ("docs/references/api.md", "References", "API Reference — public surface, symbol-anchored"),
]

# Premium-polish assets: secondary mono font, boot overlay, and a shared
# portal.js that holds the hero, pass-diagram, sidebar, copy/expand, and
# TOC scrollspy logic. All pages reference the same shell.
FONT_LINK = ('<link href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400'
             '&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400'
             '&display=swap" rel="stylesheet">')
BOOT_OVERLAY_HTML = (
    '<div id="boot-overlay" aria-hidden="true">'
    '<div class="boot-inner">'
    '<div class="boot-wordmark">DIFFUSIONGEMMA-LITE</div>'
    '<div class="boot-line">loading weights '
    '<span class="boot-bar">[\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%</span>'
    '</div></div></div>'
)
BOOT_SCRIPT = """<script>
(function () {
    var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!reduced) document.documentElement.classList.add('booting');
    document.addEventListener('DOMContentLoaded', function () {
        setTimeout(function () {
            document.documentElement.classList.remove('booting');
            var ov = document.getElementById('boot-overlay');
            if (ov && ov.parentNode) ov.parentNode.removeChild(ov);
        }, reduced ? 0 : 350);
    });
})();
</script>"""

# Shared <head> for every generated page. Doc pages pass highlight.js +
# KaTeX in `extra_head`; the index portal passes "".
HEAD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- Fonts — secondary mono voice for headings/numerics, JetBrains for body. -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    {font_link}
    {boot_script}
{extra_head}
    <!-- CSS Stylesheet -->
    <link rel="stylesheet" href="{rel_prefix}assets/style.css">
</head>
"""

DOC_EXTRA_HEAD = """    <!-- Highlight.js for Syntax Highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <!-- KaTeX for LaTeX Math -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>"""

# Multi-Scale Attention Sinks & YaRN 128K Phase-Space (FIG · A0) — the
# index hero instrument. Sliding-window attention score heatmap with the
# Token 0..3 sink column glowing olive, the W=128 SWA diagonal band,
# 12 global YaRN sampling arcs, and a moving query token scanline.
# The live interactive canvas simulation lives in assets/portal.js.

# 12-Layer Alternating Pass & Top-2 MoE Grouped-GEMM Pipeline (FIG · A1) —
# one full training step. Visualizes the entire forward activation pass,
# autograd backward pass, gradient checkpointing re-computation, and the
# AdamW weight update step across 6 architectural stations (Embed → SWA →
# Global YaRN → MoE Router → Triton GEMM → Chunked CE + AdamW).
# The live interactive canvas simulation lives in assets/portal.js.

def slugify(text: str) -> str:
    """Generate clean HTML id for headings.

    Matches the anchor convention the docs were authored against (GitHub-style):
    lowercase, keep word chars + underscore + hyphen, drop everything else,
    and turn each space into a single hyphen (no run collapsing).
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s', '-', text)
    return text.strip('-') or "heading"


@lru_cache(maxsize=1)
def github_base_url() -> str:
    """Derive the GitHub blob base (https://github.com/<owner>/<repo>/blob/<branch>)."""
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True, cwd=WORKSPACE_DIR,
        ).stdout.strip()
        out = out.replace("git@github.com:", "https://github.com/").removesuffix(".git")
        if not out.startswith("https://github.com/"):
            return ""
    except Exception:
        return ""
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, check=True, cwd=WORKSPACE_DIR,
        ).stdout.strip()
    except Exception:
        return ""
    return f"{out}/blob/{branch}" if branch else ""


def fix_md_links(content: str, src_rel_path: str) -> str:
    """Rewrite relative markdown links for the HTML build.

    - ``.md`` links (with optional ``#anchor``) -> ``.html`` twin.
    - non-``.md`` repo-relative links -> GitHub blob URL (the file isn't
      shipped inside ``docs_html/``).
    """
    repo_base = github_base_url()
    src_dir = WORKSPACE_DIR / Path(src_rel_path).parent

    def link_replacer(match):
        label = match.group(1)
        url = match.group(2)
        if url.startswith(("http://", "https://", "mailto:", "#")):
            return f"[{label}]({url})"
        path_part, _, anchor = url.partition("#")
        if path_part.endswith(".md"):
            # Resolve the target to a repo-relative .md path. Docs mix two
            # styles: repo-root-relative (``docs/...``) and file-relative
            # (``../...`` / ``basename.md``), so try both and keep the hit.
            if path_part.startswith("/"):
                repo_rel = Path(path_part[1:])
            else:
                cand_src = src_dir / path_part
                cand_root = WORKSPACE_DIR / path_part
                if cand_src.exists():
                    repo_rel = cand_src.resolve().relative_to(WORKSPACE_DIR)
                elif cand_root.exists():
                    repo_rel = cand_root
                else:
                    # Prefer src-relative; falls through to a relative href
                    # the build still ships (harmless if unresolved).
                    repo_rel = Path(src_rel_path).parent / path_part
            # Emit a href relative to the current output file's directory.
            rel = os.path.relpath(WORKSPACE_DIR / repo_rel, src_dir).replace(os.sep, '/')
            target = rel[:-3] + ".html"
            if anchor:
                target += "#" + anchor
            return f"[{label}]({target})"
        if repo_base and not path_part.startswith("/"):
            repo_rel = (src_dir / path_part).resolve().relative_to(WORKSPACE_DIR)
            return f"[{label}]({repo_base}/{repo_rel})"
        return f"[{label}]({url})"

    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link_replacer, content)


def parse_markdown_to_html(md_text: str, src_rel_path: str) -> tuple[str, list[dict]]:
    """Statically convert markdown to rich HTML structure with full LaTeX & Math protection.

    Returns (html_content, toc_items).
    """
    md_text = fix_md_links(md_text, src_rel_path)

    # STEP 1: Protect Code Blocks & Inline Code
    code_blocks = []
    def store_code_block(m):
        code_blocks.append(m.group(0))
        return f"\n\n___CODEBLOCK_{len(code_blocks)-1}___\n\n"

    md_text = re.sub(r'```[\s\S]*?```', store_code_block, md_text)

    inline_codes = []
    def store_inline_code(m):
        inline_codes.append(m.group(0))
        return f"___INLINECODE_{len(inline_codes)-1}___"

    md_text = re.sub(r'`[^`\n]+`', store_inline_code, md_text)

    # STEP 2: Protect LaTeX Math Blocks & Inline Math
    display_maths = []
    def store_display_math(m):
        inner = m.group(1).strip()
        safe_math = html.escape(inner, quote=False)
        display_maths.append(f'<div class="math-block">$$\n{safe_math}\n$$</div>')
        return f"\n\n___DISPLAYMATH_{len(display_maths)-1}___\n\n"

    md_text = re.sub(r'\$\$([\s\S]+?)\$\$', store_display_math, md_text)
    md_text = re.sub(r'\\\[([\s\S]+?)\\\]', store_display_math, md_text)

    inline_maths = []
    def store_inline_math(m):
        inner = m.group(1).strip()
        safe_math = html.escape(inner, quote=False)
        inline_maths.append(f'<span class="math-inline">${safe_math}$</span>')
        return f"___INLINEMATH_{len(inline_maths)-1}___"

    md_text = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', store_inline_math, md_text)
    md_text = re.sub(r'\\\(([\s\S]+?)\\\)', store_inline_math, md_text)

    # STEP 3: Parse Document Structure Line by Line
    toc = []
    seen_slugs = {}
    lines = md_text.splitlines()
    html_lines = []
    in_table = False
    table_headers = []
    table_rows = []

    list_stack = []  # [{'indent': int, 'tag': str, 'li_open': bool}]
    h1_seen = False  # the first H1 duplicates the page's doc-title; suppressed

    in_blockquote = False
    blockquote_type = "normal"
    blockquote_lines = []

    def close_li():
        nonlocal list_stack
        if list_stack and list_stack[-1]['li_open']:
            html_lines.append("</li>")
            list_stack[-1]['li_open'] = False

    def flush_list():
        nonlocal list_stack
        while list_stack:
            close_li()
            html_lines.append(f"</{list_stack[-1]['tag']}>")
            list_stack.pop()

    def flush_blockquote():
        nonlocal in_blockquote, blockquote_type, blockquote_lines
        if in_blockquote:
            content = "<br>".join(blockquote_lines)
            if blockquote_type != "normal":
                title = blockquote_type.upper()
                icon = {"NOTE": "ℹ️", "TIP": "💡", "IMPORTANT": "📌", "WARNING": "⚠️", "CAUTION": "🚨"}.get(title, "ℹ️")
                html_lines.append(
                    f'<div class="callout callout-{blockquote_type.lower()}">'
                    f'<div class="callout-header"><span class="callout-icon">{icon}</span><span class="callout-title">{title}</span></div>'
                    f'<div class="callout-body">{content}</div>'
                    f'</div>'
                )
            else:
                html_lines.append(f'<blockquote>{content}</blockquote>')
            in_blockquote = False
            blockquote_type = "normal"
            blockquote_lines = []

    def flush_table():
        nonlocal in_table, table_headers, table_rows
        if in_table:
            th_html = "".join(f"<th>{h}</th>" for h in table_headers)
            tr_html = ""
            for row in table_rows:
                td_html = "".join(f"<td>{c}</td>" for c in row)
                tr_html += f"<tr>{td_html}</tr>"
            html_lines.append(
                f'<div class="table-container"><table class="doc-table">'
                f'<thead><tr>{th_html}</tr></thead>'
                f'<tbody>{tr_html}</tbody>'
                f'</table></div>'
            )
            in_table = False
            table_headers = []
            table_rows = []

    def escape_preserving_entities(text: str) -> str:
        """Escape `<`, `>` and stray `&`, but leave valid HTML entities intact."""
        text = re.sub(r'&(?!(?:[a-zA-Z][a-zA-Z0-9]*|#[0-9]+|#x[0-9a-fA-F]+);)', '&amp;', text)
        return text.replace('<', '&lt;').replace('>', '&gt;')

    def render_inline_formatting(text: str) -> str:
        text = escape_preserving_entities(text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        text = re.sub(r'~~([^~]+)~~', r'<del>\1</del>', text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" class="doc-link">\1</a>', text)
        return text

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("___DISPLAYMATH_") or stripped.startswith("___CODEBLOCK_"):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append(stripped)
            i += 1
            continue

        if not stripped:
            flush_table()
            flush_list()
            flush_blockquote()
            i += 1
            continue

        # Blockquote or Callout
        if stripped.startswith(">"):
            flush_table()
            flush_list()
            bq_content = stripped.lstrip(">").strip()
            callout_match = re.match(r'^\[\!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]', bq_content, re.IGNORECASE)
            if callout_match:
                in_blockquote = True
                blockquote_type = callout_match.group(1).upper()
                remaining = bq_content[callout_match.end():].strip()
                if remaining:
                    blockquote_lines.append(render_inline_formatting(remaining))
            else:
                if not in_blockquote:
                    in_blockquote = True
                    blockquote_type = "normal"
                if bq_content:
                    blockquote_lines.append(render_inline_formatting(bq_content))
            i += 1
            continue

        # Horizontal Rule
        if re.match(r'^(---|\*\*\*|___)\s*$', stripped):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append("<hr class='doc-hr'>")
            i += 1
            continue

        # Headings
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if heading_match:
            flush_table()
            flush_list()
            flush_blockquote()
            level = len(heading_match.group(1))
            heading_text_raw = heading_match.group(2).strip()

            clean_title = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', heading_text_raw)
            clean_title = re.sub(r'`([^`]+)`', r'\1', clean_title)
            clean_title = re.sub(r'___INLINECODE_\d+___', '', clean_title)
            clean_title = re.sub(r'___INLINEMATH_\d+___', '', clean_title)
            raw_slug = slugify(clean_title)
            if raw_slug in seen_slugs:
                seen_slugs[raw_slug] += 1
                heading_id = f"{raw_slug}-{seen_slugs[raw_slug]}"
            else:
                seen_slugs[raw_slug] = 0
                heading_id = raw_slug
            rendered_heading = render_inline_formatting(heading_text_raw)

            if level in (2, 3):
                toc.append({'level': level, 'title': clean_title, 'id': heading_id})

            if level == 1 and not h1_seen:
                h1_seen = True
                html_lines.append(f'<span class="doc-anchor" id="{heading_id}"></span>')
            else:
                html_lines.append(
                    f'<h{level} id="{heading_id}" class="heading-anchor">'
                    f'{rendered_heading}'
                    f'<a href="#{heading_id}" class="anchor-link" aria-label="Link to section">#</a>'
                    f'</h{level}>'
                )
            i += 1
            continue

        # Markdown Table Detection
        if "|" in line and i + 1 < len(lines) and re.match(r'^\s*\|?\s*:?---', lines[i + 1].strip()):
            flush_list()
            flush_blockquote()
            in_table = True
            headers_raw = [c.strip() for c in line.strip().strip("|").split("|")]
            table_headers = [render_inline_formatting(h) for h in headers_raw]
            i += 2

            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                cells_raw = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                table_rows.append([render_inline_formatting(c) for c in cells_raw])
                i += 1
            flush_table()
            continue

        # Lists
        ul_match = re.match(r'^[\*\-]\s+(.+)$', stripped)
        ol_match = re.match(r'^\d+\.\s+(.+)$', stripped)
        if ul_match or ol_match:
            flush_table()
            flush_blockquote()
            tag = 'ul' if ul_match else 'ol'
            item_text = (ul_match or ol_match).group(1).strip()
            indent = len(line) - len(line.lstrip(' '))

            while list_stack and indent < list_stack[-1]['indent']:
                close_li()
                html_lines.append(f"</{list_stack[-1]['tag']}>")
                list_stack.pop()
            if list_stack and list_stack[-1]['indent'] == indent and list_stack[-1]['tag'] != tag:
                close_li()
                html_lines.append(f"</{list_stack[-1]['tag']}>")
                list_stack.pop()
            if not list_stack or list_stack[-1]['indent'] != indent:
                list_stack.append({'indent': indent, 'tag': tag, 'li_open': False})
                html_lines.append(f'<{tag} class="doc-list">')
            else:
                close_li()

            task_match = re.match(r'^\[([ xX])\]\s+(.+)$', item_text)
            if task_match:
                checked = 'checked' if task_match.group(1).lower() == 'x' else ''
                item_content = render_inline_formatting(task_match.group(2))
                html_lines.append(f'<li class="task-item"><input type="checkbox" disabled {checked}> {item_content}')
            else:
                html_lines.append(f'<li>{render_inline_formatting(item_text)}')
            list_stack[-1]['li_open'] = True
            i += 1
            continue

        # Continuation of an open list item
        if list_stack and list_stack[-1]['li_open'] and line[:1] in (' ', '\t'):
            html_lines.append("<br> " + render_inline_formatting(stripped))
            i += 1
            continue

        # Standard Paragraph
        flush_table()
        flush_list()
        flush_blockquote()
        html_lines.append(f'<p>{render_inline_formatting(stripped)}</p>')
        i += 1

    flush_table()
    flush_list()
    flush_blockquote()

    full_html = "\n".join(html_lines)

    # STEP 4: Restore Protected Tokens
    for idx, math_html in enumerate(inline_maths):
        full_html = full_html.replace(f"___INLINEMATH_{idx}___", math_html)

    for idx, raw_code in enumerate(inline_codes):
        code_content = raw_code[1:-1]
        code_html = f'<code class="inline-code">{html.escape(code_content)}</code>'
        full_html = full_html.replace(f"___INLINECODE_{idx}___", code_html)

    for idx, math_html in enumerate(display_maths):
        full_html = full_html.replace(f"___DISPLAYMATH_{idx}___", math_html)

    for idx, raw_block in enumerate(code_blocks):
        lines_b = raw_block.splitlines()
        first_line = lines_b[0].strip()
        code_lang = first_line.lstrip("```").strip().lower()
        code_content = "\n".join(lines_b[1:-1])
        escaped_content = html.escape(code_content)

        lang_attr = f' class="language-{code_lang}"' if code_lang else ''
        data_lang = code_lang if code_lang else 'code'

        n_lines = len(lines_b) - 2
        collapsed = n_lines > 14
        wrapper_cls = 'code-wrapper collapsed' if collapsed else 'code-wrapper'
        expand_btn = (
            f'<button class="expand-btn" data-label="expand ▾ · {n_lines} lines" '
            f'onclick="toggleCode(this)">expand ▾ · {n_lines} lines</button>'
            if collapsed else ''
        )

        block_html = (
            f'<div class="{wrapper_cls}" data-lines="{n_lines}">'
            f'<div class="code-header">'
            f'<span class="code-lang">{data_lang}</span>'
            f'<span class="code-actions">{expand_btn}'
            f'<button class="copy-btn" onclick="copyCode(this)">Copy</button></span>'
            f'</div>'
            f'<pre><code{lang_attr}>{escaped_content}</code></pre>'
            f'</div>'
        )
        full_html = full_html.replace(f"___CODEBLOCK_{idx}___", block_html)

    return full_html, toc


def compute_rel_prefix(target_rel_path: str) -> str:
    """Calculate relative path back to root docs_html directory."""
    parts = Path(target_rel_path).parts
    if len(parts) <= 1:
        return "./"
    return "../" * (len(parts) - 1)


def build_sidebar_html(current_rel_path: str, rel_prefix: str) -> str:
    """Build the navigation sidebar HTML."""
    sidebar_sections = {"Core": [], "Concepts": [], "Guides": [], "References": []}

    for rel_path, category, display_title in DOC_FILES:
        target_html_rel = rel_path.replace(".md", ".html")
        href = rel_prefix + target_html_rel
        is_active = (rel_path == current_rel_path)
        active_cls = "active" if is_active else ""
        sidebar_sections[category].append(
            f'<li class="nav-item"><a href="{href}" class="nav-link {active_cls}" title="{display_title}"><span class="nav-link-text">{display_title}</span></a></li>'
        )

    html_out = ['<div class="sidebar-search"><input type="text" id="navSearch" placeholder="Search docs..." onkeyup="filterNav()"></div>']

    for cat_name, items in sidebar_sections.items():
        if items:
            html_out.append('<div class="nav-group">')
            html_out.append(f'<div class="nav-group-title">{cat_name}</div>')
            html_out.append(f'<ul class="nav-list">{"".join(items)}</ul>')
            html_out.append('</div>')

    return "\n".join(html_out)


def build_toc_html(toc_items: list[dict]) -> str:
    """Build the right sidebar table of contents."""
    if not toc_items:
        return '<div class="toc-empty">No section headings</div>'

    toc_links = []
    for item in toc_items:
        indent_cls = "toc-h3" if item['level'] == 3 else "toc-h2"
        toc_links.append(f'<li class="{indent_cls}"><a href="#{item["id"]}" class="toc-link">{item["title"]}</a></li>')

    return f'<ul class="toc-list">{"".join(toc_links)}</ul>'


def generate_html_page(rel_path: str, category: str, display_title: str):
    """Generate single HTML file for a markdown document."""
    src_file = WORKSPACE_DIR / rel_path
    if not src_file.exists():
        print(f"Warning: {src_file} does not exist, skipping.")
        return

    md_text = src_file.read_text(encoding="utf-8")

    word_count = len(md_text.split())
    reading_time = max(1, round(word_count / 200))

    html_body, toc_items = parse_markdown_to_html(md_text, rel_path)

    rel_prefix = compute_rel_prefix(rel_path)
    sidebar_html = build_sidebar_html(rel_path, rel_prefix)
    toc_html = build_toc_html(toc_items)

    current_idx = next((i for i, df in enumerate(DOC_FILES) if df[0] == rel_path), 0)
    prev_doc = DOC_FILES[current_idx - 1] if current_idx > 0 else None
    next_doc = DOC_FILES[current_idx + 1] if current_idx < len(DOC_FILES) - 1 else None

    prev_html = ""
    if prev_doc:
        prev_href = rel_prefix + prev_doc[0].replace(".md", ".html")
        prev_html = f'<a href="{prev_href}" class="nav-card prev-card"><span class="card-label">← Previous</span><span class="card-title">{prev_doc[2]}</span></a>'
    next_html = ""
    if next_doc:
        next_href = rel_prefix + next_doc[0].replace(".md", ".html")
        next_html = f'<a href="{next_href}" class="nav-card next-card"><span class="card-label">Next →</span><span class="card-title">{next_doc[2]}</span></a>'
    page_html = HEAD_TEMPLATE.format(
        rel_prefix=rel_prefix,
        title=f"{display_title} | DiffusionGemma-Lite Documentation",
        extra_head=DOC_EXTRA_HEAD,
        font_link=FONT_LINK,
        boot_script=BOOT_SCRIPT,
    ) + f"""<body>
    {BOOT_OVERLAY_HTML}
    <!-- Top Header -->
    <header class="site-header">
        <div class="header-left">
            <button class="mobile-toggle" onclick="toggleSidebar()" aria-label="Toggle Sidebar">☰</button>
            <a href="{rel_prefix}index.html" class="brand-logo">
                <span class="brand-name">DiffusionGemma-Lite</span>
                <span class="brand-badge">Documentation</span>
            </a>
        </div>
        <div class="header-right">
            <a href="{rel_prefix}index.html" class="header-link">Portal</a>
            <a href="{rel_prefix}README.html" class="header-link">README</a>
        </div>
    </header>

    <div class="app-layout">
        <!-- Left Sidebar Navigation -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-inner">
                {sidebar_html}
            </div>
        </aside>

        <!-- Main Content Area -->
        <main class="main-content">
            <div class="content-container">
                <div class="breadcrumb">
                    <a href="{rel_prefix}index.html">Docs</a> &gt; <span>{category}</span> &gt; <span class="current">{display_title}</span>
                </div>

                <div class="doc-header">
                    <h1 class="doc-title">{display_title}</h1>
                    <div class="doc-meta">
                        <span class="meta-item"><span class="meta-mark">&sect;</span> {rel_path}</span>
                        <span class="meta-item"><span class="meta-mark">&para;</span> {word_count:,} words</span>
                        <span class="meta-item"><span class="meta-mark">&tau;</span> ~{reading_time} min read</span>
                    </div>
                </div>

                <article class="markdown-body" id="articleBody">
                    {html_body}
                </article>

                <div class="doc-footer-nav">
                    {prev_html}
                    {next_html}
                </div>
            </div>
        </main>

        <!-- Right Sidebar Table of Contents -->
        <aside class="toc-sidebar">
            <div class="toc-inner">
                <div class="toc-title">On This Page</div>
                {toc_html}
            </div>
        </aside>
    </div>

    <!-- Scripts -->
    <script defer src="{rel_prefix}assets/portal.js"></script>
</body>
</html>
"""

    out_file = OUTPUT_DIR / rel_path.replace(".md", ".html")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(page_html, encoding="utf-8")


def generate_index_portal():
    """Generate interactive index.html home portal."""
    sidebar_html = build_sidebar_html("index.html", "./")

    categories = {
        ("CORE", "Core Documents"): [
            ("README.html", "README", "Project Overview", "343.5M-param block-AR discrete-diffusion LM — uniform-state canvases of 256 tokens, decoded in parallel."),
            ("DIFFUSION.html", "DIFFUSION", "Authoritative Technical Doc", "Diffusion math, mask derivation, self-conditioning, sampler semantics, training recipe, recipe deltas vs upstream."),
            ("AGENTS.html", "AGENTS", "Agent Contract", "Hard rules (pure PyTorch, chunked-CE, commit monotonicity), numerical-stability rules, known caveats."),
            ("SKILLS.html", "SKILLS", "Skills Map", "Day-to-day workflows: test suite, chunked-CE probe, sampler smoke, eval CLI, docs gates."),
            ("docs/README.html", "DOCS", "Documentation Index", "Map of the concepts, guides, and symbol-anchored references in this portal."),
        ],
        ("CONCEPTS", "Architecture & Concepts"): [
            ("docs/concepts/diffusion-core.html", "C1", "Diffusion Core", "D3PM-style uniform corruption, the cosine schedule, x0-parameterization, and the chunked-CE memory story."),
            ("docs/concepts/block-causal-attention.html", "C2", "Block-Causal Attention", "Causal across canvases, bidirectional within: the mask that makes one transformer a denoiser and an LM."),
            ("docs/concepts/self-conditioning.html", "C3", "Self-Conditioning", "Zero-init posterior feedback: bit-exact at init, detached pre-pass, cross-step conditioning at eval."),
        ],
        ("GUIDES", "Guides & Playbooks"): [
            ("docs/guides/quickstart.html", "G0", "Quickstart", "Data → train → sample → evaluate, end to end, with the shard-path contract spelled out."),
            ("docs/guides/debugging-playbook.html", "G1", "Debugging Playbook", "NaN rollback, resume divergence, anchor failures, shape errors — symptom-first recipes."),
        ],
        ("REFS", "API References"): [
            ("docs/references/config.html", "R1", "Config Reference", "Every key of configs/pretrain_a100_380m.yaml and its TrainingConfig default."),
            ("docs/references/api.html", "R2", "API Reference", "The public surface of models/, training/, data/, inference/, utils/ — symbol-anchored."),
        ],
    }

    portal_cards_html = ""
    for (cat_tag, cat_title), items in categories.items():
        cards = ""
        for href, tag, title, desc in items:
            cards += f"""
            <a href="{href}" class="portal-card">
                <span class="card-tag">{tag}</span>
                <div class="card-body">
                    <h3 class="card-heading">{title}</h3>
                    <p class="card-desc">{desc}</p>
                </div>
            </a>
            """
        portal_cards_html += f"""
        <section class="portal-section">
            <header class="portal-section-head">
                <span class="portal-section-mark">&sect; {cat_tag.lower()}</span>
                <h2 class="portal-section-title">{cat_title}</h2>
                <span class="portal-section-meta">{len(items)} entries</span>
            </header>
            <div class="portal-grid">{cards}</div>
        </section>
        """

    # Hero and pass-diagram animations live in assets/portal.js so the
    # index and every doc page share one script surface.
    index_scripts = '<script defer src="assets/portal.js"></script>'

    index_html = HEAD_TEMPLATE.format(
        title="DiffusionGemma-Lite Documentation Portal",
        extra_head="",
        rel_prefix="./",
        font_link=FONT_LINK,
        boot_script=BOOT_SCRIPT,
    ) + f"""<body class="index-portal">
    {BOOT_OVERLAY_HTML}
    <!-- Top Header -->
    <header class="site-header">
        <div class="header-left">
            <button class="mobile-toggle" onclick="toggleSidebar()" aria-label="Toggle Sidebar">☰</button>
            <a href="index.html" class="brand-logo">
                <span class="brand-name">DiffusionGemma-Lite</span>
                <span class="brand-badge">Documentation</span>
            </a>
        </div>
        <div class="header-right">
            <a href="README.html" class="header-link">GitHub README</a>
        </div>
    </header>

    <div class="app-layout">
        <!-- Left Sidebar Navigation -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-inner">
                {sidebar_html}
            </div>
        </aside>

        <!-- Main Portal Content -->
        <main class="main-content">
            <div class="content-container">
                <div class="hero-banner">
                    <div class="hero-margin-ticks" aria-hidden="true"></div>

                    <div class="hero-coords" aria-hidden="true">
                        <span class="coord">FIG &middot; A0</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">PARAM 343.5M</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">CANVAS 256</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">CTX 4096</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">VOCAB 50257</span>
                    </div>
                    <h1 class="hero-title">Diffusion<span class="hero-title-em">Gemma</span><span class="hero-title-em-accent">-Lite</span></h1>
                    <p class="hero-subtitle">From-scratch PyTorch block-AR discrete diffusion LM at ~343.5M params — 256-token canvases denoised in parallel from uniform noise, block-causal attention (causal across canvases, bidirectional within), entropy-bounded adaptive stopping, and a chunked-CE loss that never materializes full-vocab logits. Pure PyTorch. Read it like a field notebook: a name, a wiring sketch, then the measurements.</p>

                    <div class="mamba-telemetry-ribbon" aria-label="Key DiffusionGemma Architectural Metrics">
                        <div class="telemetry-card terra">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> PARALLEL DECODE</div>
                            <div class="tc-val">256 <span class="unit">TOKENS / STEP</span></div>
                            <div class="tc-desc">One forward pass refines a whole canvas; KV cache grows once per finalized canvas, not per token</div>
                        </div>
                        <div class="telemetry-card olive">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> ADAPTIVE STOPPING</div>
                            <div class="tc-val">~2&times; <span class="unit">FLOP CUT</span></div>
                            <div class="tc-desc">Entropy-bounded early stop &middot; closed rule, no learned halting policy &middot; fixed-schedule fallback is bit-identical</div>
                        </div>
                        <div class="telemetry-card gold">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> CE MEMORY</div>
                            <div class="tc-val">6.6&rarr;1.1 <span class="unit">GB @ ms8</span></div>
                            <div class="tc-desc">Chunked x0-CE over 8192-token vocab chunks &middot; never materializes the (B,T,V) logits tensor</div>
                        </div>
                        <div class="telemetry-card ink">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> ZERO-INIT SELF-COND</div>
                            <div class="tc-val">&Delta;=0 <span class="unit">AT INIT</span></div>
                            <div class="tc-desc">f(xt, t, sc) == f(xt, t) bit-for-bit at init (fp64, atol=0) &middot; conditioning is learned, not imposed</div>
                        </div>
                    </div>

                    <div class="pass-widget" role="region" aria-label="The block-AR decode pipeline">
                        <div class="pass-widget-header">
                            <div class="pass-badge">
                                <span class="pass-dot" aria-hidden="true"></span>
                                <span class="pass-tag">FIG. A0</span>
                                <span class="pass-sep" aria-hidden="true">/</span>
                                <span class="pass-title">BLOCK-AR DECODE PIPELINE</span>
                                <span class="pass-dim">STATIC &middot; SEE DIFFUSION.MD FOR THE FULL DERIVATION</span>
                            </div>
                        </div>
                        <div class="pass-widget-footer">
                            <div class="pass-status-ticker">
                                <span class="ticker-beacon">&bull;</span>
                                <span class="ticker-text">PREFILL PROMPT &rarr; [ DENOISE CANVAS (T steps, 256 tokens in parallel) &rarr; COMMIT unchanged-or-stronger &rarr; RE-ENCODE into KV ] &times; n_canvases</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="portal-content">
                    {portal_cards_html}
                </div>
            </div>
        </main>
    </div>

    <!-- Scripts -->
    {index_scripts}
</body>
</html>
"""

    out_file = OUTPUT_DIR / "index.html"
    out_file.write_text(index_html, encoding="utf-8")


def generate_assets():
    """Copy assets/style.css and assets/portal.js into the docs build."""
    assets_dir = OUTPUT_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    src_css = WORKSPACE_DIR / "assets" / "style.css"
    shutil.copyfile(src_css, assets_dir / "style.css")
    src_js = WORKSPACE_DIR / "assets" / "portal.js"
    if src_js.exists():
        shutil.copyfile(src_js, assets_dir / "portal.js")
    else:
        (assets_dir / "portal.js").write_text(
            "function toggleSidebar(){document.getElementById('sidebar').classList.toggle('open');}\n",
            encoding="utf-8")


def main():
    print("Building DiffusionGemma-Lite HTML Documentation...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / ".nojekyll").touch()

    generate_assets()

    for rel_path, category, display_title in DOC_FILES:
        print(f"Generating: {rel_path} -> docs_html/{rel_path.replace('.md', '.html')}")
        generate_html_page(rel_path, category, display_title)

    generate_index_portal()
    print("\nDocumentation build complete!")
    print(f"HTML Portal location: {OUTPUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
