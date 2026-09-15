"""One body shape for every scrape backend, so both runtimes filter the same thing.

`agentic.tools.web_search` has four backends and they already agree on the
envelope — `{success, content, error, total_chars, total_lines, truncated}` — but
not on what `content` *is*:

    serper  -> payload["text"], plain text, no markup at all
    jina    -> Jina markdown, tables as pipe rows
    http    -> `response.text`, raw HTML, no conversion whatsoever
    typed   -> JSON / CSV, sometimes JSON-stat

Four shapes reaching one screen means the gates see something different depending
on which backend answered, and a trace from one run is not comparable to a trace
from another. galaxy avoided this by having a single fetch path that writes one
markdown body; this module is the equivalent for a runtime that cannot give up
its backends (this host's egress blocks `r.jina.ai` and most direct fetches, which
is the entire reason the Serper backend exists).

The contract is galaxy's, so a body normalised here can be handed to the same
gate sequence and produce the same counts:

* a table becomes an HTML block, `<table>` and `</table>` each alone on a line,
  one `<tr>` per line, because that is the form `galaxy.workbench.md_blocks`
  reads and the form that lets Gate 2 delete a record by deleting a line;
* everything else is prose, and prose is left byte-for-byte alone.

**What this does not do is invent structure.** Serper returns text that had its
markup stripped before it arrived; no amount of normalising puts the table back.
Such a body normalises to itself and Gate 2 finds nothing to prune by row, which
is the honest outcome — the alternative would be guessing which runs of digits
used to be a row. What the shared shape does buy on every backend is that Gate 3
sees identical chunk boundaries, writes identical markers, and reports identical
counts, which is what makes the two runtimes' traces comparable.
"""

from __future__ import annotations

import re
from html import escape, unescape
from typing import Final

__all__ = [
    "KIND_PROSE",
    "KIND_TABLE",
    "normalize_body",
    "looks_like_html",
]

KIND_PROSE: Final = "prose"
KIND_TABLE: Final = "table"

#: A body is treated as HTML when it carries a structural tag, not merely a
#: stray angle bracket: prose about "a < b" is prose, and an article that mentions
#: `<div>` in a code sample should not be parsed as a document.
_HTML_SIGNAL_RE: Final = re.compile(
    r"<\s*(?:html|body|div|table|section|article|main|p)\b", re.IGNORECASE
)

_TABLE_BLOCK_RE: Final = re.compile(r"<table\b[^>]*>([\s\S]*?)</table>", re.IGNORECASE)
_ROW_RE: Final = re.compile(r"<tr\b[^>]*>([\s\S]*?)</tr>", re.IGNORECASE)
_CELL_RE: Final = re.compile(r"<t([dh])\b[^>]*>([\s\S]*?)</t\1>", re.IGNORECASE)
_TAG_RE: Final = re.compile(r"<[^>]+>")

#: A markdown pipe table row: `| a | b |`. Requires both edges so a sentence
#: containing a pipe is not mistaken for tabular data.
_PIPE_ROW_RE: Final = re.compile(r"^\s*\|(.+)\|\s*$")
#: The `|---|:--:|` separator under a pipe table's header.
_PIPE_RULE_RE: Final = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")


def looks_like_html(content: str) -> bool:
    """True when this body should be parsed as HTML rather than read as text."""
    return bool(_HTML_SIGNAL_RE.search(content))


def _clean_cell(raw: str) -> str:
    """One cell's visible text.

    Tags are stripped *before* the text reaches any date rule, deliberately: a
    cell keeps its `<a href>` in the source, and `/news/2026/08/19/` inside that
    href is a date token to every rule that reads the raw line. Gate 2 must
    compare the date a reader sees, not the one in the link.
    """
    return " ".join(unescape(_TAG_RE.sub(" ", raw)).split())


def _emit_table(rows: list[list[str]]) -> str:
    """Rows as the one table form the gates read: one `<tr>` per line."""
    out = ["<table>"]
    for cells in rows:
        joined = "".join(f"<td>{escape(cell)}</td>" for cell in cells)
        out.append(f"<tr>{joined}</tr>")
    out.append("</table>")
    return "\n".join(out)


def _rows_from_html_table(inner: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_match in _ROW_RE.finditer(inner):
        cells = [_clean_cell(body) for _, body in _CELL_RE.findall(row_match.group(1))]
        if any(cell for cell in cells):
            rows.append(cells)
    return rows


def _normalize_html(content: str) -> str:
    """HTML in, prose + canonical table blocks out.

    Tables are re-emitted in the canonical form; everything outside them has its
    tags stripped and becomes prose. This is deliberately not a general HTML to
    markdown conversion — headings, lists and emphasis carry no information any
    gate acts on, and inventing a second converter that drifts from galaxy's is
    worse than plain text for the parts that are not tables.
    """
    parts: list[str] = []
    cursor = 0
    for match in _TABLE_BLOCK_RE.finditer(content):
        before = content[cursor : match.start()]
        if before.strip():
            parts.append(_strip_to_prose(before))
        rows = _rows_from_html_table(match.group(1))
        if rows:
            parts.append(_emit_table(rows))
        cursor = match.end()
    tail = content[cursor:]
    if tail.strip():
        parts.append(_strip_to_prose(tail))
    return "\n\n".join(part for part in parts if part.strip())


def _strip_to_prose(fragment: str) -> str:
    """Visible text of an HTML fragment, paragraph structure preserved."""
    without_invisible = re.sub(
        r"<(script|style)\b[\s\S]*?</\1>", " ", fragment, flags=re.IGNORECASE
    )
    with_breaks = re.sub(
        r"</(p|div|section|article|li|h[1-6]|tr)\s*>", "\n\n", without_invisible,
        flags=re.IGNORECASE,
    )
    with_breaks = re.sub(r"<br\s*/?>", "\n", with_breaks, flags=re.IGNORECASE)
    text = unescape(_TAG_RE.sub(" ", with_breaks))
    paragraphs = [" ".join(block.split()) for block in text.split("\n\n")]
    return "\n\n".join(block for block in paragraphs if block)


def _normalize_pipe_tables(content: str) -> str:
    """Jina markdown in, the same body with its pipe tables re-emitted as HTML.

    A run of two or more pipe rows is a table; a single one is a sentence that
    happens to contain pipes, and rewriting it would be the kind of guess this
    module exists to avoid.
    """
    lines = content.split("\n")
    out: list[str] = []
    buffer: list[list[str]] = []

    def flush() -> None:
        nonlocal buffer
        if len(buffer) >= 2:
            out.append(_emit_table(buffer))
        elif buffer:
            out.append(" | ".join(buffer[0]))
        buffer = []

    for line in lines:
        if _PIPE_RULE_RE.match(line):
            # The header rule carries no data; drop it rather than emit an
            # all-dashes row that a date rule would have to skip.
            continue
        match = _PIPE_ROW_RE.match(line)
        if match:
            cells = [cell.strip() for cell in match.group(1).split("|")]
            buffer.append(cells)
            continue
        flush()
        out.append(line)
    flush()
    return "\n".join(out)


def normalize_body(content: str, *, backend: str | None = None) -> str:
    """One scraped body in whatever shape its backend produced, one shape out.

    ``backend`` is advisory only. The shape is detected from the bytes, because
    the backends are not consistent about what they return — the HTTP backend
    hands back `response.text` for anything, including a URL that happens to
    serve markdown — and a body that lies about its backend would otherwise be
    parsed by the wrong reader.
    """
    if not content or not content.strip():
        return content
    if looks_like_html(content):
        return _normalize_html(content)
    if _PIPE_ROW_RE.search(content) or any(
        _PIPE_ROW_RE.match(line) for line in content.split("\n")
    ):
        return _normalize_pipe_tables(content)
    # Plain text (the Serper path) and typed data already are their own shape.
    # Returning them byte-for-byte is what keeps an unaffected page identical to
    # what an unfiltered run would have persisted.
    return content
