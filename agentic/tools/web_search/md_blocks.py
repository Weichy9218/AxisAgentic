"""Split a persisted markdown body into prose and table blocks.

Gate 2 (arithmetic over dated records) and Gate 3 (a model reading passages) want
different things, and until now both ran over the whole body. That is why one
late date in a table cost a 1,100-character chunk: prose-granularity deletion
applied to tabular content. Measured across two 200-task runs, on pages Gate 3
touched — 71% of pages carrying tables — 57-64% of the table rows that had
already passed Gate 2's arithmetic were gone from the persisted markdown.

The fix is not a smarter judge. It is giving each gate only the content it can
speak for. This module is that split: one pass over a body, out come blocks that
are either prose (Gate 3, by chunk) or a table (Gate 2, by row).

It reads exactly one table form — the one
:mod:`agentic.tools.web_search.body_shape` writes:

    <table>
    <caption>...</caption>          (optional)
    <tr><td>...</td><td>...</td></tr>
    </table>

with one ``<tr>`` per line. Because that form is generated rather than found,
parsing it needs no markdown table grammar, no escaped-pipe handling, and no
third-party parser: a row is a line, and a cell is a tag on it. Markdown from a
source that never went through that writer (a provider that returns markdown
directly) simply contains no such block and is all prose, which is correct — a
pipe table in it was never trustworthy enough to filter by row.

Splitting is lossless: ``join_blocks(split_blocks(text)) == text``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape

__all__ = [
    "KIND_PROSE",
    "KIND_TABLE",
    "Block",
    "drop_rows",
    "join_blocks",
    "split_blocks",
    "table_rows",
]

KIND_PROSE = "prose"
KIND_TABLE = "table"

_OPEN = "<table>"
_CLOSE = "</table>"
_CELL_RE = re.compile(r"<t[dh]\b[^>]*>([\s\S]*?)</t[dh]>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class Block:
    """One contiguous run of lines, and which gate speaks for it."""

    kind: str
    text: str

    @property
    def lines(self) -> list[str]:
        return self.text.split("\n")


def split_blocks(markdown: str) -> list[Block]:
    """Partition a body into prose and table blocks, preserving every line.

    An unterminated ``<table>`` stays prose. That is the fail-safe direction: a
    half-read table handed to Gate 2 would have its rows deleted by a rule that
    could not see all of them, while as prose it is judged whole.
    """
    if not markdown:
        return []
    lines = markdown.split("\n")
    blocks: list[Block] = []
    buffer: list[str] = []
    index = 0

    def flush_prose() -> None:
        nonlocal buffer
        if buffer:
            blocks.append(Block(KIND_PROSE, "\n".join(buffer)))
            buffer = []

    while index < len(lines):
        if lines[index].strip() != _OPEN:
            buffer.append(lines[index])
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end].strip() != _CLOSE:
            end += 1
        if end == len(lines):
            buffer.extend(lines[index:])
            break
        flush_prose()
        blocks.append(Block(KIND_TABLE, "\n".join(lines[index : end + 1])))
        index = end + 1

    flush_prose()
    return blocks


def join_blocks(blocks: list[Block]) -> str:
    """The inverse of :func:`split_blocks`."""
    return "\n".join(block.text for block in blocks)


def table_rows(block: Block) -> list[tuple[int, list[str]]]:
    """``(line index within the block, cell texts)`` for every row of a table.

    The line index is the handle: :func:`drop_rows` takes it back, so a caller
    that decides a row is post-cutoff never has to rebuild the table.

    Cells come back as *visible text only*. A cell keeps its ``<a href>`` on
    disk, and stripping it here is what keeps a URL out of the date comparison —
    ``/news/2026/08/19/`` is a date token to any rule that reads the raw line.
    """
    if block.kind != KIND_TABLE:
        return []
    rows: list[tuple[int, list[str]]] = []
    for index, line in enumerate(block.lines):
        stripped = line.strip()
        if not stripped.startswith("<tr"):
            continue
        cells = [
            " ".join(unescape(_TAG_RE.sub(" ", cell)).split())
            for cell in _CELL_RE.findall(stripped)
        ]
        rows.append((index, cells))
    return rows


def drop_rows(block: Block, line_indexes: set[int]) -> Block | None:
    """The table without those rows, or ``None`` when no row is left.

    ``None`` rather than an empty ``<table>`` shell: a table with a header and no
    records is a page that looks fetched and holds nothing, and the caller has a
    marker to put there instead.
    """
    if block.kind != KIND_TABLE or not line_indexes:
        return block
    kept = [line for index, line in enumerate(block.lines) if index not in line_indexes]
    if not any(line.strip().startswith("<tr") for line in kept):
        return None
    return Block(KIND_TABLE, "\n".join(kept))
