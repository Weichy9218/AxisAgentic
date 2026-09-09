# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Gate 2: deterministic pruning of post-cutoff records in structured content.

Ported from the Milkyway ``galaxy`` runtime. This gate exists because judgement
does not scale to tables. A long economic series is thousands of rows of a date
column and a value column. Asking a model "was this knowable before ``T_cut``" of
each row would cost more than the run and answer worse than a comparison
operator, because for a dated record the answer *is* a comparison operator. Prose
goes to :mod:`agentic.tools.as_of.judge`; anything with its own date column is
settled here, exactly and for free.

**This is hygiene, not a boundary.** It removes explicit date leakage from
content the agent already has, which is worth doing and is not isolation. A real
boundary would be a frozen snapshot with ``captured_at <= T_cut``, which this
system does not have. Measured upstream: 69% of live pages whose provider-claimed
date is pre-cutoff contain post-cutoff dates in the body, because the claimed
date is first publication and the content is today's. No rule that reads dates
can close that, which is why Gate 3 asks a different question rather than a
stricter version of this one.

**One rule, so there is nothing to maintain.**

    A list element is dropped when one of its cells mentions a date on or after
    ``T_cut`` — unless another cell *is* a date before ``T_cut``, in which case
    that cell is the element's own timestamp and the late mention is a reference.

It names no keys, no containers, no tools, no timezones and no date formats: it
only compares dates to each other. A new provider cannot silently escape it.

Three properties make it precise rather than blunt, and all three are structural:

*Only list elements are dropped.* Rows live in lists. Mappings are descended into
but never emptied, so the document envelope survives untouched without being named.

*An element is judged by its own cells, never by what hangs beneath it.* One hop,
through all-scalar children only. Without this, one late field deep inside a
JSON-LD node condemned the whole node: unbounded recursion pruned 14,275 of
130,565 recorded elements (10.9%), 6,536 of them provably pre-cutoff.

*A self-declared pre-cutoff timestamp wins.* ``{"date": "2026-04-30",
"instrument": "T-Note maturing 2031-02-15"}`` says when it is from, and the 2031
is an instrument label. Comparing the two dates separates that from
``["Date Range", "2023-08-21 to 2026-08-19"]`` without recognising either phrase.

**One structural exception: column-oriented series.** When a series is stored
column-wise, a record is a *position* shared across sibling lists, and pruning
each list independently drops the late dates while keeping the late numbers.
Measured on a recorded artifact: the date column lost its 105 post-cutoff
entries, the value column kept all 5348, and ``value[-1]`` still returned a
post-cutoff observation, with the kept prefix still aligned so nothing warned
that the tail was future data.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from datetime import date
from typing import Any, Final

from agentic.temporal import parse_bare_source_interval, parse_source_interval

__all__ = [
    "REDACTION_MARKER",
    "carries_post_cutoff_listing",
    "mentions_late_date",
    "prune_post_cutoff_records",
]

REDACTION_MARKER: Final = "[as_of: post-T_cut record(s) removed]"

#: A date-shaped token. Only the shape is recognised here; interpreting it is
#: :mod:`agentic.temporal`'s job.
#:
#: No trailing ``\b``: an ISO timestamp puts ``T`` straight after the day, and
#: ``T`` is a word character, so a trailing boundary matches nothing in
#: ``2026-05-11T00:00:00Z``. That is the exact defect that made an earlier guard
#: blind to every ISO date, and it must not be reintroduced.
_DATE_TOKEN_RE = re.compile(r"(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?<!\d)\d{8}(?!\d)")


def mentions_late_date(value: Any, T_cut: date) -> bool:  # noqa: N803
    """True when ``value`` contains a date token proven on/after ``T_cut``.

    Exported because Gate 1's routing decision asks the same question of prose,
    and two implementations of "is there a late date in this string" is how five
    date parsers came to disagree upstream.
    """
    if isinstance(value, date):
        return value >= T_cut
    if isinstance(value, (int, float, bool)) or value is None:
        return False
    for token in _DATE_TOKEN_RE.findall(str(value)):
        span = parse_source_interval(token)
        if span is not None and span[0] >= T_cut:
            return True
    return False


def _is_date_before(value: Any, T_cut: date) -> bool:  # noqa: N803
    """True when ``value`` *is* a date (not merely names one) before ``T_cut``.

    Whole-value, so a row's own timestamp counts and a sentence mentioning a date
    does not. Punctuation is stripped by :func:`parse_bare_source_interval`,
    because table cells carry it.
    """
    if isinstance(value, date):
        return value < T_cut
    if isinstance(value, (int, float, bool)) or value is None:
        return False
    span = parse_bare_source_interval(str(value))
    return span is not None and span[1] < T_cut


def _cells_of(element: Any) -> Iterator[Any]:
    """The element's own cells: its scalars, plus scalars of all-scalar children.

    One hop, and only through a child that is entirely scalar — that is what a
    table row looks like. A structured child is a deeper level and not this
    element's own content.
    """
    if not isinstance(element, (Mapping, list, tuple)):
        yield element
        return
    items = element.values() if isinstance(element, Mapping) else element
    for item in items:
        if not isinstance(item, (Mapping, list, tuple)):
            yield item
        elif isinstance(item, (list, tuple)) and not any(isinstance(x, (Mapping, list, tuple)) for x in item):
            yield from item


def _row_is_post_cutoff(element: Any, T_cut: date) -> bool:  # noqa: N803
    """True when a list element is a row dated on/after ``T_cut``.

    A cell that *is* a pre-cutoff date is the element's own timestamp, and it
    settles the question: whatever later date another cell mentions is a
    reference, not this row's time.
    """
    mentions_late = False
    for cell in _cells_of(element):
        if _is_date_before(cell, T_cut):
            return False
        if mentions_late_date(cell, T_cut):
            mentions_late = True
    return mentions_late


#: A list of scalars long enough to be a column rather than a short tuple of
#: fields. The floor keeps two- and three-cell rows out of the columnar branch,
#: where they belong to the row predicate instead.
_MIN_COLUMN_LENGTH: Final = 4


def _is_scalar_column(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= _MIN_COLUMN_LENGTH and not any(isinstance(item, (Mapping, list, tuple)) for item in value)


def _prune_parallel_columns(value: Mapping[str, Any], T_cut: date) -> tuple[dict[str, Any], int] | None:  # noqa: N803
    """Drop post-cutoff *positions* from a column-oriented series, or decline.

    Returns ``None`` when the mapping is not this shape, so the ordinary
    row-oriented path stays in charge of everything it already handled.
    """
    columns = {key: item for key, item in value.items() if _is_scalar_column(item)}
    if len(columns) < 2:
        return None
    lengths = {len(item) for item in columns.values()}
    if len(lengths) != 1:
        return None

    length = lengths.pop()
    # A date column is one whose cells are dates. Only those vote: a numeric
    # value column mentions no date and must not be read as one.
    date_columns = [
        key
        for key, item in columns.items()
        if sum(1 for cell in item if _is_date_before(cell, T_cut) or mentions_late_date(cell, T_cut)) > length / 2
    ]
    if not date_columns:
        return None

    doomed = {position for position in range(length) for key in date_columns if mentions_late_date(columns[key][position], T_cut)}
    if not doomed:
        return None

    out = dict(value)
    for key, item in columns.items():
        out[key] = [cell for position, cell in enumerate(item) if position not in doomed]
    # These describe the series that is now shorter.
    for key in ("data_points", "row_count"):
        if isinstance(out.get(key), int):
            out[key] = length - len(doomed)
    return out, len(doomed)


def _prune_rows(value: Any, T_cut: date) -> tuple[Any, int]:  # noqa: N803
    """Drop post-cutoff rows anywhere in the payload; keep every container."""
    if isinstance(value, Mapping):
        columnar = _prune_parallel_columns(value, T_cut)
        if columnar is not None:
            pruned, removed = columnar
            for key, item in pruned.items():
                if not _is_scalar_column(item):
                    pruned[key], count = _prune_rows(item, T_cut)
                    removed += count
            return pruned, removed
        out: dict[Any, Any] = {}
        removed = 0
        for key, item in value.items():
            out[key], count = _prune_rows(item, T_cut)
            removed += count
        return out, removed
    if isinstance(value, (list, tuple)):
        kept: list[Any] = []
        removed = 0
        for item in value:
            if _row_is_post_cutoff(item, T_cut):
                removed += 1
                continue
            pruned, count = _prune_rows(item, T_cut)
            removed += count
            kept.append(pruned)
        return kept, removed
    return value, 0


#: A line long enough that deleting it whole is a meaningful loss. Measured over
#: 637 recorded page bodies upstream: 124 contain a line of 1000+ characters.
_LONG_LINE_CHARS: Final = 1000

#: Post-cutoff dates in one long line before it is read as a flattened listing
#: rather than prose. A long line with a single late date is prose, and prose is
#: Gate 3's business; truncating it mid-sentence would leave a misleading
#: fragment, which is the rewriting this design refuses to do.
_LONG_LINE_SERIES_DATE_THRESHOLD: Final = 3


def _truncate_long_listing_line(line: str, T_cut: date) -> str | None:  # noqa: N803
    """Cut a long, listing-shaped line at its first post-cutoff date, or decline.

    Whole-line deletion is right for a rendered table row and wrong for a long
    line that is mostly pre-cutoff content. Measured upstream: a 139-character
    single-line summary lost all three of its closes, including the two a
    forecast needed, because one was late.
    """
    if len(line) < _LONG_LINE_CHARS:
        return None
    positions = [
        match.start()
        for match in _DATE_TOKEN_RE.finditer(line)
        if (span := parse_source_interval(match.group(0))) is not None and span[0] >= T_cut
    ]
    if len(positions) < _LONG_LINE_SERIES_DATE_THRESHOLD:
        return None
    head = line[: positions[0]].rstrip()
    if not head:
        return None
    return f"{head} {REDACTION_MARKER}"


def _redact_lines(text: str, T_cut: date) -> tuple[str, int]:  # noqa: N803
    """Drop whole lines of free text that carry a post-cutoff date.

    Line-oriented because the target is a rendered table row. One exception, for
    the case where "the line" is not a record: a long line carrying several late
    dates is a flattened listing, and is truncated at the first of them instead
    of deleted, so its pre-cutoff prefix survives.
    """
    out: list[str] = []
    removed = 0
    pending_marker = False
    for line in text.splitlines():
        if mentions_late_date(line, T_cut):
            truncated = _truncate_long_listing_line(line, T_cut)
            if truncated is not None:
                if pending_marker:
                    out.append(REDACTION_MARKER)
                    pending_marker = False
                out.append(truncated)
                removed += 1
                continue
            removed += 1
            pending_marker = True
            continue
        if pending_marker:
            out.append(REDACTION_MARKER)
            pending_marker = False
        out.append(line)
    if pending_marker:
        out.append(REDACTION_MARKER)
    return "\n".join(out), removed


#: The free-text fields a fetched page exposes. Named rather than detected: the
#: alternative (any long multi-line string) was measured to miss 1 of 176
#: recorded page bodies while also catching field values, and unlike the row
#: predicate this list does not grow with new providers.
_CONTENT_KEYS: Final = ("content", "markdown", "text")


def carries_post_cutoff_listing(text: str, T_cut: date) -> bool:  # noqa: N803
    """True when this passage is a *listing* of post-cutoff dated entries.

    Density alone: at least :data:`_LONG_LINE_SERIES_DATE_THRESHOLD` distinct
    post-cutoff dates. The length companion used for page bodies is deliberately
    absent here — a search snippet is a few hundred characters by construction,
    so requiring a thousand would mean this never fires. What identifies a
    listing in a snippet is that three or more dated entries fit in it, which
    prose does not do.

    One late date is not enough and must not be: that is a page footer, a
    scheduled event or a stated horizon, the population Gate 3 exists to read,
    and the population whose over-blocking moved directional accuracy from 0.933
    to 0.667 upstream.
    """
    if not text:
        return False
    late = {
        match.group(0)
        for match in _DATE_TOKEN_RE.finditer(text)
        if (span := parse_source_interval(match.group(0))) is not None and span[0] >= T_cut
    }
    return len(late) >= _LONG_LINE_SERIES_DATE_THRESHOLD


def prune_post_cutoff_records(data: Mapping[str, Any], *, T_cut: date) -> tuple[dict[str, Any], dict[str, int] | None]:  # noqa: N803
    """Strip post-cutoff rows and text lines from a page or artifact payload.

    Returns ``(payload, None)`` when nothing was pruned, so an unaffected page is
    byte-identical to what an unfiltered run would produce.
    """
    guarded = dict(data)
    lines_removed = 0
    for key in _CONTENT_KEYS:
        value = guarded.get(key)
        if isinstance(value, str) and value:
            cleaned, removed = _redact_lines(value, T_cut)
            if removed:
                guarded[key] = cleaned
                lines_removed += removed
    guarded, structured_removed = _prune_rows(guarded, T_cut)
    if lines_removed or structured_removed:
        return guarded, {"lines": lines_removed, "structured": structured_removed}
    return dict(data), None
