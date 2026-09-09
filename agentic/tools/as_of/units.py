# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""The one currency every as-of gate operates on.

Ported from the Milkyway ``galaxy`` runtime (``galaxy/forecast/tools/as_of/``).
Before that module existed there were four filtering vocabularies — search cards,
page markdown, table rows, raw bodies — each with its own rule code and its own
idea of what "block" meant. Three of them read dates out of prose with slightly
different regexes, which is why the same fix had to be applied four times and
landed in three of them.

An :class:`EvidenceUnit` is the smallest span of retrieved text that can be kept
or dropped on its own. A search packet is one unit per card; a page body is one
unit per chunk; a table is not units at all — a long dated series is settled by
arithmetic on its date column, in :mod:`agentic.tools.as_of.structured`.

Two identities, deliberately separate:

* ``ordinal`` — where the unit sits in its parent, so a block can be spliced back
  at the right place without keeping the text around.
* ``digest`` — a hash of the normalized text, and *only* of the text. It carries
  no ``T_cut``, because the verdict a unit earns is "when did this become
  knowable", which is a property of the passage and not of the run asking. That
  is what makes one stored verdict reusable across tasks and runs; a
  ``T_cut``-keyed cache would hit only within a single task, since ``T_cut``
  differs per question.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from agentic.tools.as_of.provider_dates import parse_serper_date

__all__ = [
    "CHANNEL_PAGE",
    "CHANNEL_SEARCH",
    "EvidenceUnit",
    "chunk_document_text",
    "units_from_search_cards",
]

CHANNEL_SEARCH: Final = "search"
CHANNEL_PAGE: Final = "page"

_WHITESPACE_RE = re.compile(r"\s+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")

#: Target size of one page chunk. Large enough that a claim and the sentence that
#: dates it stay together; small enough that blocking one costs a paragraph
#: rather than an article.
_CHUNK_TARGET_CHARS: Final = 1100

#: Never emit a chunk shorter than this on its own — a stray heading is not a
#: claim, and paying a judge slot for it is waste.
_CHUNK_MIN_CHARS: Final = 120


def _normalize_for_digest(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    """One keep-or-drop span of retrieved text.

    ``stated_date`` is what the *publisher* claims, carried verbatim so Gate 1
    can compare it without re-deriving it from the text. It is not evidence
    about the content: measured upstream on live pages, 69% of those with a
    pre-cutoff claimed date carry post-cutoff dates in the body.
    """

    ordinal: int
    channel: str
    text: str
    stated_date: str | None = None
    url: str | None = None

    @property
    def digest(self) -> str:
        """Content address of this unit's text — the verdict-cache key."""
        payload = _normalize_for_digest(self.text).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]


def units_from_search_cards(cards: Sequence[Mapping[str, Any]]) -> list[EvidenceUnit]:
    """One unit per card, over the card's own model-visible text.

    Title and snippet are joined because they are read together and leak
    together: upstream, 263 of one run's surviving late-date strings were in
    ``title``, a field the previous snippet-only rule never touched.

    ``ordinal`` is the card's index in the input sequence, so a caller can splice
    the survivors back without holding the text.

    ``stated_date`` is normalized through :func:`parse_serper_date` first. The
    provider serves ``"Jun 17, 2026"`` and ``"1 week ago"`` alongside ISO, and
    Gate 1's parser is ISO-first: handing it the raw string means every
    English-formatted card looks undated and goes to the model judge instead of
    being settled by a comparison. A relative form still carries no date, which
    is deliberate — it spans 5-9 days, and resolving it against a clock near a
    boundary manufactures a false "known before".
    """
    units: list[EvidenceUnit] = []
    for ordinal, card in enumerate(cards):
        if not isinstance(card, Mapping):
            continue
        parts = [
            str(card.get("title") or "").strip(),
            str(card.get("snippet") or "").strip(),
        ]
        text = "\n".join(part for part in parts if part)
        if not text:
            continue
        parsed = parse_serper_date(card.get("date"))
        units.append(
            EvidenceUnit(
                ordinal=ordinal,
                channel=CHANNEL_SEARCH,
                text=text,
                stated_date=parsed.iso,
                # Serper names the result URL ``link``; keep both readings so a
                # card from either shape carries its source.
                url=str(card.get("url") or card.get("link") or "").strip() or None,
            )
        )
    return units


def _paragraphs(text: str) -> Iterator[str]:
    for block in _PARAGRAPH_RE.split(text):
        stripped = block.strip()
        if stripped:
            yield stripped


def chunk_document_text(text: str, *, url: str | None = None, stated_date: str | None = None) -> list[EvidenceUnit]:
    """Split a page body into judgeable units on paragraph boundaries.

    Paragraphs rather than a fixed window, so a chunk is a thing a reader would
    call a passage. Consecutive short paragraphs are packed together up to
    :data:`_CHUNK_TARGET_CHARS`; a single paragraph longer than that is emitted
    whole rather than cut, because cutting mid-passage is the rewriting this
    design refuses to do.
    """
    units: list[EvidenceUnit] = []
    buffer: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        if not buffer:
            return
        units.append(
            EvidenceUnit(
                ordinal=len(units),
                channel=CHANNEL_PAGE,
                text="\n\n".join(buffer),
                stated_date=stated_date,
                url=url,
            )
        )
        buffer = []
        size = 0

    for paragraph in _paragraphs(text):
        if size and size + len(paragraph) > _CHUNK_TARGET_CHARS:
            flush()
        buffer.append(paragraph)
        size += len(paragraph)
        if size >= _CHUNK_TARGET_CHARS:
            flush()

    if buffer and size < _CHUNK_MIN_CHARS and units:
        # Too small to stand alone: fold it into the previous chunk rather than
        # spending a judge slot on a trailing heading.
        last = units[-1]
        units[-1] = EvidenceUnit(
            ordinal=last.ordinal,
            channel=last.channel,
            text=last.text + "\n\n" + "\n\n".join(buffer),
            stated_date=last.stated_date,
            url=last.url,
        )
        buffer = []
    flush()
    return units
