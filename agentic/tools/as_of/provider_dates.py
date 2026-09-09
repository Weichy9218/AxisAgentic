# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Read a Serper result date, committing to a day only when one is named.

Ported from the Milkyway ``galaxy`` runtime (``galaxy/workbench/serper_client.py``)
so both systems label the same card the same way. Without it the raw string
reaches Gate 1, whose parser is ISO-first and does not know ``"Jun 17, 2026"``,
so every English-formatted card falls through to the model judge — the same
verdict at a much higher price. Measured on a partial run before this existed:
0.49 provider-date blocks per task here against 1.66 in the reference arm, with
9.7x the judge requests.

**Relative dates are not converted.** ``"1 week ago"`` spans 5-9 days and
``"1 month ago"`` spans 26-35, so resolving one against a clock invents precision
the source never had. Near a boundary that rounding manufactures a false
``known_before``, which is the one label admission-plus-annotation cannot defend
against. They are reported as *relative* and carry no date, which sends them to
the judge — the right place for something genuinely undated.

The adapter also needs no clock, which is why there is no ``now`` parameter to
pass and no way for a replay to label a card differently than the live run did.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Final, NamedTuple

__all__ = [
    "DATE_PRECISION_ABSENT",
    "DATE_PRECISION_DAY",
    "DATE_PRECISION_RELATIVE",
    "DATE_PRECISION_UNPARSEABLE",
    "SerperDate",
    "parse_serper_date",
]

#: Absolute forms Serper actually serves. Measured upstream on a live probe:
#: ``/search`` is roughly 50% absolute, 30% relative, 20% absent; ``/news`` is
#: almost entirely relative.
_MONTH_FORMATS: Final = ("%b %d, %Y", "%B %d, %Y")
_ISO_RE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")
_RELATIVE_RE: Final = re.compile(r"\d+\s+(second|minute|hour|day|week|month|year)s?\s+ago", re.IGNORECASE)

DATE_PRECISION_DAY: Final = "day"
DATE_PRECISION_RELATIVE: Final = "relative"
DATE_PRECISION_UNPARSEABLE: Final = "unparseable"
DATE_PRECISION_ABSENT: Final = "absent"


class SerperDate(NamedTuple):
    """A parsed provider date: a day when one was actually named, plus its shape."""

    iso: str | None
    precision: str


def parse_serper_date(raw: str | None) -> SerperDate:
    """Parse one Serper ``date`` field.

    Returns an ISO day only for a form that names one. Everything else returns
    ``None`` with the shape recorded, so a caller can tell "no date" apart from
    "a date this parser does not know" without either becoming a guess.
    """
    if raw is None:
        return SerperDate(None, DATE_PRECISION_ABSENT)
    text = str(raw).strip()
    if not text:
        return SerperDate(None, DATE_PRECISION_ABSENT)
    for fmt in _MONTH_FORMATS:
        try:
            return SerperDate(datetime.strptime(text, fmt).date().isoformat(), DATE_PRECISION_DAY)  # noqa: DTZ007
        except ValueError:
            pass
    if _ISO_RE.fullmatch(text):
        return SerperDate(text, DATE_PRECISION_DAY)
    if _RELATIVE_RE.search(text):
        return SerperDate(None, DATE_PRECISION_RELATIVE)
    return SerperDate(None, DATE_PRECISION_UNPARSEABLE)
