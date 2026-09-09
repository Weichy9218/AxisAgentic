# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Executable spec for provider date normalization.

This exists because Gate 1's parser is ISO-first, so an unnormalized provider
string makes an English-dated card look undated and sends it to the model judge.
Measured on a partial run before this module: 0.49 provider-date blocks per task
against the reference arm's 1.66, with 9.7x the judge requests for the same
boundary.
"""

from __future__ import annotations

from datetime import date

import pytest

from agentic.tools.as_of import units_from_search_cards
from agentic.tools.as_of.provider_dates import (
    DATE_PRECISION_ABSENT,
    DATE_PRECISION_DAY,
    DATE_PRECISION_RELATIVE,
    DATE_PRECISION_UNPARSEABLE,
    parse_serper_date,
)
from agentic.tools.as_of.rules import is_after_cut


@pytest.mark.parametrize(
    ("raw", "iso"),
    [
        ("Jun 17, 2026", "2026-06-17"),
        ("June 17, 2026", "2026-06-17"),
        ("Dec 1, 2025", "2025-12-01"),
        ("2026-07-30", "2026-07-30"),
    ],
)
def test_absolute_forms_name_a_day(raw: str, iso: str) -> None:
    parsed = parse_serper_date(raw)
    assert parsed == (iso, DATE_PRECISION_DAY)


@pytest.mark.parametrize("raw", ["1 week ago", "3 days ago", "2 months ago", "10 hours ago"])
def test_relative_forms_carry_no_date_and_are_not_resolved(raw: str) -> None:
    """Resolving these against a clock invents precision the source never had.

    "1 week ago" spans 5-9 days. Near a boundary that rounding manufactures a
    false "known before", the one label that admitting-and-annotating cannot
    defend against. They stay undated, which routes them to the judge — the right
    place for something genuinely undated.
    """
    parsed = parse_serper_date(raw)
    assert parsed.iso is None
    assert parsed.precision == DATE_PRECISION_RELATIVE


def test_absent_and_unparseable_are_distinguished() -> None:
    """"No date" and "a date this parser does not know" are different facts."""
    assert parse_serper_date(None).precision == DATE_PRECISION_ABSENT
    assert parse_serper_date("").precision == DATE_PRECISION_ABSENT
    assert parse_serper_date("   ").precision == DATE_PRECISION_ABSENT
    assert parse_serper_date("2026年6月17日").precision == DATE_PRECISION_UNPARSEABLE
    assert parse_serper_date("sometime last spring").precision == DATE_PRECISION_UNPARSEABLE


def test_the_parser_reads_no_clock() -> None:
    """No ``now`` parameter, so a replay cannot label a card differently than the live run."""
    import inspect

    assert set(inspect.signature(parse_serper_date).parameters) == {"raw"}


def test_an_english_dated_card_now_reaches_gate_one() -> None:
    """The whole point: this card used to look undated and cost a judge slot."""
    cards = [{"title": "Fed cuts", "snippet": "by 25bp", "date": "Jul 30, 2026", "link": "u"}]
    unit = units_from_search_cards(cards)[0]
    assert unit.stated_date == "2026-07-30"
    assert is_after_cut(unit, date(2026, 6, 29)) is True


def test_a_relative_card_still_has_nothing_for_gate_one_to_compare() -> None:
    cards = [{"title": "Fed cuts", "snippet": "by 25bp", "date": "1 week ago", "link": "u"}]
    unit = units_from_search_cards(cards)[0]
    assert unit.stated_date is None
    assert is_after_cut(unit, date(2026, 6, 29)) is False


def test_an_undated_card_is_unchanged() -> None:
    cards = [{"title": "Reference page", "snippet": "no date here", "link": "u"}]
    unit = units_from_search_cards(cards)[0]
    assert unit.stated_date is None
