# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Gate 1 (provider bound) and the routing decision that feeds Gate 3.

Ported from the Milkyway ``galaxy`` runtime. Two jobs, and the second one is not
filtering.

**Gate 1 — the provider bound.** Drop a unit whose *declared publication date* is
proven on or after ``T_cut``. This is the client-side half of a bound the search
provider already applied server-side (Serper ``cd_max``), and it is cheap and
exact. It catches almost nothing by itself — measured upstream across a
1,966-task run, 92 of 53,728 cards (0.17%) — because the server-side half already
removed the rest. It stays because those 92 are the signal that tells you the
server-side half is still armed; the run where that number silently goes to zero
is the run where the gate has broken.

**Routing.** Everything Gate 1 keeps and Gate 2 does not settle gets a cheap test
for whether Gate 3 should look at it. This is the inversion that makes the design
work: under a filter-only stack a heuristic false positive *destroys evidence*,
so the heuristics have to be tuned narrow — widening them was separately measured
to move directional accuracy 0.933 -> 0.667. Here a routing false positive costs
one slot in a batch that was going to be sent anyway. So route generously.

Request count does not depend on how much is routed: Gate 3 sends one batched
request per tool call whatever happens. Only tokens do.
"""

from __future__ import annotations

from datetime import date

from agentic.temporal import parse_source_interval
from agentic.tools.as_of.structured import mentions_late_date
from agentic.tools.as_of.units import CHANNEL_SEARCH, EvidenceUnit

__all__ = ["is_after_cut", "needs_judgement"]


def is_after_cut(unit: EvidenceUnit, T_cut: date) -> bool:  # noqa: N803
    """Gate 1: the publisher's own declared date proves this unit is too new.

    Interval semantics, matching every other date decision in the system: a
    conclusion is only drawn when the *whole* parsed interval falls on one side.
    ``"2026-05"`` under a ``T_cut`` of ``2026-05-15`` is not proven late, because
    half of May is not.
    """
    if not unit.stated_date:
        return False
    span = parse_source_interval(unit.stated_date)
    return span is not None and span[0] >= T_cut


def needs_judgement(unit: EvidenceUnit, T_cut: date) -> bool:  # noqa: N803
    """Should Gate 3 look at this unit?

    A search card is skipped only when both cheap signals are clean: the
    publisher dated it before the cutoff, **and** its own text names no date on
    or after the cutoff. That conjunction is the whole shortcut — a short snippet
    from a dated pre-cutoff source that mentions no later date has nothing for a
    judge to find, and it is the single largest class in the corpus.

    Page chunks are never skipped on the publisher's date alone. That is the
    69%-of-live-pages case: a claimed date is when the page first appeared, and
    the body is whatever it serves today. For a short snippet that gap is narrow;
    for a page body it is the main risk, and it is invisible to every date rule
    by construction — a page published after ``T_cut`` that states the outcome in
    prose and carries no date at all matches nothing here.
    """
    if mentions_late_date(unit.text, T_cut):
        return True
    if unit.channel != CHANNEL_SEARCH:
        return True
    if not unit.stated_date:
        return True
    span = parse_source_interval(unit.stated_date)
    return span is None or span[1] >= T_cut
