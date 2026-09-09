# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""The one choke point. Three gates, in order, over one batch of units.

Ported from the Milkyway ``galaxy`` runtime.

**Where this runs.** Immediately after the provider or fetcher returns, inside
the tool, *before* the result is formatted for the model and *before* anything is
written to a cache. That placement is the point of the design. Under an earlier
architecture upstream the guard was the last gate before context, which put it
downstream of both the artifact writer and the summariser — so the artifact on
disk and the message in context were two different documents produced from the
same bytes by different paths. Measured on a 1,966-task run: zero search packets
on disk carried a truncation marker while 158 task logs did, and 263 packets
carried a summary naming a post-cutoff date because the summary had been composed
over the untruncated cards. A filter downstream of the summariser cannot win.
This one is upstream of everything, so there is one document and no way for the
two to disagree.

**The three gates.**

1. :func:`~agentic.tools.as_of.rules.is_after_cut` — the publisher's own declared
   date proves the unit postdates the boundary. Exact, free, and mostly already
   done server-side by the provider's own date parameter.
2. :mod:`~agentic.tools.as_of.structured` — post-cutoff records inside tables and
   columnar series, settled by arithmetic. Applied two ways, because the right
   action differs by channel. On a **page** it runs on the payload before the
   body is chunked, and it *truncates*: a long page is mostly prose the boundary
   has no quarrel with, and refusing the document over one late row was measured
   to cost more evidence than it protects. On a **search card** it runs here and
   *blocks the card whole* — a short snippet that is a listing is the listing, so
   there is no safe half to keep.
3. :mod:`~agentic.tools.as_of.judge` — when did this passage's strongest claim
   become knowable. The only gate that can see a leak with no date in it.

Gate 2 runs before Gate 3 on the search channel for a reason beyond cost. A dated
listing is exactly the shape the judge misreads: an index of snapshot names looks
like navigation, and navigation is the prompt's ``null`` bucket. Measured
upstream, 5 of 272 cards that reached context were such listings and the judge
returned ``null`` for all five. Arithmetic settles them without asking.

**Blocking, not rewriting.** A blocked unit is dropped whole. Nothing is
truncated, patched or paraphrased, because a rewrite is an assertion about which
half of a passage was safe. Dropping is auditable; a rewrite is not.

**Fail closed.** With a ``T_cut`` in force, a unit routed to the judge that comes
back without a verdict is blocked. There is no second flag to consult and no
degraded mode: a cutoff exists or it does not.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Final

from agentic.tools.as_of.cache import CachedVerdict, get_verdict_cache
from agentic.tools.as_of.judge import UNDATABLE, JudgeUnavailable, judge_units
from agentic.tools.as_of.rules import is_after_cut, needs_judgement
from agentic.tools.as_of.structured import carries_post_cutoff_listing, mentions_late_date
from agentic.tools.as_of.units import CHANNEL_SEARCH, EvidenceUnit

__all__ = [
    "GATE_JUDGE",
    "GATE_JUDGE_OVERFLOW",
    "GATE_JUDGE_UNAVAILABLE",
    "GATE_PROVIDER_DATE",
    "GATE_STRUCTURED",
    "BlockedUnit",
    "ScreenOutcome",
    "screen_units",
]

GATE_PROVIDER_DATE: Final = "provider_date"
GATE_STRUCTURED: Final = "structured"
GATE_JUDGE: Final = "judge"
GATE_JUDGE_UNAVAILABLE: Final = "judge_unavailable"
GATE_JUDGE_OVERFLOW: Final = "judge_overflow"


def _int_env(name: str, default: int, *, floor: int = 1) -> int:
    try:
        return max(floor, int(os.environ.get(name, str(default))))
    except ValueError:
        return max(floor, default)


#: How many units may be sent to the judge on suspicion alone — routed because
#: the channel cannot vouch for them, not because anything in them is wrong. A
#: 40-chunk page therefore costs a bounded number of judged chunks (the leading
#: ones, where a page states its topline claim) rather than 40. Overflow here is
#: **kept**: suspicion that was never examined is not evidence of anything.
#:
#: The residue is real and is stated rather than assumed away: a late chunk of a
#: long page that leaks in prose, with no date anywhere in it, is not reached.
def _judge_budget_per_call() -> int:
    return _int_env("AS_OF_JUDGE_BUDGET_PER_CALL", 12)


#: Ceiling on units whose own text names a post-cutoff date. Those are a finding
#: rather than a suspicion, so they outrank the budget above — but they still
#: need a bound, or one very large series page could bill hundreds of judge
#: requests. Overflow here is **blocked**, the same fail-closed rule that governs
#: an unreachable judge: a passage that names a date past the boundary and was
#: never cleared does not get the benefit of the doubt.
def _judge_hard_cap_per_call() -> int:
    return max(_judge_budget_per_call(), _int_env("AS_OF_JUDGE_HARD_CAP_PER_CALL", 24))


@dataclass(frozen=True, slots=True)
class BlockedUnit:
    """Identity and reason for one dropped unit — never its content."""

    ordinal: int
    gate: str
    url: str | None = None
    knowable_from: str | None = None


@dataclass(frozen=True)
class ScreenOutcome:
    """What one screening pass decided, and the boundary it decided against.

    ``T_cut`` travels with the outcome so a caller stamping a payload cannot
    stamp a boundary other than the one actually enforced.
    """

    T_cut: date
    kept: tuple[EvidenceUnit, ...] = ()
    blocked: tuple[BlockedUnit, ...] = ()
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def blocked_ordinals(self) -> frozenset[int]:
        return frozenset(item.ordinal for item in self.blocked)


def _blocks(knowable_from: str | None, T_cut: date) -> bool:  # noqa: N803
    """Turn one verdict into a decision. This is where the boundary lives.

    Three verdicts, three outcomes. ``None`` — the passage settles nothing — is
    the only one that passes unconditionally; a schedule or a forecast states no
    fact anyone had to wait for. :data:`~agentic.tools.as_of.judge.UNDATABLE` —
    settled but undatable — blocks, because a claim that cannot be placed in time
    cannot be shown to predate the boundary. A date is compared.
    """
    if not knowable_from:
        return False
    if knowable_from == UNDATABLE:
        return True
    try:
        return date.fromisoformat(knowable_from) >= T_cut
    except ValueError:
        # The judge was asked for YYYY-MM-DD and the parser accepted it; an
        # unparseable value here means the contract broke, which is not a pass.
        return True


def _partition(units: Sequence[EvidenceUnit], T_cut: date) -> tuple[list, list, list, list]:  # noqa: N803
    """Gates 1 and 2, plus the routing split that feeds Gate 3."""
    kept: list[EvidenceUnit] = []
    blocked: list[BlockedUnit] = []
    must_judge: list[EvidenceUnit] = []
    may_judge: list[EvidenceUnit] = []

    for unit in units:
        if is_after_cut(unit, T_cut):
            blocked.append(BlockedUnit(ordinal=unit.ordinal, gate=GATE_PROVIDER_DATE, url=unit.url))
            continue
        if unit.channel == CHANNEL_SEARCH and carries_post_cutoff_listing(unit.text, T_cut):
            blocked.append(BlockedUnit(ordinal=unit.ordinal, gate=GATE_STRUCTURED, url=unit.url))
            continue
        if not needs_judgement(unit, T_cut):
            kept.append(unit)
            continue
        if mentions_late_date(unit.text, T_cut):
            must_judge.append(unit)
        else:
            may_judge.append(unit)
    return kept, blocked, must_judge, may_judge


async def screen_units(units: Sequence[EvidenceUnit], *, T_cut: date, judge_enabled: bool = True) -> ScreenOutcome:  # noqa: N803
    """Run the gates over one tool call's worth of evidence units.

    Args:
        units: The evidence, in the order it arrived.
        T_cut: The exclusive boundary to screen against.
        judge_enabled: Whether Gate 3 runs. When ``False``, Gates 1 and 2 still
            run and everything they do not settle is **kept** — the run is
            screened by arithmetic alone. This is not the same as a judge that
            failed: a failure blocks, because a boundary that silently stopped
            being enforced is worse than one that costs evidence. A deliberate
            two-gate run records ``judge_enabled: 0`` in its counts, so a reader
            can tell the two apart in the trace rather than having to infer it
            from the absence of judge requests.
    """
    if not units:
        return ScreenOutcome(T_cut=T_cut, counts={"units": 0})

    kept, blocked, must_judge, may_judge = _partition(units, T_cut)

    if not judge_enabled:
        kept.extend(must_judge)
        kept.extend(may_judge)
        kept.sort(key=lambda unit: unit.ordinal)
        blocked.sort(key=lambda item: item.ordinal)
        return ScreenOutcome(
            T_cut=T_cut,
            kept=tuple(kept),
            blocked=tuple(blocked),
            counts={
                "units": len(units),
                "judge_enabled": 0,
                "judged": 0,
                "judge_requests": 0,
                "blocked_provider_date": sum(1 for item in blocked if item.gate == GATE_PROVIDER_DATE),
                "blocked_structured": sum(1 for item in blocked if item.gate == GATE_STRUCTURED),
                "kept": len(kept),
            },
        )

    hard_cap = _judge_hard_cap_per_call()
    budget = max(0, _judge_budget_per_call() - len(must_judge))
    routed = must_judge[:hard_cap] + may_judge[:budget]
    kept.extend(may_judge[budget:])
    overflow = 0
    for unit in must_judge[hard_cap:]:
        overflow += 1
        blocked.append(BlockedUnit(ordinal=unit.ordinal, gate=GATE_JUDGE_OVERFLOW, url=unit.url))

    cache = get_verdict_cache()
    verdicts: dict[str, CachedVerdict] = {}
    misses: list[EvidenceUnit] = []
    seen: set[str] = set()
    cache_hits = 0
    for unit in routed:
        digest = unit.digest
        hit = cache.get(digest)
        if hit is not None:
            cache_hits += 1
            verdicts[digest] = hit
        elif digest not in seen:
            seen.add(digest)
            misses.append(unit)

    judge_failed = False
    if misses:
        try:
            fresh = await judge_units(misses)
        except JudgeUnavailable:
            judge_failed = True
        else:
            for digest, verdict in fresh.items():
                cache.put(digest, verdict)
            verdicts.update(fresh)

    judge_blocked = 0
    unavailable = 0
    for unit in routed:
        verdict = verdicts.get(unit.digest)
        if verdict is None:
            unavailable += 1
            blocked.append(BlockedUnit(ordinal=unit.ordinal, gate=GATE_JUDGE_UNAVAILABLE, url=unit.url))
            continue
        if _blocks(verdict.knowable_from, T_cut):
            judge_blocked += 1
            blocked.append(
                BlockedUnit(
                    ordinal=unit.ordinal,
                    gate=GATE_JUDGE,
                    url=unit.url,
                    knowable_from=verdict.knowable_from,
                )
            )
            continue
        kept.append(unit)

    kept.sort(key=lambda unit: unit.ordinal)
    blocked.sort(key=lambda item: item.ordinal)
    counts = {
        "units": len(units),
        "judged": len(routed),
        "judge_requests": len(misses),
        "cache_hits": cache_hits,
        "blocked_provider_date": sum(1 for item in blocked if item.gate == GATE_PROVIDER_DATE),
        "blocked_structured": sum(1 for item in blocked if item.gate == GATE_STRUCTURED),
        "blocked_judge": judge_blocked,
        "blocked_judge_unavailable": unavailable,
        "blocked_judge_overflow": overflow,
        "kept": len(kept),
    }
    if judge_failed:
        counts["judge_failed"] = 1
    return ScreenOutcome(T_cut=T_cut, kept=tuple(kept), blocked=tuple(blocked), counts=counts)
