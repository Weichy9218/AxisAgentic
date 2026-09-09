# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Executable spec for the three as-of gates.

Several assertions here look like they test trivia. They do not: each one names a
failure that was measured upstream, and the comment says which. In particular the
``null`` / ``"unknown"`` split and the overflow asymmetry are the two places where
a plausible simplification silently converts the filter into either an
evidence-starver or an open-book run.
"""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from agentic.tools.as_of import (
    GATE_JUDGE,
    GATE_JUDGE_OVERFLOW,
    GATE_JUDGE_UNAVAILABLE,
    GATE_PROVIDER_DATE,
    GATE_STRUCTURED,
    REDACTION_MARKER,
    UNDATABLE,
    CachedVerdict,
    EvidenceUnit,
    JudgeUnavailable,
    VerdictCache,
    carries_post_cutoff_listing,
    chunk_document_text,
    is_after_cut,
    mentions_late_date,
    needs_judgement,
    prune_post_cutoff_records,
    screen_units,
    units_from_search_cards,
)
from agentic.tools.as_of import screen as screen_module
from agentic.tools.as_of.judge import _SYSTEM_PROMPT, _parse_verdicts
from agentic.tools.as_of.screen import _blocks

T_CUT = date(2026, 6, 29)


# --- the judge's isolation --------------------------------------------------


@pytest.mark.parametrize("forbidden", ["cutoff", "t_cut", "withheld", "blocked", "allowed"])
def test_the_judge_prompt_never_names_the_boundary(forbidden: str) -> None:
    """The judge is asked when something became knowable, not whether it leaks.

    Keeping ``T_cut`` out of the prompt is what makes a verdict a property of the
    passage, and therefore cacheable across tasks with different boundaries. It
    also keeps the comparison in code, where it can be audited and changed
    without re-spending a token.

    The list is deliberately not "every word that sounds temporal": the prompt
    has to say "knowable before that period ends" to state its dating rule.
    What must not appear is a *boundary this run enforces*.
    """
    assert forbidden not in _SYSTEM_PROMPT.lower()


def test_the_judge_prompt_states_no_concrete_date() -> None:
    """A date in the prompt would be a boundary leaking in through the back door."""
    import re

    dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", _SYSTEM_PROMPT)
    # The only dates present are inside the worked examples of the dating rules,
    # which are fixed illustrations rather than anything this run computed.
    assert set(dates) <= {"2026-06-30", "2023-12-31", "2026-01-04", "2026-04-15"}


def test_the_judge_contract_admits_exactly_three_answer_shapes() -> None:
    verdicts = _parse_verdicts(
        '[{"i":0,"knowable_from":"2026-04-15"},{"i":1,"knowable_from":null},{"i":2,"knowable_from":"unknown"}]',
        3,
    )
    assert verdicts == {0: "2026-04-15", 1: None, 2: UNDATABLE}


def test_a_misaligned_batch_is_rejected_rather_than_trusted() -> None:
    """Measured upstream on a 12-unit batch: verdicts came back shifted by one.

    That put a settled 2004 date on a methodology note and left the passage that
    owned it unanswered. A batch that does not come back one-for-one, in order,
    is a batch the judge did not answer.
    """
    with pytest.raises(JudgeUnavailable, match="misaligned"):
        _parse_verdicts('[{"i":0,"knowable_from":null},{"i":2,"knowable_from":null}]', 2)


def test_a_short_batch_is_rejected() -> None:
    with pytest.raises(JudgeUnavailable, match="1 verdicts for 2"):
        _parse_verdicts('[{"i":0,"knowable_from":null}]', 2)


@pytest.mark.parametrize("payload", ["not json", "[", '[{"i":0,"knowable_from":"soon"}]', '{"i":0}'])
def test_anything_off_contract_is_unavailable_not_a_pass(payload: str) -> None:
    with pytest.raises(JudgeUnavailable):
        _parse_verdicts(payload, 1)


def test_a_verdict_with_no_knowable_from_key_reads_as_null_and_therefore_passes() -> None:
    """Pins a real asymmetry in the upstream contract, kept for alignment.

    Everywhere else this design fails closed: an unreachable judge blocks, an
    unparseable date blocks, a misaligned batch is rejected. A verdict *object*
    that simply omits ``knowable_from`` does not — ``.get`` returns ``None``,
    which is the "settles nothing" answer, which passes.

    This is what the Milkyway implementation does, and this port matches it so
    the two systems screen identically. It is pinned here rather than fixed so
    the divergence from the fail-closed principle is visible: if a judge model
    ever starts dropping the key under load, evidence passes silently.
    """
    assert _parse_verdicts('[{"i":0}]', 1) == {0: None}


# --- the decision boundary --------------------------------------------------


def test_null_passes_because_a_passage_that_settles_nothing_waits_for_nothing() -> None:
    """Schedules, forecasts and navigation have no knowable-from date.

    An earlier prompt asked when a passage "was said"; the model answered with its
    own idea of today, which is on or after every benchmark cutoff, so every
    undated schedule blocked. ``null`` passing is the fix.
    """
    assert _blocks(None, T_CUT) is False
    assert _blocks("", T_CUT) is False


def test_unknown_blocks_because_an_undatable_settled_claim_cannot_be_shown_to_predate() -> None:
    """Collapsing this into ``null`` is what makes a judge leak undated outcomes."""
    assert _blocks(UNDATABLE, T_CUT) is True


def test_a_date_is_compared_in_code_and_the_comparison_is_exclusive() -> None:
    assert _blocks("2026-06-28", T_CUT) is False
    assert _blocks("2026-06-29", T_CUT) is True
    assert _blocks("2026-06-30", T_CUT) is True


def test_a_broken_contract_is_not_a_pass() -> None:
    assert _blocks("29 June 2026", T_CUT) is True


# --- Gate 1 -----------------------------------------------------------------


def test_gate_one_needs_proof_not_suspicion() -> None:
    """Interval semantics: half of June is not on one side of a 29 June cutoff."""
    assert is_after_cut(EvidenceUnit(0, "search", "x", stated_date="2026-06-30"), T_CUT) is True
    assert is_after_cut(EvidenceUnit(0, "search", "x", stated_date="2026-06-28"), T_CUT) is False
    assert is_after_cut(EvidenceUnit(0, "search", "x", stated_date="2026-06"), T_CUT) is False
    assert is_after_cut(EvidenceUnit(0, "search", "x", stated_date=None), T_CUT) is False


def test_a_clean_dated_search_card_skips_the_judge_entirely() -> None:
    """The single largest class in the corpus, and the whole point of the shortcut."""
    clean = EvidenceUnit(0, "search", "Bank holds rates steady", stated_date="2026-05-01")
    assert needs_judgement(clean, T_CUT) is False


def test_a_page_chunk_is_never_skipped_on_the_publishers_date_alone() -> None:
    """The 69%-of-live-pages case: a claimed date is first publication, the body is today's."""
    chunk = EvidenceUnit(0, "page", "Rates were held steady.", stated_date="2020-01-01")
    assert needs_judgement(chunk, T_CUT) is True


def test_a_card_naming_a_late_date_is_routed_even_when_its_own_date_is_clean() -> None:
    unit = EvidenceUnit(0, "search", "Preview of the 2026-07-15 meeting", stated_date="2026-05-01")
    assert needs_judgement(unit, T_CUT) is True


# --- Gate 2 -----------------------------------------------------------------


def test_a_rows_own_pre_cutoff_timestamp_settles_it_against_a_later_mention() -> None:
    """Separates a valid holding from a metadata row without reading either phrase."""
    payload = {
        "rows": [
            {"date": "2026-04-30", "instrument": "T-Note maturing 2031-02-15"},
            {"date": "2026-07-02", "value": "9.9"},
        ]
    }
    pruned, counts = prune_post_cutoff_records(payload, T_cut=T_CUT)
    kept = [row.get("instrument") or row.get("value") for row in pruned["rows"]]
    assert kept == ["T-Note maturing 2031-02-15"]
    assert counts == {"lines": 0, "structured": 1}


def test_cells_are_read_one_hop_so_a_late_field_deep_inside_does_not_condemn_the_node() -> None:
    """Unbounded recursion pruned 14,275 of 130,565 elements upstream, 6,536 provably valid."""
    payload = {"nodes": [{"name": "Acme", "meta": {"dateModified": "2026-08-01"}}]}
    pruned, counts = prune_post_cutoff_records(payload, T_cut=T_CUT)
    assert len(pruned["nodes"]) == 1
    assert counts is None


def test_a_columnar_series_drops_the_position_from_every_column_at_once() -> None:
    """Pruning columns independently kept the late numbers while dropping the late dates."""
    payload = {
        "period": ["2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01"],
        "value": [1.0, 2.0, 3.0, 4.0],
        "data_points": 4,
    }
    pruned, counts = prune_post_cutoff_records(payload, T_cut=T_CUT)
    assert pruned["period"] == ["2026-05-01", "2026-06-01"]
    assert pruned["value"] == [1.0, 2.0]
    assert pruned["data_points"] == 2
    assert counts["structured"] == 2


def test_an_untouched_payload_comes_back_byte_identical() -> None:
    """An unaffected page must be indistinguishable from an unfiltered run's output."""
    payload = {"text": "Nothing dated here.", "rows": [{"a": 1}]}
    pruned, counts = prune_post_cutoff_records(payload, T_cut=T_CUT)
    assert pruned == payload
    assert counts is None


def test_free_text_loses_the_late_line_and_keeps_the_rest() -> None:
    payload = {"text": "2026-05-11  3018.7\n2026-07-11  3100.2\n2026-06-01  3050.0"}
    pruned, counts = prune_post_cutoff_records(payload, T_cut=T_CUT)
    assert "3100.2" not in pruned["text"]
    assert "3018.7" in pruned["text"]
    assert REDACTION_MARKER in pruned["text"]
    assert counts["lines"] == 1


def test_a_long_flattened_listing_is_truncated_not_deleted() -> None:
    """A 139-character one-line summary lost all three closes upstream, including the two a forecast needed."""
    head = "closed 2026-05-26 at 101.50, prior close 2026-05-24 at 100.80, " + ("filler text " * 90)
    tail = "later 2026-07-30 at 145.00 and 2026-08-01 at 150.00 and 2026-08-02 at 151.00"
    payload = {"text": head + tail}
    pruned, counts = prune_post_cutoff_records(payload, T_cut=T_CUT)
    assert "101.50" in pruned["text"]
    assert "145.00" not in pruned["text"]
    assert REDACTION_MARKER in pruned["text"]
    assert counts["lines"] == 1


def test_a_snippet_listing_needs_three_late_dates_not_one() -> None:
    """One late date is a footer, a schedule or a horizon, and over-blocking those moved accuracy 0.933 -> 0.667."""
    assert carries_post_cutoff_listing("Meeting scheduled for 2026-07-15.", T_CUT) is False
    assert carries_post_cutoff_listing("Nightly 2026-07-01, 2026-07-02, 2026-07-03", T_CUT) is True


def test_mentions_late_date_sees_iso_timestamps() -> None:
    """A trailing word boundary would match none of these. That defect cost 61.8% of evidence."""
    assert mentions_late_date("published 2026-07-11T00:00:00Z", T_CUT) is True
    assert mentions_late_date("published 2026-05-11T00:00:00Z", T_CUT) is False


# --- units ------------------------------------------------------------------


def test_a_card_unit_carries_title_and_snippet_because_both_leak() -> None:
    """263 of one run's surviving late-date strings were in ``title``."""
    units = units_from_search_cards([{"title": "Fed cuts", "snippet": "on 2026-07-30", "date": "2026-07-30", "link": "u"}])
    assert len(units) == 1
    assert units[0].text == "Fed cuts\non 2026-07-30"
    assert units[0].stated_date == "2026-07-30"
    assert units[0].url == "u"


def test_the_digest_ignores_whitespace_and_case_but_not_content() -> None:
    a = EvidenceUnit(0, "search", "Hello   World")
    b = EvidenceUnit(7, "page", "hello world")
    c = EvidenceUnit(0, "search", "hello worlds")
    assert a.digest == b.digest
    assert a.digest != c.digest


def test_the_digest_carries_no_boundary_so_a_verdict_is_reusable_across_tasks() -> None:
    """A T_cut-keyed cache would hit only inside one task, since T_cut differs per question."""
    unit = EvidenceUnit(0, "search", "Apple reported Q2 revenue.")
    assert unit.digest == EvidenceUnit(3, "page", "apple reported q2 revenue.").digest


def test_chunking_folds_a_short_trailer_into_the_previous_chunk() -> None:
    text = ("paragraph body " * 90) + "\n\n" + ("second body " * 90) + "\n\nShort tail."
    units = chunk_document_text(text, url="u", stated_date="2026-01-01")
    assert len(units) >= 1
    assert "Short tail." in units[-1].text
    assert all(unit.channel == "page" for unit in units)


# --- the screen -------------------------------------------------------------


def _screen(units, *, verdicts=None, fail=False, cache=None, monkeypatch=None):
    async def fake_judge(misses):
        if fail:
            raise JudgeUnavailable("no endpoint in a test")
        return {unit.digest: CachedVerdict((verdicts or {}).get(unit.text), "test-model") for unit in misses}

    monkeypatch.setattr(screen_module, "judge_units", fake_judge)
    monkeypatch.setattr(screen_module, "get_verdict_cache", lambda: cache)
    return asyncio.run(screen_units(units, T_cut=T_CUT))


@pytest.fixture
def cache(tmp_path) -> VerdictCache:
    return VerdictCache(root=tmp_path / "verdicts")


def test_gate_one_blocks_before_any_model_call(cache, monkeypatch) -> None:
    units = [EvidenceUnit(0, "search", "late", stated_date="2026-07-01")]
    outcome = _screen(units, cache=cache, monkeypatch=monkeypatch)
    assert outcome.blocked[0].gate == GATE_PROVIDER_DATE
    assert outcome.counts["judge_requests"] == 0
    assert outcome.kept == ()


def test_a_snippet_listing_is_settled_by_arithmetic_not_by_the_judge(cache, monkeypatch) -> None:
    """The judge answers ``null`` for a snapshot index; arithmetic does not have to ask."""
    units = [EvidenceUnit(0, "search", "Nightly 2026-07-01, 2026-07-02, 2026-07-03", stated_date=None)]
    outcome = _screen(units, cache=cache, monkeypatch=monkeypatch)
    assert outcome.blocked[0].gate == GATE_STRUCTURED
    assert outcome.counts["judge_requests"] == 0


def test_the_judge_verdict_decides_the_rest(cache, monkeypatch) -> None:
    units = [
        EvidenceUnit(0, "page", "The result was announced.", stated_date=None),
        EvidenceUnit(1, "page", "The vote is scheduled.", stated_date=None),
    ]
    outcome = _screen(
        units,
        verdicts={"The result was announced.": "2026-07-05", "The vote is scheduled.": None},
        cache=cache,
        monkeypatch=monkeypatch,
    )
    assert [u.ordinal for u in outcome.kept] == [1]
    assert outcome.blocked[0].gate == GATE_JUDGE
    assert outcome.blocked[0].knowable_from == "2026-07-05"


def test_an_unreachable_judge_blocks_rather_than_passing(cache, monkeypatch) -> None:
    """A blip that silently converts a truncated arm into an open-book one is the worst outcome."""
    units = [EvidenceUnit(0, "page", "Something settled.", stated_date=None)]
    outcome = _screen(units, fail=True, cache=cache, monkeypatch=monkeypatch)
    assert outcome.kept == ()
    assert outcome.blocked[0].gate == GATE_JUDGE_UNAVAILABLE
    assert outcome.counts["judge_failed"] == 1


def test_overflow_is_asymmetric(cache, monkeypatch) -> None:
    """Suspicion overflow is kept; finding overflow is blocked.

    Routing on suspicion costs a slot in a batch that was going out anyway, so
    unexamined suspicion is not evidence of anything. A passage that *names* a
    post-cutoff date and was never cleared does not get the benefit of the doubt.
    """
    monkeypatch.setenv("AS_OF_JUDGE_BUDGET_PER_CALL", "1")
    monkeypatch.setenv("AS_OF_JUDGE_HARD_CAP_PER_CALL", "1")
    suspicion = [EvidenceUnit(i, "page", f"undated prose {i}", stated_date=None) for i in range(3)]
    outcome = _screen(suspicion, verdicts={}, cache=cache, monkeypatch=monkeypatch)
    assert outcome.counts["blocked_judge_overflow"] == 0
    assert len(outcome.kept) == 3

    findings = [EvidenceUnit(i, "page", f"settled on 2026-07-0{i + 1}", stated_date=None) for i in range(3)]
    outcome = _screen(findings, verdicts={}, cache=cache, monkeypatch=monkeypatch)
    assert outcome.counts["blocked_judge_overflow"] == 2
    assert all(item.gate == GATE_JUDGE_OVERFLOW for item in outcome.blocked[1:])


def test_a_cached_verdict_costs_no_request(cache, monkeypatch) -> None:
    unit = EvidenceUnit(0, "page", "The result was announced.", stated_date=None)
    cache.put(unit.digest, CachedVerdict("2026-01-01", "test-model"))
    outcome = _screen([unit], cache=cache, monkeypatch=monkeypatch)
    assert outcome.counts["judge_requests"] == 0
    assert outcome.counts["cache_hits"] == 1
    assert len(outcome.kept) == 1


def test_the_outcome_carries_the_boundary_it_enforced(cache, monkeypatch) -> None:
    """So a caller stamping a payload cannot stamp a boundary other than the enforced one."""
    outcome = _screen([EvidenceUnit(0, "search", "x", stated_date="2026-01-01")], cache=cache, monkeypatch=monkeypatch)
    assert outcome.T_cut == T_CUT


def test_an_empty_batch_still_reports_that_the_gate_ran(cache, monkeypatch) -> None:
    """"Yes, and it held nothing" is a different answer from silence."""
    outcome = _screen([], cache=cache, monkeypatch=monkeypatch)
    assert outcome.counts == {"units": 0}
