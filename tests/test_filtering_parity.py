"""Lock the filtering contract this runtime now shares with galaxy.

The synchronisation is only worth what it is worth as long as it holds. These
tests state the parts a future edit on either side would have to break
deliberately rather than by accident: the marker vocabulary, which gates count as
"examined", the page budget, and the routing of tables away from the judge.

They do not import galaxy — the two repos deploy separately and a test that
needed both checked out would simply be skipped in CI. Instead they pin the
literal values, so a divergence shows up here and the reader is pointed at the
other runtime's constant by name.
"""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from agentic.tools.as_of import REDACTION_MARKER, UNEXAMINED_MARKER
from agentic.tools.as_of import cache as cache_module
from agentic.tools.as_of import screen as screen_module
from agentic.tools.as_of.cache import CachedVerdict
from agentic.tools.as_of.screen import (
    GATE_JUDGE,
    GATE_JUDGE_OVERFLOW,
    GATE_JUDGE_UNAVAILABLE,
    GATE_PROVIDER_DATE,
    GATE_STRUCTURED,
)
from agentic.tools.as_of.structured import prune_markdown_tables
from agentic.tools.web_search import scrape as scrape_module
from agentic.tools.web_search.body_shape import normalize_body

T_CUT = date(2026, 6, 1)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("AS_OF_VERDICT_CACHE_DIR", str(tmp_path / "verdicts"))
    monkeypatch.setattr(cache_module, "_CACHE", None)


def _stub_judge(late_needle: str | None = None):
    async def judge(units):
        out = {}
        for unit in units:
            late = bool(late_needle) and late_needle in unit.text
            out[unit.digest] = CachedVerdict("2026-08-19" if late else None, "stub")
        return out

    return judge


def _run(content, monkeypatch, *, judge=None, page_date=None):
    monkeypatch.setattr(screen_module, "judge_units", judge or _stub_judge())
    return asyncio.run(
        scrape_module.apply_page_as_of_gates(
            content=content,
            url="https://example.com/page",
            page_date=page_date,
            t_cut=T_CUT,
            judge_prose=True,
        )
    )


# --- the vocabulary itself ---------------------------------------------------


def test_the_markers_match_galaxy_byte_for_byte():
    """A trace from either runtime must read the same.

    These are galaxy's `structured.REDACTION_MARKER` and
    `screen.UNEXAMINED_MARKER`. Changing one without the other makes the two
    runtimes' artifacts non-comparable, which is the thing this synchronisation
    bought.
    """
    assert REDACTION_MARKER == "[as_of: post-T_cut record(s) removed]"
    assert UNEXAMINED_MARKER == "[as_of: passage removed without being examined]"


def test_neither_marker_contains_the_other():
    """Reports grep bodies for these strings.

    If one were a substring of the other, every unexamined deletion would also be
    counted as a boundary finding and the distinction would be invisible to every
    downstream reader.
    """
    assert REDACTION_MARKER != UNEXAMINED_MARKER
    assert REDACTION_MARKER not in UNEXAMINED_MARKER
    assert UNEXAMINED_MARKER not in REDACTION_MARKER


def test_only_gates_that_looked_at_the_text_claim_a_finding():
    """Mirrors galaxy's `document._EXAMINED_GATES`."""
    examined = frozenset({GATE_PROVIDER_DATE, GATE_STRUCTURED, GATE_JUDGE})
    assert scrape_module._EXAMINED_GATES == examined
    assert scrape_module._markers_for(frozenset({GATE_JUDGE})) == [REDACTION_MARKER]
    assert scrape_module._markers_for(frozenset({GATE_JUDGE_OVERFLOW})) == [
        UNEXAMINED_MARKER
    ]
    assert scrape_module._markers_for(frozenset({GATE_JUDGE_UNAVAILABLE})) == [
        UNEXAMINED_MARKER
    ]
    # A run of removals that mixes reasons states both; neither speaks for the
    # other.
    assert scrape_module._markers_for(
        frozenset({GATE_JUDGE, GATE_JUDGE_OVERFLOW})
    ) == [REDACTION_MARKER, UNEXAMINED_MARKER]


def test_the_page_budget_matches_galaxy():
    """galaxy's `_PAGE_BODY_JUDGE_BUDGET`, sized from its stamped-body
    distribution (p50=11, p90=35, p99=65, max=66 over 113 bodies)."""
    assert scrape_module._PAGE_BODY_JUDGE_BUDGET == 96


# --- the normaliser ----------------------------------------------------------


def test_plain_text_is_returned_byte_for_byte():
    """The Serper path. A body with no markup must not be rewritten at all —
    an unaffected page persists exactly what an unfiltered run would have."""
    text = "Brent crude settled at 111 dollars a barrel on August 19, 2026."
    assert normalize_body(text) == text


def test_html_and_pipe_tables_reach_the_canonical_form():
    """Two backends, two markups, one shape — the form Gate 2 deletes by row."""
    html = (
        "<div><p>Intro.</p><table>"
        "<tr><td>2026-05-02</td><td>98.4</td></tr>"
        "</table></div>"
    )
    pipe = "Intro.\n\n| 2026-05-02 | 98.4 |\n| 2026-04-01 | 95.0 |\n"
    for body in (normalize_body(html), normalize_body(pipe)):
        assert "<table>" in body
        assert "<tr><td>2026-05-02</td><td>98.4</td></tr>" in body


# --- the routing -------------------------------------------------------------


def test_a_late_row_costs_its_row_and_not_its_neighbours(monkeypatch):
    """Gate 2, by arithmetic, one record at a time.

    This is the measurement galaxy's refactor was built on: running the judge
    over a whole body removed 57-64% of the rows that had already passed this
    gate, because one late row cost every row in its chunk.
    """
    body = (
        "<div><p>Prices moved through the spring.</p><table>"
        "<tr><td>2026-05-02</td><td>98.4</td></tr>"
        "<tr><td>2026-08-19</td><td>111.2</td></tr>"
        "</table></div>"
    )
    content, counts = _run(body, monkeypatch)

    assert counts["records_seen"] == 2
    assert counts["pruned_records"] == 1
    assert "2026-05-02" in content
    assert "2026-08-19" not in content
    assert "Prices moved through the spring." in content


def test_a_blocked_prose_chunk_leaves_a_marker_not_a_hole(monkeypatch):
    """A removal the reader cannot see is worse than a labelled one.

    Measured on this runtime before the change: 23,590 chunks removed by Gate 3
    with nothing written in their place.
    """
    safe = "Refiners reported steady demand through the spring. " * 25
    late = "On August 19, 2026 Brent settled at 111.2 dollars a barrel. " * 25
    content, counts = _run(
        f"{safe}\n\n{late}", monkeypatch, judge=_stub_judge("August 19, 2026")
    )

    assert counts["blocked_judge"] >= 1
    assert REDACTION_MARKER in content
    assert "August 19, 2026" not in content
    assert "Refiners reported steady demand" in content


def test_unjudged_overflow_is_kept_on_this_runtime(monkeypatch):
    """This runtime keeps what the budget could not reach, and that is correct.

    galaxy prunes its overflow because there it would become a persisted file
    that `read`/`grep`/`bash` can reach, bypassing the context edge. AxisAgentic
    writes no such file — the raw scrape cache is in-memory and the model sees
    the extraction output, not the body — so pruning would import the fix for a
    problem this runtime does not have and pay for it in evidence that was only
    ever *suspected*, never examined.

    The residue is real and is stated rather than assumed away: a late passage
    with no date in it, past the budget, is not reached.
    """
    monkeypatch.setattr(scrape_module, "_PAGE_BODY_JUDGE_BUDGET", 1)
    paragraphs = [
        f"Paragraph {index} discusses refining margins at length. " * 25
        for index in range(4)
    ]
    content, counts = _run("\n\n".join(paragraphs), monkeypatch)

    assert counts["blocked_judge_overflow"] == 0
    assert counts["kept"] == counts["chunks"]
    assert UNEXAMINED_MARKER not in content
    assert REDACTION_MARKER not in content


def test_pruned_overflow_says_it_was_never_examined(monkeypatch):
    """When a caller does ask for fail-closed overflow, it says the true thing.

    The policy is a parameter rather than a constant precisely so the two
    runtimes can differ on it while agreeing on what a deletion *claims*. Under
    `prune`, the removed chunks were never shown to the judge, so the body must
    not tell the model a post-cutoff record was found there.
    """
    import asyncio

    from agentic.tools.as_of.units import CHANNEL_PAGE, EvidenceUnit, chunk_document_text

    monkeypatch.setattr(screen_module, "judge_units", _stub_judge())
    text = "\n\n".join(
        f"Paragraph {index} discusses refining margins at length. " * 25
        for index in range(4)
    )
    units = [
        EvidenceUnit(
            ordinal=index,
            channel=CHANNEL_PAGE,
            text=chunk.text,
            url="https://example.com/page",
        )
        for index, chunk in enumerate(chunk_document_text(text))
    ]
    outcome = asyncio.run(
        screen_module.screen_units(
            units, T_cut=T_CUT, judge_budget=1, overflow_policy="prune"
        )
    )

    assert outcome.counts["blocked_judge_overflow"] > 0
    markers = scrape_module._markers_for(
        frozenset(item.gate for item in outcome.blocked)
    )
    assert markers == [UNEXAMINED_MARKER]


def test_gate_3_no_longer_refuses_a_whole_page(monkeypatch):
    """Retired upstream: whole-page refusal was 77-81% of all refusals, and most
    of those bodies were fine. Gate 1 keeps its refusal; Gate 3 does not."""
    late = "On August 19, 2026 the index closed at a record high. " * 30
    content, _ = _run(late, monkeypatch, judge=_stub_judge("August 19, 2026"))

    assert content is not None
    assert REDACTION_MARKER in content


def test_a_page_declaring_a_late_date_is_still_refused_whole(monkeypatch):
    """Gate 1 is where refusal survives: nothing on such a page is salvageable."""
    content, counts = _run(
        "Anything at all.", monkeypatch, page_date="2026-08-19"
    )
    assert content is None
    assert counts["blocked_provider_date"] == 1


def test_the_counts_account_for_every_chunk(monkeypatch):
    """`chunks - kept` must equal the sum of the gate counters.

    The invariant that caught the budget-prune path having no counter of its own,
    in both runtimes.
    """
    monkeypatch.setattr(scrape_module, "_PAGE_BODY_JUDGE_BUDGET", 2)
    paragraphs = [
        f"Paragraph {index} on inventories and refining margins. " * 25
        for index in range(5)
    ]
    _, counts = _run(
        "\n\n".join(paragraphs), monkeypatch, judge=_stub_judge("Paragraph 0")
    )

    blocked = sum(
        counts.get(key, 0)
        for key in (
            "blocked_provider_date",
            "blocked_structured",
            "blocked_judge",
            "blocked_judge_unavailable",
            "blocked_judge_overflow",
        )
    )
    assert counts["chunks"] - counts["kept"] == blocked


def test_tables_never_reach_the_judge(monkeypatch):
    """The routing rule itself.

    A judge that blocked everything it saw must still leave the table intact,
    because it is never shown one.
    """

    async def block_everything(units):
        return {unit.digest: CachedVerdict("2026-08-19", "stub") for unit in units}

    body = (
        "<div><p>Some prose that will be blocked.</p><table>"
        "<tr><td>2026-05-02</td><td>98.4</td></tr>"
        "</table></div>"
    )
    content, _ = _run(body, monkeypatch, judge=block_everything)

    assert "<tr><td>2026-05-02</td><td>98.4</td></tr>" in content
    assert "Some prose that will be blocked." not in content


def test_prune_markdown_tables_reports_seen_as_well_as_removed():
    """A numerator with no denominator cannot be read."""
    body = (
        "<table>\n"
        "<tr><td>2026-05-02</td><td>98.4</td></tr>\n"
        "<tr><td>2026-08-19</td><td>111.2</td></tr>\n"
        "</table>"
    )
    pruned, removed, seen = prune_markdown_tables(body, T_cut=T_CUT)
    assert (removed, seen) == (1, 2)
    assert "2026-08-19" not in pruned
    assert "2026-05-02" in pruned
