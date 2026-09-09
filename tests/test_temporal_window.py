# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Executable spec for the temporal boundary layer.

Each test pins a decision that cost something to learn upstream; the comments
say which, so a future edit that "simplifies" one of them fails loudly.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from agentic.temporal import (
    DATE_STATUS_AFTER_CUT,
    DATE_STATUS_KNOWN_BEFORE,
    DATE_STATUS_UNKNOWN,
    TEMPORAL_POLICY_DISABLED_CONTROL,
    TEMPORAL_POLICY_STRICT,
    TemporalPolicyError,
    classify_date_status,
    global_exclusive_cutoff_utc,
    mapping_date_status,
    parse_bare_source_interval,
    parse_cutoff,
    parse_observation_time,
    parse_source_date,
    parse_source_interval,
    resolve_forecast_time,
)


# --- parsing ----------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "2026-05-07T00:00:00.000Z",
        "2026-05-07T00:00:00Z",
        "2026-05-07T12:30:00+08:00",
        "2026-05-07 12:30",
        "2026-05-07",
    ],
)
def test_iso8601_timestamps_parse_to_their_civil_day(value: str) -> None:
    """The regression that motivated this module.

    A date regex ending in ``\\b`` matches none of these: ``T`` is a word
    character, so the boundary after the day never fires. That single character
    discarded 61.8% of valid pre-cutoff evidence upstream.
    """
    assert parse_source_interval(value) == (date(2026, 5, 7), date(2026, 5, 7))


def test_partial_dates_are_intervals_not_points() -> None:
    """``2026-05`` is a month and ``2026`` is a year. Collapsing either would lie."""
    assert parse_source_interval("2026-05") == (date(2026, 5, 1), date(2026, 5, 31))
    assert parse_source_interval("2026") == (date(2026, 1, 1), date(2026, 12, 31))
    assert parse_source_interval("2026-02") == (date(2026, 2, 1), date(2026, 2, 28))


def test_a_date_inside_prose_parses_only_when_a_fallback_regex_covers_its_shape() -> None:
    """Pins a real asymmetry, verified identical to the upstream parser.

    The ISO path runs ``fromisoformat`` on the *whole* value, so a bare
    ``"2026-05-07"`` resolves to a day. The fallback regexes that search inside
    text cover the compact and slash shapes but not the dashed one, so a dashed
    ISO date embedded in prose degrades to year precision.

    That degradation is safe on the path this repository uses: year precision
    straddles most cutoffs, which yields ``unknown``, which is *admitted*. Gate 1
    reads a provider's own date field, where the value arrives bare; prose is
    Gate 3's job and Gate 3 asks a model rather than this parser. Widening the
    fallbacks would also start reading ``"T-Note maturing 2031-02-15"`` as a row
    timestamp, which is the failure ``parse_bare_source_interval`` exists to stop.
    """
    day = (date(2026, 5, 7), date(2026, 5, 7))
    year = (date(2026, 1, 1), date(2026, 12, 31))

    assert parse_source_interval("2026-05-07") == day
    assert parse_source_interval("updated 2026/5/7 by staff") == day
    assert parse_source_interval("as of 20260507 close") == day
    assert parse_source_interval("Published 2026-05-07 by the wire") == year


def test_parse_source_interval_returns_none_when_nothing_is_date_like() -> None:
    assert parse_source_interval("") is None
    assert parse_source_interval(None) is None
    assert parse_source_interval("no dates here") is None


def test_parse_source_date_refuses_a_span() -> None:
    """A month-only date is genuinely not a day; collapsing it here would hide that."""
    assert parse_source_date("2026-05-07") == date(2026, 5, 7)
    assert parse_source_date("2026-05") is None
    assert parse_source_date("2026") is None


def test_bare_interval_rejects_a_date_embedded_in_a_label() -> None:
    """``parse_source_interval`` searches; ``parse_bare_source_interval`` does not.

    ``"T-Note maturing 2031-02-15"`` is an instrument name, not a row timestamp,
    and pruning a table row on it removes legitimate data.
    """
    assert parse_bare_source_interval("T-Note maturing 2031-02-15") is None
    assert parse_bare_source_interval("2031-02-15") == (date(2031, 2, 15), date(2031, 2, 15))


def test_bare_interval_strips_delimiters_real_cells_carry() -> None:
    """FRED serves series rows as ``"2026-08-19:"``. Rejecting that left a channel unredacted."""
    assert parse_bare_source_interval("2026-08-19:") == (date(2026, 8, 19), date(2026, 8, 19))
    assert parse_bare_source_interval("(2026-08-19)") == (date(2026, 8, 19), date(2026, 8, 19))
    # Loosening the *match* instead would read this as the whole of 2026.
    assert parse_bare_source_interval("(2026-08-19)") != (date(2026, 1, 1), date(2026, 12, 31))


def test_bare_interval_ignores_non_text_scalars() -> None:
    assert parse_bare_source_interval(2026) is None
    assert parse_bare_source_interval(None) is None
    assert parse_bare_source_interval(True) is None


# --- classification ---------------------------------------------------------


def test_classification_is_exclusive_at_the_cutoff() -> None:
    """A source dated ``T_cut`` itself is after the cut, not before it."""
    cut = date(2026, 5, 10)
    assert classify_date_status("2026-05-09", cut) == DATE_STATUS_KNOWN_BEFORE
    assert classify_date_status("2026-05-10", cut) == DATE_STATUS_AFTER_CUT
    assert classify_date_status("2026-05-11", cut) == DATE_STATUS_AFTER_CUT


def test_a_span_straddling_the_cutoff_is_unknown_not_a_guess() -> None:
    """May 2026 contains days on both sides of 2026-05-10. Neither label is provable."""
    assert classify_date_status("2026-05", date(2026, 5, 10)) == DATE_STATUS_UNKNOWN
    assert classify_date_status("2026-04", date(2026, 5, 10)) == DATE_STATUS_KNOWN_BEFORE
    assert classify_date_status("2026-06", date(2026, 5, 10)) == DATE_STATUS_AFTER_CUT


def test_undated_evidence_is_unknown_and_therefore_admitted() -> None:
    """Dropping undated evidence costs a third of all cards and buys no safety."""
    assert classify_date_status(None, date(2026, 5, 10)) == DATE_STATUS_UNKNOWN
    assert classify_date_status("1 week ago", date(2026, 5, 10)) == DATE_STATUS_UNKNOWN


def test_no_cutoff_means_everything_is_unknown() -> None:
    assert classify_date_status("2026-05-11", None) == DATE_STATUS_UNKNOWN


def test_mapping_status_prefers_an_existing_label_and_is_shallow() -> None:
    cut = date(2026, 5, 10)
    assert mapping_date_status({"date_status": DATE_STATUS_AFTER_CUT, "date": "2020-01-01"}, cut) == DATE_STATUS_AFTER_CUT
    assert mapping_date_status({"published_date": "2026-05-11"}, cut) == DATE_STATUS_AFTER_CUT
    # A nested date is a data point, not a publication date. Recursing would
    # block a whole page for containing an ordinary time series.
    assert mapping_date_status({"rows": [{"date": "2026-05-11"}]}, cut) == DATE_STATUS_UNKNOWN


def test_mapping_status_takes_the_strongest_claim() -> None:
    cut = date(2026, 5, 10)
    assert mapping_date_status({"date": "2020-01-01", "source_date": "2026-05-11"}, cut) == DATE_STATUS_AFTER_CUT


# --- cutoff configuration ---------------------------------------------------


def test_cutoff_config_must_be_an_unambiguous_day() -> None:
    """A cutoff is configuration; a typo must not degrade to "no cutoff"."""
    assert parse_cutoff("2026-05-10") == date(2026, 5, 10)
    assert parse_cutoff("2026-05") is None
    assert parse_cutoff("") is None


def test_observation_time_has_no_clock_fallback() -> None:
    assert parse_observation_time("2026-09-01") == date(2026, 9, 1)
    with pytest.raises(TemporalPolicyError):
        parse_observation_time("")
    with pytest.raises(TemporalPolicyError):
        parse_observation_time("sometime in May")


# --- the boundary -----------------------------------------------------------


def test_derived_backtest_boundary() -> None:
    window = resolve_forecast_time(
        "2026-07-06",
        observation_time="2026-09-01",
        delta_days=7,
        start_time="2026-06-29",
    )
    assert window.T_cut == date(2026, 6, 29)
    assert window.forecast_time == date(2026, 6, 29)
    assert window.effective_delta_days == 7
    assert window.t_cut_source == "derived"
    assert window.policy == TEMPORAL_POLICY_STRICT


def test_forecast_time_is_clamped_forward_to_the_question_creation_day() -> None:
    """A forecast time before the question existed makes it unanswerable by construction.

    ΔT stays an upper bound, so ``effective_delta_days`` records the real horizon.
    """
    window = resolve_forecast_time(
        "2026-07-06",
        observation_time="2026-09-01",
        delta_days=30,
        start_time="2026-06-29",
    )
    assert window.forecast_time == date(2026, 6, 29)
    assert window.T_cut == date(2026, 6, 29)
    assert window.delta_days == 30
    assert window.effective_delta_days == 7
    assert window.t_cut_source == "derived_clamped_to_start"


def test_no_clamp_to_observation_time() -> None:
    """Clamping to today would assert "today is the information boundary"."""
    window = resolve_forecast_time(
        "2026-07-06",
        observation_time="2026-09-01",
        delta_days=7,
        start_time="2020-01-01",
    )
    assert window.T_cut == date(2026, 6, 29)


def test_live_task_needs_no_cut() -> None:
    """The event has not resolved, so no post-boundary information exists to leak."""
    window = resolve_forecast_time(
        "2027-01-01",
        observation_time="2026-09-01",
        delta_days=7,
    )
    assert window.T_cut is None
    assert window.T_cut_instant is None
    assert window.forecast_time == date(2026, 12, 25)
    assert window.t_cut_source == "live_no_cut_needed"


def test_override_wins() -> None:
    window = resolve_forecast_time(
        "2026-07-06",
        observation_time="2026-09-01",
        delta_days=7,
        override="2026-01-01",
    )
    assert window.T_cut == date(2026, 1, 1)
    assert window.t_cut_source == "override"


def test_disabled_control_is_declared_and_labelled() -> None:
    """An unfiltered arm must be declared. An absent field is not a policy."""
    window = resolve_forecast_time(
        "2026-07-06",
        observation_time="2026-09-01",
        delta_days=7,
        policy=TEMPORAL_POLICY_DISABLED_CONTROL,
    )
    assert window.T_cut is None
    assert window.t_cut_source == TEMPORAL_POLICY_DISABLED_CONTROL
    assert window.policy == TEMPORAL_POLICY_DISABLED_CONTROL
    # It still records the horizon it would have used, so a reader can tell this
    # apart from "no boundary was needed".
    assert window.forecast_time == date(2026, 6, 29)


def test_strict_without_a_derivable_forecast_time_refuses_to_run() -> None:
    with pytest.raises(TemporalPolicyError, match="disabled_control"):
        resolve_forecast_time("2026-07-06", observation_time="2026-09-01", delta_days=None)


def test_negative_and_non_integer_horizons_are_refused() -> None:
    with pytest.raises(TemporalPolicyError):
        resolve_forecast_time("2026-07-06", observation_time="2026-09-01", delta_days=-1)
    with pytest.raises(TemporalPolicyError):
        resolve_forecast_time("2026-07-06", observation_time="2026-09-01", delta_days="a week")


def test_unknown_policy_is_refused() -> None:
    with pytest.raises(TemporalPolicyError):
        resolve_forecast_time("2026-07-06", observation_time="2026-09-01", delta_days=7, policy="off")


def test_t_cut_date_is_not_recoverable_from_t_cut_instant() -> None:
    """The instant is UTC+14 midnight in UTC, so ``.date()`` reads a day early.

    Storing only the instant and calling ``.date()`` shifts every boundary back
    one day. They are two fields for this reason.
    """
    window = resolve_forecast_time("2026-05-17", observation_time="2026-09-01", delta_days=7)
    assert window.T_cut == date(2026, 5, 10)
    assert window.T_cut_instant == datetime(2026, 5, 9, 10, 0, tzinfo=timezone.utc)
    assert window.T_cut_instant.date() == date(2026, 5, 9)
    assert window.T_cut_instant.date() != window.T_cut


def test_manifest_fields_are_json_ready_and_complete() -> None:
    """Every input that produced the boundary travels with it, or the run is irreproducible."""
    window = resolve_forecast_time(
        "2026-07-06",
        observation_time="2026-09-01",
        delta_days=7,
        start_time="2026-06-29",
    )
    fields = window.as_manifest_fields()
    assert fields == {
        "observation_time": "2026-09-01",
        "forecast_time": "2026-06-29",
        "start_time": "2026-06-29",
        "end_time": "2026-07-06",
        "delta_days": 7,
        "effective_delta_days": 7,
        "T_cut": "2026-06-29",
        "T_cut_instant": "2026-06-28T10:00:00+00:00",
        "t_cut_source": "derived",
        "temporal_policy": "strict",
    }


def test_global_exclusive_cutoff_uses_the_earliest_civil_timezone() -> None:
    assert global_exclusive_cutoff_utc(date(2026, 5, 10)) == datetime(2026, 5, 9, 10, 0, tzinfo=timezone.utc)
    # An input that already names an instant is returned as one.
    aware = datetime(2026, 5, 10, 3, 0, tzinfo=timezone.utc)
    assert global_exclusive_cutoff_utc(aware) == aware
