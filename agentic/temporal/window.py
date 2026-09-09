# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for source-date parsing and ``T_cut`` classification.

Every provider adapter, as-of gate, prompt, and evaluation path that needs to
know *when* a piece of evidence was published goes through this module. Ported
from the Milkyway ``galaxy`` runtime (``galaxy/workbench/temporal.py``), whose
history is worth keeping because it explains the two rules below.

Two design decisions worth stating, because they are not obvious:

**Dates are parsed as intervals, not points.** ``2026-05`` and ``2026`` carry
real information (a 2019 source is clearly before a 2026 cutoff) but do not
identify a day. Returning a single date would force a lie in one direction or
the other. :func:`parse_source_interval` returns ``(earliest, latest)`` and
:func:`classify_date_status` only commits when the whole interval falls on one
side of ``T_cut``.

**Comparison is on civil dates, exclusive at ``T_cut``.** A source dated
``T_cut`` itself is treated as after the cut: the calendar date alone does not
say which timezone it refers to, so ``T_cut`` is the first date that could
contain post-boundary information anywhere in the world. Provider-side gates
that need an instant resolve this the same way, by using the earliest civil
timezone — see :func:`global_exclusive_cutoff_utc`.

**``T_cut`` is derived in exactly one place**, :func:`resolve_forecast_time`,
from three named inputs (the question's ``end_time``, the run's
``observation_time``, and the run's ``delta_days``). The forecast time is
*computed* (``end_time - delta_days``) rather than looked up in the task row, so
sweeping ΔT is a config change rather than a dataset regeneration.

A note on the date regexes below: none of them ends in ``\\b``. An earlier
version of this parser required a word boundary after the day
(``\\b(20\\d{2})[-/](\\d{1,2})[-/](\\d{1,2})\\b``) and therefore matched no
ISO-8601 timestamp at all (``2026-05-07T00:00:00.000Z`` -> no match, because
``T`` is a word character). Combined with a strict mode that dropped every
unlabelled card, it discarded 61.8% of genuinely valid pre-cutoff evidence.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Final, Mapping

__all__ = [
    "DATE_STATUSES",
    "DATE_STATUS_AFTER_CUT",
    "DATE_STATUS_FIELD",
    "DATE_STATUS_KNOWN_BEFORE",
    "DATE_STATUS_UNKNOWN",
    "SOURCE_DATE_KEYS",
    "TEMPORAL_POLICIES",
    "TEMPORAL_POLICY_DISABLED_CONTROL",
    "TEMPORAL_POLICY_STRICT",
    "ForecastWindow",
    "TemporalPolicyError",
    "classify_date_status",
    "global_exclusive_cutoff_utc",
    "mapping_date_status",
    "parse_bare_source_interval",
    "parse_cutoff",
    "parse_observation_time",
    "parse_source_date",
    "parse_source_interval",
    "resolve_forecast_time",
]


#: Evidence carries a parseable date that is provably before ``T_cut``.
DATE_STATUS_KNOWN_BEFORE: Final = "known_before"
#: No usable date, or a date whose span straddles ``T_cut``. Admitted, labelled.
DATE_STATUS_UNKNOWN: Final = "unknown"
#: Evidence is provably on/after ``T_cut``. Excluded from context.
DATE_STATUS_AFTER_CUT: Final = "after_cut"

DATE_STATUSES: Final = (
    DATE_STATUS_KNOWN_BEFORE,
    DATE_STATUS_UNKNOWN,
    DATE_STATUS_AFTER_CUT,
)

#: The one field name a labelled record carries. Consumers read this, not dates.
DATE_STATUS_FIELD: Final = "date_status"

#: Keys that carry *the record's own* publication date — never a date mentioned
#: inside its content. Used by :func:`mapping_date_status` to label a payload
#: that was not produced by the search adapter (a fetched page, an artifact).
#: Deliberately narrow: a key named ``date`` nested in a data table is a data
#: point, not a publication date, and treating it as one blocks whole pages for
#: containing an ordinary time series.
SOURCE_DATE_KEYS: Final = frozenset(
    {
        "date",
        "published_date",
        "publication_date",
        "source_date",
        "source_date_or_period",
        "source_period",
    }
)

# Fallbacks for shapes ``datetime.fromisoformat`` rejects. Deliberately small:
# anything that needs a locale-specific month name belongs to the provider
# adapter that produced it, not here.
_COMPACT_RE = re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")
_SLASH_RE = re.compile(r"(?<!\d)(20\d{2})/(\d{1,2})/(\d{1,2})(?!\d)")
_YEAR_MONTH_RE = re.compile(r"(?<!\d)(20\d{2})[-/](0[1-9]|1[0-2])(?![-/\d])")
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

# A value that is *nothing but* a date expression. Used by
# :func:`parse_bare_source_interval` to tell a timestamp cell apart from prose
# that happens to name a date. Covers every shape the parsers above accept:
# ISO date/datetime with optional time, fraction, and offset; slash-separated;
# compact; and year-month / year alone.
_BARE_DATE_RE = re.compile(
    r"""(?x)
    \s*(?:
        20\d{2}[-/]\d{1,2}[-/]\d{1,2}
            (?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?
               (?:Z|[+-]\d{2}:?\d{2})?)?
      | 20\d{2}\d{2}\d{2}
      | 20\d{2}[-/]\d{1,2}
      | 20\d{2}
    )\s*
    """
)

#: Delimiters stripped before deciding whether a cell *is* a date. Real table
#: cells carry them: FRED serves its series rows as ``"2026-08-19:"``.
_BARE_DATE_DELIMITERS: Final = "()[]{}<>\"'“”‘’,;:|=*·•　"


def _month_span(year: int, month: int) -> tuple[date, date] | None:
    try:
        last = calendar.monthrange(year, month)[1]
    except (ValueError, calendar.IllegalMonthError):
        return None
    return date(year, month, 1), date(year, month, last)


def _exact(year: int, month: int, day: int) -> tuple[date, date] | None:
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    return parsed, parsed


def parse_source_interval(value: Any) -> tuple[date, date] | None:
    """Parse a source date into an inclusive ``(earliest, latest)`` civil-date span.

    Returns ``None`` when nothing date-like is found. A full date yields a
    one-day span; ``2026-05`` yields that month; ``2026`` yields that year.
    Timezone offsets are discarded rather than converted: the civil date the
    publisher asserted is the quantity of interest, and converting it to UTC
    would shift some sources across the boundary for a reason unrelated to their
    content.
    """
    if isinstance(value, datetime):
        return value.date(), value.date()
    if isinstance(value, date):
        return value, value

    text = str(value or "").strip()
    if not text:
        return None

    # ISO-8601 is the dominant shape (provider structured fields, our own trace
    # records). ``fromisoformat`` accepts offsets and fractional seconds; the
    # ``Z`` suffix is normalized first for older interpreter behavior.
    candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    for attempt in (candidate, candidate[:10]):
        try:
            parsed = datetime.fromisoformat(attempt)
        except ValueError:
            continue
        return parsed.date(), parsed.date()

    match = _COMPACT_RE.search(text)
    if match:
        span = _exact(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if span:
            return span

    match = _SLASH_RE.search(text)
    if match:
        span = _exact(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if span:
            return span

    match = _YEAR_MONTH_RE.search(text)
    if match:
        span = _month_span(int(match.group(1)), int(match.group(2)))
        if span:
            return span

    match = _YEAR_RE.search(text)
    if match:
        year = int(match.group(1))
        try:
            return date(year, 1, 1), date(year, 12, 31)
        except ValueError:
            return None
    return None


def parse_source_date(value: Any) -> date | None:
    """Parse a source date to a single day, or ``None`` if not day-resolved.

    Convenience wrapper over :func:`parse_source_interval` for callers that need
    a concrete day (display, sorting). Prefer :func:`classify_date_status` for
    any cutoff decision — a month- or year-only date is genuinely not a day, and
    collapsing it here would hide that.
    """
    span = parse_source_interval(value)
    if span is None or span[0] != span[1]:
        return None
    return span[0]


def parse_bare_source_interval(value: Any) -> tuple[date, date] | None:
    """Like :func:`parse_source_interval`, but the *whole* value must be a date.

    :func:`parse_source_interval` searches inside the text, which is right for a
    provider's date field (``"Published 2026-05-07"``) and wrong for a data
    cell: ``"T-Note maturing 2031-02-15"`` is an instrument label, not a
    timestamp, and treating it as one prunes a legitimate row out of a table.

    A value counts as bare when it parses as a date on its own, ignoring
    surrounding punctuation but not surrounding words. Delimiters are stripped
    first because real table cells carry them, and rejecting ``"2026-08-19:"``
    for the trailing colon left an entire channel of post-cutoff data
    unredacted. Stripping is the right fix rather than loosening the match,
    because a looser pattern reads ``"(2026-08-19)"`` as the whole of 2026 and
    shifts the row's date by months.
    """
    if isinstance(value, (date, datetime)):
        return parse_source_interval(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return None
    text = str(value).strip().strip(_BARE_DATE_DELIMITERS).strip()
    if not text:
        return None
    span = parse_source_interval(text)
    if span is None:
        return None
    return span if _BARE_DATE_RE.fullmatch(text) else None


def classify_date_status(value: Any, T_cut: date | None) -> str:  # noqa: N803
    """Label one piece of evidence relative to ``T_cut``.

    ``after_cut`` requires proof: the *earliest* possible date is on/after the
    cutoff. ``known_before`` also requires proof: the *latest* possible date is
    strictly before it. Everything else — missing, unparseable, or a span that
    straddles the boundary — is ``unknown``.

    ``unknown`` is admitted rather than dropped. Dropping it costs a third of
    all real evidence (measured upstream: 36.6% of provider cards expose no
    parseable date) and buys no safety, because a genuinely leaking undated page
    is precisely the case an "undated" filter cannot see.
    """
    if T_cut is None:
        return DATE_STATUS_UNKNOWN
    span = parse_source_interval(value)
    if span is None:
        return DATE_STATUS_UNKNOWN
    earliest, latest = span
    if earliest >= T_cut:
        return DATE_STATUS_AFTER_CUT
    if latest < T_cut:
        return DATE_STATUS_KNOWN_BEFORE
    return DATE_STATUS_UNKNOWN


def mapping_date_status(
    value: Mapping[str, Any],
    T_cut: date | None,  # noqa: N803
    *,
    default: str = DATE_STATUS_UNKNOWN,
) -> str:
    """Label a record that carries its own publication date in a known field.

    Prefers an existing :data:`DATE_STATUS_FIELD` — if the adapter that produced
    the record already labelled it, that label wins, because the adapter saw the
    provider's native format and this function only sees a string. Otherwise the
    *shallow* keys in :data:`SOURCE_DATE_KEYS` are classified and the strongest
    claim wins: one proven ``after_cut`` date makes the record ``after_cut``.

    Shallow by design. Recursing would pick up every date inside a page's
    content and block a page for quoting a future scheduled event, which is not
    leakage — the page itself may predate the cutoff entirely.
    """
    if not isinstance(value, Mapping):
        return default
    existing = value.get(DATE_STATUS_FIELD)
    if isinstance(existing, str) and existing in DATE_STATUSES:
        return existing
    seen: set[str] = set()
    for key, item in value.items():
        if str(key or "").strip().lower() in SOURCE_DATE_KEYS:
            seen.add(classify_date_status(item, T_cut))
    if DATE_STATUS_AFTER_CUT in seen:
        return DATE_STATUS_AFTER_CUT
    if DATE_STATUS_KNOWN_BEFORE in seen:
        return DATE_STATUS_KNOWN_BEFORE
    return default


def parse_cutoff(value: Any) -> date | None:
    """Parse a run's ``T_cut`` / cutoff configuration value.

    Stricter than :func:`parse_source_interval`: a cutoff is configuration, so
    it must be an unambiguous single day. Callers are expected to treat ``None``
    on a non-empty input as a configuration error rather than "no cutoff" —
    silently degrading to "no cutoff" turns a typo into an unmarked open-book
    run.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    span = parse_source_interval(text)
    if span is None or span[0] != span[1]:
        return None
    return span[0]


# --- deriving the boundary --------------------------------------------------

#: Enforce a boundary; a run that cannot determine one must fail at startup.
TEMPORAL_POLICY_STRICT: Final = "strict"
#: Deliberately unfiltered. Only for the leakage-control arm of an ablation, and
#: it must be *declared*: an absent field is not a policy.
TEMPORAL_POLICY_DISABLED_CONTROL: Final = "disabled_control"

TEMPORAL_POLICIES: Final = (
    TEMPORAL_POLICY_STRICT,
    TEMPORAL_POLICY_DISABLED_CONTROL,
)


class TemporalPolicyError(ValueError):
    """Raised when a run cannot determine the boundary it claims to enforce.

    Deliberately fatal: a run that silently searches open-book while reporting
    itself as time-truncated produces a number that looks valid and is not.
    """


@dataclass(frozen=True)
class ForecastWindow:
    """The boundary plus every input that produced it.

    All fields travel together into the run artifacts. Without them a backtest
    cannot be reproduced and cannot even answer "where was this run actually
    cut?" — which is why they are one object rather than a bare date.

    ``T_cut`` (the civil date) and ``T_cut_instant`` are both carried, and the
    date is **not** recoverable from the instant: the instant is UTC+14 midnight
    expressed in UTC, so ``2026-05-10`` becomes ``2026-05-09T10:00:00Z`` and
    ``.date()`` on it reads 05-09. Storing only the instant and calling
    ``.date()`` silently shifts the boundary a day earlier.
    """

    T_cut: date | None
    T_cut_instant: datetime | None
    observation_time: date | None
    forecast_time: date | None
    end_time: date | None
    delta_days: int | None
    #: ``end_time - forecast_time`` after clamping. Equal to ``delta_days`` unless
    #: the question did not exist that early, in which case it is smaller. Recorded
    #: separately because a run described as "ΔT=7" whose rows were partly clamped
    #: has a per-row horizon, and averaging across it silently mixes two horizons.
    effective_delta_days: int | None = None
    #: The question's creation day, when the caller supplied one. Kept so a reader
    #: can see why a row was clamped without re-joining the dataset.
    start_time: date | None = None
    t_cut_source: str = TEMPORAL_POLICY_STRICT
    policy: str = TEMPORAL_POLICY_STRICT

    def as_manifest_fields(self) -> dict[str, Any]:
        """The recorded values, JSON-ready."""
        return {
            "observation_time": (self.observation_time.isoformat() if self.observation_time else None),
            "forecast_time": (self.forecast_time.isoformat() if self.forecast_time else None),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "delta_days": self.delta_days,
            "effective_delta_days": self.effective_delta_days,
            "T_cut": self.T_cut.isoformat() if self.T_cut else None,
            "T_cut_instant": (self.T_cut_instant.isoformat() if self.T_cut_instant else None),
            "t_cut_source": self.t_cut_source,
            "temporal_policy": self.policy,
        }


def parse_observation_time(value: Any) -> date:
    """Parse the run's observation time, or fail.

    Separate from :func:`parse_cutoff` only in that there is no "absent" case:
    the observation time is captured once at the run entry point and passing it
    is mandatory. A default here would reintroduce the wall-clock read this
    whole design exists to remove.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = parse_cutoff(value)
    if parsed is None:
        msg = (
            f"observation_time {value!r} is not an unambiguous calendar day "
            "(expected YYYY-MM-DD). It is captured once at the run entry point "
            "and must be passed explicitly; there is no clock fallback."
        )
        raise TemporalPolicyError(msg)
    return parsed


def _parse_end_time(value: Any) -> date | None:
    """Parse the question's resolution date, or ``None`` if absent."""
    if value in (None, ""):
        return None
    parsed = parse_cutoff(value)
    if parsed is None:
        msg = (
            f"end_time {value!r} is not an unambiguous calendar day (expected "
            "YYYY-MM-DD). The forecast time is computed from it, so an "
            "uninterpretable value would silently disable time truncation."
        )
        raise TemporalPolicyError(msg)
    return parsed


def _normalize_delta_days(delta_days: Any) -> int | None:
    """Parse the run's forecast horizon; ``None`` means no derived forecast time."""
    if delta_days is None or str(delta_days).strip() == "":
        # A run that pins its cutoff by hand derives no forecast time, and
        # recording ``delta_days: 0`` for it would assert a horizon it never used.
        return None
    try:
        delta = int(delta_days)
    except (TypeError, ValueError) as exc:
        msg = f"delta_days {delta_days!r} is not an integer number of days."
        raise TemporalPolicyError(msg) from exc
    if delta < 0:
        msg = f"delta_days {delta_days!r} is negative; the forecast cannot be made after the outcome is known."
        raise TemporalPolicyError(msg)
    return delta


def resolve_forecast_time(  # noqa: PLR0913
    end_time: Any,
    *,
    observation_time: Any,
    delta_days: Any = None,
    start_time: Any = None,
    override: Any = None,
    policy: str = TEMPORAL_POLICY_STRICT,
) -> ForecastWindow:
    """Compute the forecast time and the boundary it implies. The only place this happens.

    Three inputs, one rule:

    * ``end_time`` — the day the question resolves, a *fact about the question*,
      read straight off the task row.
    * ``delta_days`` — the run's forecast horizon, declared in the run config.
      ``forecast_time = end_time - delta_days``: the forecast is made this many
      days before the outcome is known. ``None`` means the run does not derive a
      forecast time at all, which is only valid alongside an ``override``.
    * ``observation_time`` — when this run is considered to take place. Required,
      no default: it is captured once at the run entry point and threaded down.
      A clock read here would mean the same task replayed on two days filtered
      differently, which is exactly the irreproducibility being removed.

    ::

        forecast_time = end_time - delta_days   (None when delta_days is None)

        override present                        -> T_cut = override
        observation_time >= forecast_time       -> T_cut = forecast_time
        observation_time <  forecast_time       -> T_cut = None

    The last branch is the honest reading of a run that happens *before* the
    forecast time: the event has not resolved yet, so there is no future
    information in existence to leak, and no cut is needed. Using
    ``observation_time`` itself as the boundary would assert "today is the
    information boundary", a different claim that no caller wants.

    The forecast time is **computed, not looked up**, so sweeping ΔT is a
    property of the run rather than of the data file.
    """
    normalized_policy = str(policy or "").strip() or TEMPORAL_POLICY_STRICT
    if normalized_policy not in TEMPORAL_POLICIES:
        msg = f"temporal_policy {policy!r} is not one of {list(TEMPORAL_POLICIES)}."
        raise TemporalPolicyError(msg)

    delta = _normalize_delta_days(delta_days)
    resolution = _parse_end_time(end_time)
    forecast_time = resolution - timedelta(days=delta) if resolution is not None and delta is not None else None

    # Clamp forward to the day the question existed. Without this, ΔT=7 puts 417
    # of the reference cohort's 1979 rows (21.1%) at a forecast time *before
    # their own start_time* — the agent is asked to stand on a day when the
    # question had not been posed. That is not merely a tighter cutoff: a
    # question naming an entity that did not exist yet becomes unanswerable by
    # construction rather than hard, and its error lands in the accuracy number
    # as if it were a forecasting failure.
    #
    # The cost is that ΔT becomes an upper bound rather than a value, which is
    # why ``effective_delta_days`` is recorded per row. The alternative —
    # dropping the 21% — keeps ΔT exact but makes each ΔT arm score a different
    # subset, and comparing arms over different subsets is how a comparison
    # stops meaning anything.
    #
    # NOT clamped to ``observation_time``. A question whose forecast time is
    # still in the future is handled below by returning ``T_cut=None``: nothing
    # post-boundary exists yet, so there is nothing to filter.
    creation = _parse_end_time(start_time) if start_time not in (None, "") else None
    clamped_to_start = False
    if forecast_time is not None and creation is not None and forecast_time < creation:
        forecast_time = creation
        clamped_to_start = True

    effective_delta = (resolution - forecast_time).days if resolution is not None and forecast_time is not None else None

    if normalized_policy == TEMPORAL_POLICY_DISABLED_CONTROL:
        # The unfiltered leakage-control arm. Declared, never inferred from a
        # missing field, and labelled in the artifacts so no downstream reader
        # mistakes it for a temporally valid backtest.
        return ForecastWindow(
            T_cut=None,
            T_cut_instant=None,
            observation_time=(parse_observation_time(observation_time) if observation_time not in (None, "") else None),
            forecast_time=forecast_time,
            start_time=creation,
            end_time=resolution,
            delta_days=delta,
            effective_delta_days=effective_delta,
            t_cut_source=TEMPORAL_POLICY_DISABLED_CONTROL,
            policy=normalized_policy,
        )

    observed = parse_observation_time(observation_time)

    override_cutoff: date | None = None
    if override not in (None, ""):
        override_cutoff = parse_cutoff(override)
        if override_cutoff is None:
            msg = (
                f"cutoff override {override!r} is not an unambiguous calendar day "
                "(expected YYYY-MM-DD). Refusing to run: an uninterpretable "
                "cutoff would silently disable time truncation."
            )
            raise TemporalPolicyError(msg)

    if override_cutoff is not None:
        cutoff, source = override_cutoff, "override"
    elif forecast_time is None:
        msg = (
            "temporal_policy is 'strict' but this task yields no forecast time: it "
            f"has end_time={end_time!r} and the run declared delta_days="
            f"{delta_days!r}, with no cutoff override. Refusing to run: proceeding "
            "would search open-book while reporting the run as time-truncated. "
            "Declare temporal_policy: disabled_control if the unfiltered behaviour "
            "is intended."
        )
        raise TemporalPolicyError(msg)
    elif observed < forecast_time:
        # Live prediction: the event has not resolved, so nothing post-boundary
        # exists to be found. No cut, and the search adapters send no date bound.
        return ForecastWindow(
            T_cut=None,
            T_cut_instant=None,
            observation_time=observed,
            forecast_time=forecast_time,
            start_time=creation,
            end_time=resolution,
            delta_days=delta,
            effective_delta_days=effective_delta,
            t_cut_source="live_no_cut_needed",
            policy=normalized_policy,
        )
    else:
        cutoff = forecast_time
        source = "derived_clamped_to_start" if clamped_to_start else "derived"

    return ForecastWindow(
        T_cut=cutoff,
        T_cut_instant=global_exclusive_cutoff_utc(cutoff),
        observation_time=observed,
        forecast_time=forecast_time,
        start_time=creation,
        end_time=resolution,
        delta_days=delta,
        effective_delta_days=effective_delta,
        t_cut_source=source,
        policy=normalized_policy,
    )


#: UTC+14 is the earliest civil timezone: the first place a given calendar date
#: begins. Anchoring the exclusive instant there is what makes "before T_cut"
#: mean the same thing in every timezone.
_EARLIEST_CIVIL_TZ: Final = timezone(timedelta(hours=14))


def global_exclusive_cutoff_utc(value: Any) -> datetime:
    """Widen a date-only cutoff to the first instant it begins anywhere on Earth.

    A calendar date does not say which timezone it means, so ``T_cut`` is taken
    as the earliest instant at which *any* timezone has entered that date.
    Everything accepted is then before ``T_cut`` worldwide, and no provider
    adapter has to re-decide the question.

    An input that already carries a time is returned as an aware instant
    unchanged (assumed UTC when naive), since it names an instant rather than a
    day.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    cutoff_date = value if isinstance(value, date) else parse_cutoff(value)
    if cutoff_date is None:
        msg = f"cutoff {value!r} is not an unambiguous calendar day (expected YYYY-MM-DD)."
        raise TemporalPolicyError(msg)
    return datetime.combine(cutoff_date, time.min, tzinfo=_EARLIEST_CIVIL_TZ).astimezone(timezone.utc)
