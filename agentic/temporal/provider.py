# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Translate one boundary into one search provider's request parameters.

The single place ``T_cut`` becomes vendor syntax. Everything upstream reasons
about civil dates; everything here is format conversion. Ported from the
Milkyway ``galaxy`` runtime (``galaxy/workbench/search_providers.py``).

**Only an upper bound is ever compiled.** The search interval is
``(-inf, T_cut)``. No provider receives a floor parameter. That is a policy, not
an omission: an unbounded past costs nothing (the corpus is what it is), while a
floor was the source of empty-interval failures that wasted 137 recorded calls
upstream, plus a live gap where the floor reached one provider and silently did
not reach another.

**Fails closed.** When a boundary is required and the provider cannot express
it, this raises rather than returning a request without one. Adapters are
expected to catch :class:`ProviderTemporalFilterUnsupported` *before* issuing
any HTTP call, so an unbounded request never goes out.

**No field is named ``applied``.** What this module can assert is that the
parameter was compiled and sent, not that the provider honoured it. The bound
also applies to the date a publisher *claims*, and a claimed pre-cutoff date
does not guarantee pre-cutoff body content.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Final, Mapping

__all__ = [
    "GRANULARITY_DAY",
    "GRANULARITY_TIMESTAMP",
    "SEMANTICS_EXCLUSIVE",
    "SEMANTICS_INCLUSIVE",
    "ProviderTemporalFilterUnsupported",
    "ProviderTimeFilter",
    "compile_provider_time_filter",
]

#: The bound names a calendar day.
GRANULARITY_DAY: Final = "day"
#: The bound names an instant.
GRANULARITY_TIMESTAMP: Final = "timestamp"

#: The named boundary value is itself admitted.
SEMANTICS_INCLUSIVE: Final = "inclusive"
#: The named boundary value is itself excluded.
SEMANTICS_EXCLUSIVE: Final = "exclusive"


class ProviderTemporalFilterUnsupported(ValueError):
    """A boundary was required and the provider cannot express it.

    Subclasses :class:`ValueError`, so adapters that catch ``ValueError`` for
    argument problems must order their ``except`` clauses to catch this first.
    """


@dataclass(frozen=True)
class ProviderTimeFilter:
    """The compiled request parameters plus what can honestly be claimed about them."""

    request_params: Mapping[str, Any]
    requested: bool
    encoded: str | None
    granularity: str | None
    semantics: str | None
    #: The last calendar day the bound admits. The assertable invariant is
    #: ``compiled_end_date <= T_cut``, **not** equality: providers differ in
    #: precision, so demanding equality necessarily lies about one of them.
    compiled_end_date: date | None

    def as_provenance(self) -> dict[str, Any]:
        """Provenance fields for the tool trace. Deliberately has no ``applied`` key."""
        return {
            "provider_temporal_filter_requested": self.requested,
            "provider_temporal_filter_encoded": self.encoded,
            "provider_temporal_granularity": self.granularity,
            "provider_temporal_semantics": self.semantics,
            "compiled_provider_end": (self.compiled_end_date.isoformat() if self.compiled_end_date else None),
        }


def _serper_cd_max_day(T_cut: date) -> date:  # noqa: N803
    """The inclusive last day Serper's ``cd_max`` may name for an exclusive ``T_cut``.

    ``cd_max`` is inclusive — verified upstream by live probe rather than from
    documentation: a single-day window ``cd_min=cd_max=D`` returned results dated
    exactly ``D``, and no absolutely-dated result newer than ``cd_max`` came back
    across six windows. So the exclusive boundary is encoded by naming the
    previous day.

    That probe's first version concluded the opposite ("leaks up to 3 days")
    because it judged the gate using relative strings ("1 week ago") resolved
    against the wall clock. Those span 5-9 days, so a date past ``cd_max`` was
    equally explained by rounding: the criterion could not tell a leak from its
    own imprecision. Only absolutely-dated results can answer the question.
    """
    return T_cut - timedelta(days=1)


_NO_FILTER: Final = ProviderTimeFilter(
    request_params={},
    requested=False,
    encoded=None,
    granularity=None,
    semantics=None,
    compiled_end_date=None,
)


def compile_provider_time_filter(provider: str, *, T_cut: date | None) -> ProviderTimeFilter:  # noqa: N803
    """Compile ``T_cut`` into *provider*'s request parameters.

    Args:
        provider: Provider name. Only ``"serper"`` has a compilation rule in this
            repository; anything else raises rather than going out unbounded.
        T_cut: Exclusive civil-date boundary, or ``None`` for an unfiltered run
            (a declared ``disabled_control`` arm, or a live task whose forecast
            time has not arrived).

    Raises:
        ProviderTemporalFilterUnsupported: *provider* has no compilation rule.
    """
    if T_cut is None:
        return _NO_FILTER

    name = str(provider or "").strip().lower()
    if name == "serper":
        cd_max = _serper_cd_max_day(T_cut)
        # cd_max only. There is no lower bound to express; a hardcoded
        # ``cd_min:1/1/2000`` would be a fabricated floor.
        encoded = f"cdr:1,cd_max:{cd_max.month}/{cd_max.day}/{cd_max.year}"
        return ProviderTimeFilter(
            request_params={"tbs": encoded},
            requested=True,
            encoded=encoded,
            granularity=GRANULARITY_DAY,
            semantics=SEMANTICS_INCLUSIVE,
            compiled_end_date=cd_max,
        )

    msg = (
        f"provider {provider!r} has no time-filter compilation rule. Add one here "
        "rather than letting the request go out unbounded."
    )
    raise ProviderTemporalFilterUnsupported(msg)
