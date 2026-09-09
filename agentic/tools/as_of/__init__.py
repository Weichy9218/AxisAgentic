# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""As-of screening: three gates over one currency, run where evidence enters.

Flow order is ``units`` -> ``rules``/``structured`` -> ``judge`` -> ``screen``,
with ``boundary`` stamping the result and checking at the context edge.
"""

from agentic.tools.as_of.boundary import (
    AS_OF_STAMP_KEY,
    POST_CUTOFF_GUARDRAIL,
    AsOfCutoffError,
    AsOfScreeningError,
    assert_screened,
    blocked_payload,
    resolve_cutoff,
    stamp,
)
from agentic.tools.as_of.cache import CachedVerdict, VerdictCache, get_verdict_cache
from agentic.tools.as_of.judge import UNDATABLE, JudgeUnavailable, judge_units
from agentic.tools.as_of.rules import is_after_cut, needs_judgement
from agentic.tools.as_of.screen import (
    GATE_JUDGE,
    GATE_JUDGE_OVERFLOW,
    GATE_JUDGE_UNAVAILABLE,
    GATE_PROVIDER_DATE,
    GATE_STRUCTURED,
    BlockedUnit,
    ScreenOutcome,
    screen_units,
)
from agentic.tools.as_of.structured import (
    REDACTION_MARKER,
    carries_post_cutoff_listing,
    mentions_late_date,
    prune_post_cutoff_records,
)
from agentic.tools.as_of.units import (
    CHANNEL_PAGE,
    CHANNEL_SEARCH,
    EvidenceUnit,
    chunk_document_text,
    units_from_search_cards,
)

__all__ = [
    "AS_OF_STAMP_KEY",
    "CHANNEL_PAGE",
    "CHANNEL_SEARCH",
    "GATE_JUDGE",
    "GATE_JUDGE_OVERFLOW",
    "GATE_JUDGE_UNAVAILABLE",
    "GATE_PROVIDER_DATE",
    "GATE_STRUCTURED",
    "POST_CUTOFF_GUARDRAIL",
    "REDACTION_MARKER",
    "UNDATABLE",
    "AsOfCutoffError",
    "AsOfScreeningError",
    "BlockedUnit",
    "CachedVerdict",
    "EvidenceUnit",
    "JudgeUnavailable",
    "ScreenOutcome",
    "VerdictCache",
    "assert_screened",
    "blocked_payload",
    "carries_post_cutoff_listing",
    "chunk_document_text",
    "get_verdict_cache",
    "is_after_cut",
    "judge_units",
    "mentions_late_date",
    "needs_judgement",
    "prune_post_cutoff_records",
    "resolve_cutoff",
    "screen_units",
    "stamp",
    "units_from_search_cards",
]
