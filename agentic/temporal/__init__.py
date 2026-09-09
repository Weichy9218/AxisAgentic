# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Temporal boundary primitives: ``T_cut`` derivation, source-date parsing, provider encoding.

``window`` owns the semantic layer (civil dates, exclusive at ``T_cut``);
``provider`` owns the one place a boundary becomes vendor request syntax.
"""

from agentic.temporal.provider import (
    GRANULARITY_DAY,
    GRANULARITY_TIMESTAMP,
    SEMANTICS_EXCLUSIVE,
    SEMANTICS_INCLUSIVE,
    ProviderTemporalFilterUnsupported,
    ProviderTimeFilter,
    compile_provider_time_filter,
)
from agentic.temporal.window import (
    DATE_STATUS_AFTER_CUT,
    DATE_STATUS_FIELD,
    DATE_STATUS_KNOWN_BEFORE,
    DATE_STATUS_UNKNOWN,
    DATE_STATUSES,
    SOURCE_DATE_KEYS,
    TEMPORAL_POLICIES,
    TEMPORAL_POLICY_DISABLED_CONTROL,
    TEMPORAL_POLICY_STRICT,
    ForecastWindow,
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

__all__ = [
    "DATE_STATUSES",
    "DATE_STATUS_AFTER_CUT",
    "DATE_STATUS_FIELD",
    "DATE_STATUS_KNOWN_BEFORE",
    "DATE_STATUS_UNKNOWN",
    "GRANULARITY_DAY",
    "GRANULARITY_TIMESTAMP",
    "SEMANTICS_EXCLUSIVE",
    "SEMANTICS_INCLUSIVE",
    "SOURCE_DATE_KEYS",
    "TEMPORAL_POLICIES",
    "TEMPORAL_POLICY_DISABLED_CONTROL",
    "TEMPORAL_POLICY_STRICT",
    "ForecastWindow",
    "ProviderTemporalFilterUnsupported",
    "ProviderTimeFilter",
    "TemporalPolicyError",
    "classify_date_status",
    "compile_provider_time_filter",
    "global_exclusive_cutoff_utc",
    "mapping_date_status",
    "parse_bare_source_interval",
    "parse_cutoff",
    "parse_observation_time",
    "parse_source_date",
    "parse_source_interval",
    "resolve_forecast_time",
]
