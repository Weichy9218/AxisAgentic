# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Executable spec for provider-side boundary encoding.

The invariants here are about what the system may *claim*, not only about what
it computes. Two of them are negative assertions, and those are the point.
"""

from __future__ import annotations

import inspect
from datetime import date

import pytest

from agentic.temporal import (
    GRANULARITY_DAY,
    SEMANTICS_INCLUSIVE,
    ProviderTemporalFilterUnsupported,
    compile_provider_time_filter,
)


def test_serper_encodes_the_previous_day_because_cd_max_is_inclusive() -> None:
    """``T_cut`` is exclusive; ``cd_max`` is inclusive. The bridge is one day."""
    compiled = compile_provider_time_filter("serper", T_cut=date(2026, 6, 29))
    assert compiled.request_params == {"tbs": "cdr:1,cd_max:6/28/2026"}
    assert compiled.requested is True
    assert compiled.granularity == GRANULARITY_DAY
    assert compiled.semantics == SEMANTICS_INCLUSIVE
    assert compiled.compiled_end_date == date(2026, 6, 28)


def test_serper_sends_no_floor_parameter() -> None:
    """The interval is ``(-inf, T_cut)``. A floor caused 137 empty-interval failures upstream."""
    compiled = compile_provider_time_filter("serper", T_cut=date(2026, 6, 29))
    encoded = compiled.request_params["tbs"]
    assert "cd_min" not in encoded
    assert set(compiled.request_params) == {"tbs"}


def test_compile_signature_offers_no_lower_bound() -> None:
    """A floor must not be reachable by a caller, not merely unused by this one."""
    params = set(inspect.signature(compile_provider_time_filter).parameters)
    assert params == {"provider", "T_cut"}


def test_the_assertable_invariant_is_less_than_or_equal_not_equality() -> None:
    """Providers differ in precision, so demanding equality would lie about one of them."""
    cut = date(2026, 6, 29)
    compiled = compile_provider_time_filter("serper", T_cut=cut)
    assert compiled.compiled_end_date is not None
    assert compiled.compiled_end_date <= cut


def test_provenance_says_requested_and_encoded_never_applied() -> None:
    """This layer can assert the parameter was sent, not that the provider honoured it."""
    provenance = compile_provider_time_filter("serper", T_cut=date(2026, 6, 29)).as_provenance()
    assert provenance["provider_temporal_filter_requested"] is True
    assert provenance["provider_temporal_filter_encoded"] == "cdr:1,cd_max:6/28/2026"
    assert provenance["compiled_provider_end"] == "2026-06-28"
    assert not any("applied" in key for key in provenance)


def test_no_cutoff_compiles_to_no_parameters() -> None:
    """A declared unfiltered arm, or a live task, sends no date bound at all."""
    compiled = compile_provider_time_filter("serper", T_cut=None)
    assert compiled.request_params == {}
    assert compiled.requested is False
    assert compiled.encoded is None
    assert compiled.compiled_end_date is None
    assert compiled.as_provenance()["provider_temporal_filter_requested"] is False


def test_an_unknown_provider_fails_closed() -> None:
    """Refusing is the point: the alternative is an unbounded request that looks bounded."""
    with pytest.raises(ProviderTemporalFilterUnsupported, match="no time-filter compilation rule"):
        compile_provider_time_filter("exa", T_cut=date(2026, 6, 29))


def test_unsupported_is_a_value_error_so_adapters_must_order_their_excepts() -> None:
    assert issubclass(ProviderTemporalFilterUnsupported, ValueError)


def test_provider_name_is_matched_case_insensitively() -> None:
    assert compile_provider_time_filter("Serper", T_cut=date(2026, 6, 29)).requested is True
