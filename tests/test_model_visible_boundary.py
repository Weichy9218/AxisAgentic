# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Audit: the run owns the information boundary, and the model cannot touch it.

These read the tool's *actual* schema and the *actual* request payload rather
than the documentation, because the failure they guard against is a boundary
that looks enforced and is not.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from agentic.temporal import ProviderTemporalFilterUnsupported
from agentic.tools.web_search import SIMPLE_WEB_SEARCH_PARAMETERS, create_web_search_tool
from agentic.tools.web_search.search import _build_payload
from agentic.temporal import compile_provider_time_filter

T_CUT = date(2026, 6, 29)

#: Anything the runtime injects must not be reachable from the tool schema.
MUST_BE_HIDDEN = ("T_cut", "t_cut", "tbs", "cd_max", "search_provider", "cutoff", "as_of")


def _schema_properties(tool) -> set[str]:
    definition = tool.to_tool_definition()
    return set(definition["function"]["parameters"].get("properties", {}))


def test_the_boundary_is_not_in_the_model_visible_schema() -> None:
    """Read from the built tool, not from the docs: this must be a fact about code."""
    tool = create_web_search_tool(parameters=SIMPLE_WEB_SEARCH_PARAMETERS, t_cut=T_CUT)
    properties = _schema_properties(tool)
    for hidden in MUST_BE_HIDDEN:
        assert hidden not in properties
    assert properties == {"query", "num", "gl", "hl"}


def test_the_rendered_definition_never_mentions_the_boundary_anywhere() -> None:
    """Not in a description or a default either, only in the request the runtime builds."""
    tool = create_web_search_tool(parameters=SIMPLE_WEB_SEARCH_PARAMETERS, t_cut=T_CUT)
    rendered = json.dumps(tool.to_tool_definition())
    assert "2026-06-29" not in rendered
    assert "cd_max" not in rendered
    assert "T_cut" not in rendered


def test_a_model_supplied_time_range_cannot_widen_the_boundary() -> None:
    """Assignment, not setdefault. A model that could set ``tbs`` could unset the cutoff."""
    compiled = compile_provider_time_filter("serper", T_cut=T_CUT)
    payload = _build_payload(
        "q",
        num=10,
        gl="us",
        hl="en",
        location=None,
        # The widest recency filter Google accepts, as if the model asked for it.
        time_range="qdr:y",
        page=None,
        autocorrect=None,
        time_filter=compiled,
    )
    assert payload["tbs"] == "cdr:1,cd_max:6/28/2026"


def test_without_a_boundary_a_model_supplied_time_range_still_works() -> None:
    """The override exists to protect a boundary, not to disable a normal feature."""
    payload = _build_payload(
        "q",
        num=10,
        gl="us",
        hl="en",
        location=None,
        time_range="qdr:y",
        page=None,
        autocorrect=None,
        time_filter=compile_provider_time_filter("serper", T_cut=None),
    )
    assert payload["tbs"] == "qdr:y"


def test_an_unencodable_provider_fails_when_the_tool_is_built_not_mid_run() -> None:
    """Fail at second zero. A boundary discovered to be unencodable on task 47 has already leaked."""
    with pytest.raises(ProviderTemporalFilterUnsupported):
        create_web_search_tool(parameters=SIMPLE_WEB_SEARCH_PARAMETERS, t_cut=T_CUT, search_provider="exa")


def test_no_boundary_means_no_date_parameter_is_sent() -> None:
    payload = _build_payload(
        "q",
        num=10,
        gl="us",
        hl="en",
        location=None,
        time_range=None,
        page=None,
        autocorrect=None,
        time_filter=compile_provider_time_filter("serper", T_cut=None),
    )
    assert "tbs" not in payload
