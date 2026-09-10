"""Pin the series tool: what it puts forward, and when it refuses to trend.

The tool was ported because copying the last observed value scored 0.7496 on this
task family while the answers actually produced scored 0.4100. Following the
fit's own recommendation scored 0.6185 — worse than copying — so the wrapper adds
a horizon-matched holdout test and defaults to persistence. These tests pin that
policy, not the arithmetic, which lives in agentic/analysis/timeseries.py.
"""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from agentic.contracts.messages import ToolResultStatus
from agentic.tools.forecasting import create_fit_timeseries_tool

if TYPE_CHECKING:
    import pytest


def _run(points: Any, **kw: Any) -> dict:
    tool = create_fit_timeseries_tool()
    result = asyncio.run(tool._fn(points=points, **kw))
    return {"result": result, "payload": json.loads(result.content) if result.content.startswith("{") else {}}


def _series(values: list[float], start_day: int = 1) -> list[dict]:
    return [{"date": f"2026-06-{start_day + i:02d}", "value": v} for i, v in enumerate(values)]


def test_missing_points_fails_loudly_instead_of_returning_an_empty_fit() -> None:
    out = _run([])
    assert out["result"].status is ToolResultStatus.FAILED
    assert out["result"].metadata["error"] == "no_points"
    assert "YYYY-MM-DD" in out["result"].content


def test_a_flat_series_puts_forward_the_last_value() -> None:
    out = _run(_series([100.0, 100.2, 99.8, 100.1, 100.0, 99.9, 100.1, 100.0]), horizon_steps=5)
    assert out["payload"]["recommended_method"] == "naive_last"
    assert abs(out["payload"]["point"] - 100.0) < 1e-6


def test_a_short_series_keeps_persistence_because_the_trend_cannot_be_tested() -> None:
    out = _run(_series([10.0, 12.0, 14.0, 16.0]), horizon_steps=4)
    payload = out["payload"]
    assert payload["recommended_method"] == "naive_last"
    assert payload["selection"]["tested_steps"] == 0
    assert "too short" in payload["selection"]["reason"] or "could not be tested" in payload["selection"]["reason"]


def test_a_clean_trend_that_holds_up_at_the_horizon_is_allowed_through() -> None:
    out = _run(_series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]), horizon_steps=2)
    payload = out["payload"]
    assert payload["recommended_method"] != "naive_last"
    assert payload["selection"]["holdout_abs_error_fit"] < payload["selection"]["holdout_abs_error_naive_last"]
    assert payload["point"] > 28.0


def test_the_fits_own_pick_stays_visible_when_the_wrapper_overrides_it() -> None:
    # A series that drifts then reverses: the fit will want a slope, the holdout
    # at the horizon will not support one.
    out = _run(_series([50.0, 52.0, 54.0, 56.0, 58.0, 57.0, 55.0, 53.0, 51.0, 50.0]), horizon_steps=4)
    payload = out["payload"]
    assert "fit_recommended_method" in payload
    if payload["recommended_method"] == "naive_last":
        assert payload["fit_recommended_method"] != "naive_last"
        assert "did not beat naive_last" in payload["selection"]["reason"]


def test_the_unit_is_echoed_because_unit_confusion_is_the_dominant_error() -> None:
    out = _run(_series([845.0, 844.0, 843.0, 842.0, 841.0, 840.0]), horizon_steps=3,
               metric="COMEX gold inventory", unit="tonnes")
    assert out["result"].metadata["unit"] == "tonnes"
    assert "tonnes" in out["payload"]["summary"]


def test_points_may_arrive_as_a_json_string() -> None:
    out = _run(json.dumps(_series([1.0, 1.1, 1.2, 1.15, 1.18, 1.2])), horizon_steps=2)
    assert out["result"].status is ToolResultStatus.SUCCESS
    assert out["result"].metadata["n_points"] == 6


def test_the_prompt_rule_appears_only_when_the_tool_is_registered() -> None:
    from recipe.web_search.agent.prompts import generate_system_prompt

    without = generate_system_prompt(prompt_profile="forecast", t_cut="2026-05-13")
    with_tool = generate_system_prompt(prompt_profile="forecast", t_cut="2026-05-13",
                                       fit_timeseries_enabled=True)
    assert "fit_timeseries_forecast" not in without
    assert "fit_timeseries_forecast" in with_tool
