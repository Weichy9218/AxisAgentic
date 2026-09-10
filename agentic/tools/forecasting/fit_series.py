# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Tool: fit a same-metric historical series and extrapolate to the target date.

The agent gathers the series itself with ``web_search`` and
``scrape_and_extract_info``; this tool only turns those observations into an
auditable point and interval. It never fetches anything, so it cannot cross the
information boundary — whatever the agent passes in is what it already had.

It exists because of a measured failure. Over a held-out batch of numeric
questions, answering with the last value published at or before the cutoff scored
0.7190 under the committed scoring rule while the answers actually produced
scored 0.4429; two of the worst were wrong by four orders of magnitude with the
correct value already in hand. Returning naive-last alongside drift and the
fitted models makes the anchor a candidate the agent has to argue against rather
than one it can silently drop.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agentic.analysis.timeseries import fit_timeseries
from agentic.contracts.messages import ToolResultStatus
from agentic.tools.base import CallableTool, ToolResult

logger = logging.getLogger(__name__)

_DESCRIPTION = (
    "Fit a historical series of the SAME metric and extrapolate it to the forecast "
    "target date. Use this for numeric questions once you have collected the series "
    "(daily closes, weekly inventory, monthly rates) from search or a page. It "
    "compares naive-last, drift, linear, robust and log-linear fits and returns a "
    "recommended point with an 80% interval plus fit diagnostics.\n\n"
    "It does no fetching: it only computes on the points you pass, so the answer is "
    "reproducible and stays inside whatever evidence you already gathered."
)

FIT_TIMESERIES_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "description": (
                "Observations of the SAME metric, oldest to newest. Each item is "
                '{"date": "YYYY-MM-DD", "value": <number>}. A bare list of numbers '
                "is accepted when the series has no dates (treated as evenly spaced)."
            ),
            "items": {"type": "object"},
        },
        "horizon_date": {
            "type": "string",
            "description": "Forecast target date, YYYY-MM-DD. Preferred when the series is dated.",
        },
        "horizon_steps": {
            "type": "number",
            "description": "Steps ahead to forecast when the series carries no dates.",
        },
        "asof_date": {
            "type": "string",
            "description": "Date of the last observation, YYYY-MM-DD, when it is not the last point's own date.",
        },
        "metric": {"type": "string", "description": "What the series measures, echoed back for provenance."},
        "unit": {
            "type": "string",
            "description": (
                "The unit the values are in, echoed back unchanged. State it: unit "
                "confusion is the single largest source of numeric error in this task family."
            ),
        },
    },
    "required": ["points"],
}


def _coerce_points(points: Any) -> list[Any]:
    """Accept a list, or a JSON string of a list, and normalise to a list."""
    if points is None:
        return []
    if isinstance(points, str):
        text = points.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else [parsed]
    if isinstance(points, list):
        return points
    return [points]


def _backtest_prefers_fit(
    points: list[Any], fit: dict[str, Any], horizon_steps: float | None
) -> tuple[str, dict[str, Any]]:
    """Decide between the fitted trend and persistence, tested at the real horizon.

    Refit on the series minus its tail, predict the whole tail in one jump — the
    same distance the caller is actually forecasting — and compare absolute error
    against naive-last. Persistence wins ties and wins whenever the test cannot be
    run, because extrapolating a slope is the claim that carries the burden.

    Measured on this task family, following the fit's own pick scored 0.6185 under
    the committed scoring rule against 0.7496 for persistence, so the default has
    to be the other way round from what the fit reports.
    """
    methods = fit.get("methods") or {}
    fitted_name = fit.get("recommended_method")
    naive = (methods.get("naive_last") or {}).get("point")
    if fitted_name in (None, "naive_last") or naive is None:
        return "naive_last", {"reason": "no trend model to test", "tested_steps": 0}

    # How far ahead the caller is going, in observations rather than days.
    span = fit.get("history_span_days")
    n = len(points)
    if horizon_steps and horizon_steps > 0:
        steps = float(horizon_steps)
    elif span and n > 1:
        per_obs = float(span) / (n - 1)
        steps = (fit.get("horizon_days") or per_obs) / per_obs if per_obs else 1.0
    else:
        steps = 1.0
    # The holdout must be the full horizon. Shortening it to fit the series would
    # answer a different question — whether the slope holds one step ahead — and
    # that is precisely the licence this guard exists to withhold.
    holdout = max(1, int(round(steps)))
    if n - holdout < 3:
        return "naive_last", {
            "reason": f"series has {n} points, too short to test a {holdout}-step trend",
            "tested_steps": 0,
        }

    train, tail = points[:-holdout], points[-holdout:]
    actual = _point_value(tail[-1])
    if actual is None:
        return "naive_last", {"reason": "trend could not be tested", "tested_steps": 0}
    try:
        sub = fit_timeseries(train, horizon_steps=float(holdout))
    except Exception:
        return "naive_last", {"reason": "trend could not be tested", "tested_steps": 0}
    sub_methods = sub.get("methods") or {}
    cand = (sub_methods.get(fitted_name) or {}).get("point")
    base = (sub_methods.get("naive_last") or {}).get("point")
    if cand is None or base is None:
        return "naive_last", {"reason": "trend could not be tested", "tested_steps": 0}

    err_fit, err_naive = abs(cand - actual), abs(base - actual)
    evidence = {
        "tested_steps": holdout,
        "holdout_abs_error_fit": round(err_fit, 6),
        "holdout_abs_error_naive_last": round(err_naive, 6),
    }
    if err_fit < err_naive:
        evidence["reason"] = f"{fitted_name} beat naive_last {holdout} steps ahead on held-out history"
        return fitted_name, evidence
    evidence["reason"] = f"{fitted_name} did not beat naive_last {holdout} steps ahead on held-out history"
    return "naive_last", evidence


def _point_value(item: Any) -> float | None:
    if isinstance(item, dict):
        item = item.get("value")
    try:
        return float(item)
    except (TypeError, ValueError):
        return None


def create_fit_timeseries_tool(
    *,
    name: str = "fit_timeseries_forecast",
    description: str = _DESCRIPTION,
    parameters: dict[str, Any] | None = None,
    server_name: str | None = None,
    max_calls_per_task: int | None = None,
) -> CallableTool:
    """Build the series-fitting tool. Pure compute; no network, no credentials."""

    async def _fit(
        points: Any = None,
        horizon_date: str | None = None,
        horizon_steps: float | None = None,
        asof_date: str | None = None,
        metric: str | None = None,
        unit: str | None = None,
        **_ignored: Any,
    ) -> ToolResult:
        parsed = _coerce_points(points)
        if not parsed:
            # Say what is missing rather than returning an empty fit: an agent that
            # reads "ok" here would treat a nonexistent forecast as a result.
            return ToolResult(
                content="No usable points were supplied. Pass the observed series as "
                '[{"date": "YYYY-MM-DD", "value": <number>}, ...], oldest first.',
                status=ToolResultStatus.FAILED,
                metadata={"success": False, "error": "no_points", "n_points": 0},
            )
        try:
            result = fit_timeseries(
                parsed,
                horizon_date=horizon_date,
                horizon_steps=horizon_steps,
                asof_date=asof_date,
                metric=metric,
                unit=unit,
            )
        except Exception as exc:
            logger.warning("fit_timeseries_forecast failed: %s", exc)
            return ToolResult(
                content=f"Series fit failed: {exc}",
                status=ToolResultStatus.FAILED,
                metadata={"success": False, "error": str(exc), "n_points": len(parsed)},
            )

        methods = (result or {}).get("methods") or {}
        chosen, evidence = _backtest_prefers_fit(parsed, result, horizon_steps)
        chosen_point = (methods.get(chosen) or {}).get("point")
        if chosen == "naive_last" and chosen_point is None:
            chosen_point = result.get("last_value")

        # The fit's own pick stays visible; what changes is which one is put
        # forward, and the agent is shown the holdout evidence for that choice.
        result = dict(result)
        result["fit_recommended_method"] = result.get("recommended_method")
        result["recommended_method"] = chosen
        result["point"] = chosen_point
        result["selection"] = evidence
        result["summary"] = (
            f"{result.get('n_points')} points; putting forward {chosen} = {chosen_point}"
            f"{(' ' + unit) if unit else ''}. {evidence.get('reason')}. "
            f"The fit's own pick was {result['fit_recommended_method']}. "
            "Report the value in the unit the source published."
        )
        return ToolResult(
            content=json.dumps(result, ensure_ascii=False),
            metadata={
                "success": True,
                "n_points": len(parsed),
                "recommended_method": chosen,
                "recommended_point": chosen_point,
                "fit_recommended_method": result["fit_recommended_method"],
                "metric": metric or "",
                "unit": unit or "",
            },
        )

    return CallableTool(
        name=name,
        description=description,
        server_name=server_name,
        parameters=parameters or FIT_TIMESERIES_PARAMETERS,
        strict_mode=False,
        fn=_fit,
        max_calls_per_task=max_calls_per_task,
        emoji="📈",
    )
