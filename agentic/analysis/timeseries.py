# Ported unchanged from Milkyway-Harness-Agent
# (galaxy/forecast/state/timeseries_forecast.py). The two runtimes are compared
# against each other, so the numeric backbone must be the same code rather than
# a reimplementation that could differ in a tail case. Keep edits in sync.
# -*- coding: utf-8 -*-
"""Deterministic univariate time-series extrapolation for numeric forecasts.

Core responsibility: given a same-metric historical series that the agent has
already gathered from a source tool, produce an auditable numeric forecast
(point + uncertainty interval) by *fitting* the series, instead of anchoring a
single latest value and applying a hand-waved drift.

This module is pure (no network, no MCP, no LLM). It depends only on numpy;
scipy is used opportunistically for a robust Theil-Sen slope when available.
The MCP tool ``fit_timeseries_forecast`` is a thin wrapper over
``fit_timeseries`` — keeping the math here makes it unit-testable in isolation.

Design invariants:
- Never invent data: with fewer than 3 points we fall back to naive / drift and
  flag ``insufficient_history`` with a wide band, rather than pretending a fit.
- Every returned method carries a point and an 80% interval so the caller can
  compare naive-last vs drift vs linear and justify the pick.
- Intervals widen with the forecast horizon (∝ sqrt(steps)) — a 5-day-ahead
  band must be wider than a 1-day-ahead band for the same series.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # scipy is in the sandbox allowlist but guard anyway for pure-import safety
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:  # pragma: no cover - exercised only when scipy is absent
    _scipy_stats = None

SCHEMA_VERSION = "timeseries_forecast.v0"

# z-multipliers for a normal approximation of the predictive interval.
_Z80 = 1.2815515594
_Z50 = 0.6744897502


def _parse_date(value: Any) -> Optional[date]:
    """Parse a date from common string/`date`/`datetime` inputs; None on failure."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    # Accept an ISO datetime prefix (e.g. "2026-07-02T00:00:00") too.
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[: len(fmt) + 6], fmt).date()
        except Exception:
            continue
    try:
        return datetime.fromisoformat(text).date()
    except Exception:
        return None


def _coerce_value(value: Any) -> Optional[float]:
    """Coerce a numeric value, stripping thousands separators and stray symbols."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    text = str(value).strip().replace(",", "")
    # Keep leading sign, digits, and a single decimal point.
    cleaned = []
    seen_dot = False
    for ch in text:
        if ch.isdigit() or (ch == "-" and not cleaned):
            cleaned.append(ch)
        elif ch == "." and not seen_dot:
            cleaned.append(ch)
            seen_dot = True
    try:
        v = float("".join(cleaned))
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _normalize_points(points: Sequence[Any]) -> List[Tuple[Optional[date], float]]:
    """Normalize heterogeneous point inputs into a cleaned (date|None, value) list.

    Accepts dicts ({date/x, value/y/close}), (date, value) pairs, or bare values.
    Drops entries with no usable numeric value; deduplicates by date keeping the
    last occurrence; sorts by date when dates are present.
    """
    parsed: List[Tuple[Optional[date], float]] = []
    for item in points:
        d: Optional[date] = None
        v: Optional[float] = None
        if isinstance(item, dict):
            d = _parse_date(
                item.get("date")
                or item.get("d")
                or item.get("x")
                or item.get("time")
                or item.get("timestamp")
            )
            v = _coerce_value(
                item.get("value")
                if item.get("value") is not None
                else item.get("y")
                if item.get("y") is not None
                else item.get("close")
                if item.get("close") is not None
                else item.get("price")
            )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            d = _parse_date(item[0])
            v = _coerce_value(item[1])
        else:
            v = _coerce_value(item)
        if v is None:
            continue
        parsed.append((d, v))

    have_dates = all(d is not None for d, _ in parsed) and len(parsed) > 0
    if have_dates:
        # Deduplicate by date (keep last), then sort chronologically.
        by_date: Dict[date, float] = {}
        for d, v in parsed:
            by_date[d] = v  # type: ignore[index]
        parsed = sorted(((d, v) for d, v in by_date.items()), key=lambda t: t[0])  # type: ignore[arg-type]
    return parsed


def _x_axis(points: List[Tuple[Optional[date], float]]) -> Tuple[np.ndarray, bool, float]:
    """Return the x-axis (day offsets or indices), whether it is dated, and median step."""
    have_dates = all(d is not None for d, _ in points) and len(points) > 0
    if have_dates:
        base = points[0][0]
        xs = np.array([(d - base).days for d, _ in points], dtype=float)  # type: ignore[operator]
    else:
        xs = np.arange(len(points), dtype=float)
    steps = np.diff(xs)
    steps = steps[steps > 0]
    median_step = float(np.median(steps)) if steps.size else 1.0
    if median_step <= 0:
        median_step = 1.0
    return xs, have_dates, median_step


def _interval(point: float, sigma: float, horizon_scale: float) -> Dict[str, float]:
    """Symmetric predictive interval that widens with sqrt(horizon)."""
    spread = sigma * math.sqrt(max(horizon_scale, 1e-9))
    return {
        "point": point,
        "lo80": point - _Z80 * spread,
        "hi80": point + _Z80 * spread,
        "lo50": point - _Z50 * spread,
        "hi50": point + _Z50 * spread,
        "sigma": sigma,
    }


def fit_timeseries(
    points: Sequence[Any],
    *,
    horizon_date: Optional[str] = None,
    horizon_steps: Optional[float] = None,
    asof_date: Optional[str] = None,
    metric: Optional[str] = None,
    unit: Optional[str] = None,
) -> Dict[str, Any]:
    """Fit a univariate series and extrapolate to the target horizon.

    Args:
        points: historical observations (see ``_normalize_points`` for accepted shapes).
        horizon_date: target date "YYYY-MM-DD"; used with dated series to derive horizon.
        horizon_steps: explicit number of steps ahead (used when dates are absent,
            or as an override). Defaults to 1 step / to horizon_date when given.
        asof_date: the run's information cutoff; points after it are dropped to
            respect temporal leakage (only applied to dated series).
        metric, unit: echoed through for the caller's ledger; not used in math.

    Returns:
        A JSON-serializable dict (schema ``timeseries_forecast.v0``) with the
        recommended point, interval, per-method comparison, diagnostics, caveats,
        and a human-readable ``summary``.
    """
    pts = _normalize_points(points)
    asof = _parse_date(asof_date)
    if asof is not None:
        pts = [(d, v) for d, v in pts if d is None or d <= asof]

    if not pts:
        return {
            "schema_version": SCHEMA_VERSION,
            "error": "no_usable_points",
            "summary": "No usable numeric points were provided; cannot fit or extrapolate.",
            "n_points": 0,
            "caveats": ["Provide at least one (date, value) observation of the same metric."],
        }

    xs, have_dates, median_step = _x_axis(pts)
    ys = np.array([v for _, v in pts], dtype=float)
    n = len(pts)
    last_x = float(xs[-1])
    last_value = float(ys[-1])

    # --- Resolve the forecast horizon in x-axis units and in "steps". ---
    target_date = _parse_date(horizon_date)
    horizon_days: Optional[float] = None
    if have_dates and target_date is not None:
        horizon_days = float((target_date - pts[0][0]).days) - last_x  # type: ignore[operator]
        if horizon_days < 0:
            horizon_days = 0.0
        x_target = last_x + horizon_days
        horizon_scale = max(horizon_days / median_step, 0.0)
    elif horizon_steps is not None:
        steps = float(horizon_steps)
        x_target = last_x + steps * (median_step if have_dates else 1.0)
        horizon_scale = max(steps, 0.0)
        if have_dates:
            horizon_days = steps * median_step
    else:
        # Default: one median step ahead.
        x_target = last_x + (median_step if have_dates else 1.0)
        horizon_scale = 1.0
        if have_dates:
            horizon_days = median_step
    # A zero horizon still carries one step of predictive noise.
    interval_scale = max(horizon_scale, 1.0)

    methods: Dict[str, Dict[str, float]] = {}
    caveats: List[str] = []

    # --- Diff-based volatility for naive / drift bands. ---
    diffs = np.diff(ys) if n >= 2 else np.array([])
    step_sigma = float(np.std(diffs, ddof=1)) if diffs.size >= 2 else (
        float(abs(diffs[0])) if diffs.size == 1 else abs(last_value) * 0.05
    )

    # naive_last: random walk (forecast = last observation).
    methods["naive_last"] = _interval(last_value, step_sigma, interval_scale)

    # drift: random walk with drift = mean per-step change.
    if diffs.size >= 1:
        drift_per_step = float(np.mean(diffs))
        drift_point = last_value + drift_per_step * (horizon_scale if horizon_scale > 0 else 1.0)
        methods["drift"] = _interval(drift_point, step_sigma, interval_scale)

    # linear_ols: least-squares line with a proper prediction interval.
    slope_per_day: Optional[float] = None
    r2: Optional[float] = None
    resid_std: Optional[float] = None
    fit_sse: Dict[str, float] = {}  # original-unit SSE per fitted model, for comparison
    if n >= 3 and np.ptp(xs) > 0:
        b, a = np.polyfit(xs, ys, 1)  # y = b*x + a
        fitted = b * xs + a
        resid = ys - fitted
        dof = max(n - 2, 1)
        s = float(math.sqrt(float(np.sum(resid ** 2)) / dof))
        ss_tot = float(np.sum((ys - float(np.mean(ys))) ** 2))
        r2 = float(1.0 - float(np.sum(resid ** 2)) / ss_tot) if ss_tot > 0 else 1.0
        sxx = float(np.sum((xs - float(np.mean(xs))) ** 2))
        # Prediction-interval leverage term for a new x_target.
        lev = 1.0 + 1.0 / n + ((x_target - float(np.mean(xs))) ** 2) / sxx if sxx > 0 else 1.0
        pi_sigma = s * math.sqrt(max(lev, 0.0))
        lin_point = float(b * x_target + a)
        methods["linear_ols"] = {
            "point": lin_point,
            "lo80": lin_point - _Z80 * pi_sigma,
            "hi80": lin_point + _Z80 * pi_sigma,
            "lo50": lin_point - _Z50 * pi_sigma,
            "hi50": lin_point + _Z50 * pi_sigma,
            "sigma": pi_sigma,
        }
        slope_per_day = float(b) if have_dates else None
        resid_std = s

        # theil_sen: robust slope, resilient to outliers (scipy only).
        if _scipy_stats is not None:
            try:
                ts_slope, ts_intercept, _, _ = _scipy_stats.theilslopes(ys, xs)
                ts_point = float(ts_slope * x_target + ts_intercept)
                ts_resid = ys - (ts_slope * xs + ts_intercept)
                ts_sigma = float(np.median(np.abs(ts_resid)) * 1.4826) or step_sigma
                methods["theil_sen"] = _interval(ts_point, ts_sigma, interval_scale)
            except Exception:
                pass

        # Linear fit SSE (original units) for later model comparison.
        lin_sse = float(np.sum(resid ** 2))
        fit_sse["linear_ols"] = lin_sse

        # log_linear: for strictly-positive series with multiplicative growth.
        if np.all(ys > 0):
            lb, la = np.polyfit(xs, np.log(ys), 1)
            log_resid = np.log(ys) - (lb * xs + la)
            log_s = float(math.sqrt(float(np.sum(log_resid ** 2)) / dof))
            log_point = float(math.exp(lb * x_target + la))
            # Compare fit quality back in original units, not log units.
            log_fitted_orig = np.exp(lb * xs + la)
            log_sse = float(np.sum((ys - log_fitted_orig) ** 2))
            fit_sse["log_linear"] = log_sse
            methods["log_linear"] = {
                "point": log_point,
                "lo80": float(math.exp(lb * x_target + la - _Z80 * log_s)),
                "hi80": float(math.exp(lb * x_target + la + _Z80 * log_s)),
                "lo50": float(math.exp(lb * x_target + la - _Z50 * log_s)),
                "hi50": float(math.exp(lb * x_target + la + _Z50 * log_s)),
                "sigma": log_s,
            }

    # --- Recommend a method (parsimony-first, honest about short history). ---
    insufficient_history = n < 3
    if insufficient_history:
        recommended = "drift" if "drift" in methods else "naive_last"
        caveats.append(
            f"Only {n} point(s) of same-metric history; using a {recommended} baseline "
            "with a wide band rather than a fitted trend."
        )
    else:
        # Prefer the linear trend only when it explains the series well AND the
        # slope is material over the horizon; otherwise the series is better
        # modeled as a (drifting) random walk.
        recommended = "drift" if "drift" in methods else "naive_last"
        if "linear_ols" in methods and r2 is not None and r2 >= 0.5:
            recommended = "linear_ols"
            # Log-linear wins only if it genuinely fits better in original units
            # (a straight arithmetic line must stay linear, not go log).
            if "log_linear" in fit_sse and fit_sse["log_linear"] < 0.9 * fit_sse["linear_ols"]:
                recommended = "log_linear"
            elif _scipy_stats is not None and "theil_sen" in methods and r2 < 0.7:
                # Robust slope is safer when the linear fit is only moderate.
                recommended = "theil_sen"

    chosen = methods[recommended]
    point = float(chosen["point"])

    span_days = float(xs[-1] - xs[0]) if have_dates else float(n - 1)
    unit_str = f" {unit}" if unit else ""
    metric_str = metric or "the metric"
    summary = (
        f"Fitted {metric_str} on {n} points"
        + (f" spanning {int(span_days)}d" if have_dates else "")
        + f"; recommended={recommended} → point={point:.6g}{unit_str}"
        + f", 80% interval=[{chosen['lo80']:.6g}, {chosen['hi80']:.6g}]"
        + (f", slope/day={slope_per_day:.4g}" if slope_per_day is not None else "")
        + (f", R2={r2:.3f}" if r2 is not None else "")
        + (f"; horizon={int(horizon_days)}d ahead" if horizon_days is not None else "")
        + "."
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "metric": metric,
        "unit": unit,
        "asof_date": asof.isoformat() if asof else None,
        "horizon_date": target_date.isoformat() if target_date else None,
        "horizon_days": horizon_days,
        "n_points": n,
        "history_span_days": span_days if have_dates else None,
        "have_dates": have_dates,
        "first_date": pts[0][0].isoformat() if have_dates else None,
        "last_date": pts[-1][0].isoformat() if have_dates else None,
        "last_value": last_value,
        "recommended_method": recommended,
        "point": point,
        "interval_80": [float(chosen["lo80"]), float(chosen["hi80"])],
        "interval_50": [float(chosen["lo50"]), float(chosen["hi50"])],
        "slope_per_day": slope_per_day,
        "r2": r2,
        "resid_std": resid_std,
        "insufficient_history": insufficient_history,
        "methods": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in methods.items()},
        "caveats": caveats,
        "summary": summary,
    }
