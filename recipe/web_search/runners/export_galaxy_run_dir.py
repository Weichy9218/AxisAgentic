# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Export an AxisAgentic forecast run into the shape the Milkyway scorer reads.

``scripts/eval/score_offline.py`` in the Milkyway repository walks a run
directory for ``task_summary.jsonl`` files and, for each task, branches on
whether the row carries a ``probability`` **key**::

    if "probability" in row:
        output = row["canonical_prediction"] or row["prediction"]
        probability = row["probability"]
    else:
        ... fall back to a finalize event inside <task_id>/main_agent.jsonl

This exporter always takes the first branch. Writing the probability into the
summary row is why no per-task subdirectory has to be synthesised: the fallback
path exists for runs recorded before that field did, and imitating it would mean
fabricating an agent transcript in another runtime's private format.

**Why reuse that scorer instead of reimplementing it.** The two systems are
being compared, and a comparison in which each side computes its own score
measures the scorers as much as the agents. Protocol v2 (``docs/SCORING_SPEC.md``
upstream) also has enough surface — a max-entropy completion for single-probability
choice submissions, an asinh/logistic integrated Brier for numeric targets — that
an independent implementation would differ somewhere, and the difference would
land in the comparison.

**Nothing is fabricated.** A task whose submission could not be read is written
with an empty prediction, so it lands in ``n_excluded`` with the scorer's own
reason rather than becoming a wrong answer. Filling a missing probability from
the predicted direction is the specific bug the upstream scorer's docstring
records shipping 11 times: it did not add noise, it biased the arm comparison.

**The notebook mirror** (``--emit-notebook``) exists for the leakage page rather
than the score. ``render_leakage_report.py`` and ``audit_time_filtering.py``
read ``<task>/notebook/*.search.json`` and take the gate counters from
``provenance.as_of``; AxisAgentic records the same counters, under the same
names, on ``tool_calls[].metadata.as_of``. Rewriting them into that layout lets
both arms be measured by one unmodified script, which is the whole reason the
scorer is reused rather than reimplemented.

One difference is worth stating rather than hiding: galaxy's notebook holds
artifacts *written to a workspace*, which the agent may or may not re-read,
while this mirror holds the tool results themselves — exactly the bytes that
entered the model's context. For a leakage scan that is the stricter corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from agentic.temporal import ForecastWindow, TemporalPolicyError, resolve_forecast_time
from recipe.web_search.eval.forecast_submission import normalize_boxed_answer

logger = logging.getLogger(__name__)

#: Written into every row so a reader can tell where the file came from without
#: joining it back to a run config.
EXPORTER_ID = "axisagentic.export_galaxy_run_dir.v1"

#: Trace files are named per attempt; summary rows are named per task.
_ATTEMPT_SUFFIX = re.compile(r"_attempt-\d+$")

#: A path component cannot exceed 255 bytes on ext4, and 64 of this benchmark's
#: ids are percent-encoded JSON longer than 200. galaxy truncates at exactly this
#: limit, so mirroring the rule means the leakage scan covers the same task set
#: on both arms — including dropping the same three ids that collide after
#: truncation. Diverging here would make the two pages count different corpora.
_MAX_DIR_BYTES = 255

#: Copied onto each search packet's provenance. These are the names galaxy uses,
#: and the leakage page reads them positionally by name.
_PROVIDER_PROVENANCE_KEYS = (
    "provider_temporal_filter_requested",
    "provider_temporal_filter_encoded",
    "provider_temporal_granularity",
    "provider_temporal_semantics",
    "compiled_provider_end",
)


def _task_dir_name(task_id: str) -> str:
    """The directory a task's notebook lives in, under the 255-byte path limit."""
    encoded = task_id.encode("utf-8")
    if len(encoded) <= _MAX_DIR_BYTES:
        return task_id
    return encoded[:_MAX_DIR_BYTES].decode("utf-8", errors="ignore")


def _load_dataset_rows(data_path: Path) -> dict[str, dict[str, Any]]:
    """Index the benchmark rows by task id, for the fields the boundary needs."""
    rows: dict[str, dict[str, Any]] = {}
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            task_id = str(record.get("task_id") or "").strip()
            if task_id:
                rows[task_id] = record
    return rows


def _load_results(run_dir: Path) -> list[dict[str, Any]]:
    """Read every ``benchmark_results.jsonl`` under *run_dir*, newest row per task.

    ``rglob`` mirrors the scorer's own walk, so a sharded AxisAgentic run (one
    ``run_N`` directory per repetition) exports the same way a sharded Milkyway
    run scores.
    """
    latest: dict[str, dict[str, Any]] = {}
    paths = sorted(run_dir.rglob("benchmark_results.jsonl"))
    if not paths:
        msg = f"No benchmark_results.jsonl found under {run_dir}"
        raise FileNotFoundError(msg)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                record = json.loads(text)
                task_id = str(record.get("task_id") or "").strip()
                if task_id:
                    latest[task_id] = record
    logger.info("Read %d task results from %d file(s) under %s", len(latest), len(paths), run_dir)
    return list(latest.values())


def _load_effort(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-task rollout and token counters, in the shape ``run_shape`` reads.

    ``benchmark_results.jsonl`` records the score-bearing fields and two cheap
    counters (``num_turns``, ``tool_count``); it has never carried model calls,
    token usage or context size. Those live only in the trace, so the effort
    half of a report is empty unless the trace is read.

    An empty effort table is not a neutral omission. ``render_metrics_html.py``
    renders a missing ``llm_calls_total`` as **"LLM 调用中位数 0 / 全轮 0 次"**,
    which a reader takes as a measurement — this agent made no model calls —
    rather than as a gap in the export. Published beside a comparison arm that
    does report the field, that misstates the one asymmetry the comparison is
    most likely to be read for.

    Field mapping, and where it is not exact:

    ``llm_calls_total``
        ``run_info.inference_timing.num_inference_calls`` — every request sent
        to the model, retries and rolled-back turns included. It is not
        ``turn_used``: a turn that hit the generation limit and was retried is
        one turn and two calls. Cost follows calls, so calls is what travels,
        and ``num_turns`` stays beside it for whoever wants the other one.
    ``estimated_input_tokens_peak`` / ``_last``
        Max and final ``token_usage.per_step[].input_tokens``. Measured rather
        than estimated; the key keeps the scorer's name, not a claim about this
        runtime's accuracy.
    ``llm_message_count``
        Conversation entries excluding the runtime's own state-transition rows,
        which share the list but are never sent to the model.

    ``cached_tokens`` is deliberately absent: this runtime records no cache-hit
    field, and a zero would read as "caching was off" rather than "not
    measured". The scorer treats an absent key as unknown and reports no hit
    rate, which is the honest outcome.
    """
    effort: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.rglob("web-search-benchmark/*.json")):
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        task_id = _ATTEMPT_SUFFIX.sub("", str(trace.get("task_id") or ""))
        if not task_id:
            continue

        meta = trace.get("metadata") or {}
        usage = trace.get("token_usage") or {}
        per_step = [
            int(step.get("input_tokens") or 0)
            for step in (usage.get("per_step") or [])
            if isinstance(step, dict)
        ]
        timing = (meta.get("run_info") or {}).get("inference_timing") or {}

        calls = [c for c in (trace.get("tool_calls") or []) if isinstance(c, dict)]
        by_tool: dict[str, int] = {}
        for call in calls:
            name = str(call.get("tool_name") or "")
            if name:
                by_tool[name] = by_tool.get(name, 0) + 1

        conversation = trace.get("conversation") or []
        llm_messages = sum(
            1 for m in conversation
            if isinstance(m, dict)
            and str(m.get("role")) in {"system", "user", "assistant", "tool"}
        )

        metrics: dict[str, Any] = {
            "tool_calls": {"total": len(calls), "unique_tools_count": len(by_tool)},
            "top_tools": [
                {"name": name, "count": count}
                for name, count in sorted(by_tool.items(), key=lambda kv: -kv[1])
            ],
        }
        if isinstance(timing.get("num_inference_calls"), (int, float)):
            metrics["llm_calls_total"] = int(timing["num_inference_calls"])
        if isinstance(meta.get("turn_used"), (int, float)):
            metrics["num_turns"] = int(meta["turn_used"])

        llm_usage: dict[str, int] = {}
        for source, target in (("input_tokens", "prompt_tokens"),
                               ("output_tokens", "completion_tokens"),
                               ("total_tokens", "total_tokens")):
            if isinstance(usage.get(source), (int, float)):
                llm_usage[target] = int(usage[source])
        if llm_usage:
            metrics["llm_usage"] = llm_usage

        window: dict[str, int] = {"message_count": len(conversation),
                                  "llm_message_count": llm_messages}
        if per_step:
            window["estimated_input_tokens_peak"] = max(per_step)
            window["estimated_input_tokens_last"] = per_step[-1]
        metrics["context_window"] = window

        effort[task_id] = metrics

    logger.info("Read effort counters for %d task(s) from traces under %s",
                len(effort), run_dir)
    return effort


def _market_age_days(window: ForecastWindow) -> int | None:
    """How old the question was on the day the forecast was made.

    The scorer bands this into ``by_age``; a question posed the same day it is
    forecast is a different kind of task from one that has been open for months.
    """
    if window.forecast_time is None or window.start_time is None:
        return None
    return (window.forecast_time - window.start_time).days


def _submission_fields(result: dict[str, Any]) -> tuple[str, float | None]:
    """Decide what answer and probability this task submits.

    A contract violation (a refusal, a LaTeX wrapper, a probability appended to
    the answer) is withheld: the answer text is the thing being refused, and
    forwarding it would let the scorer grade a string this runtime already
    judged invalid. Every other rejection forwards what was found, so the scorer
    assigns the exclusion reason in its own vocabulary.

    The answer is passed through :func:`normalize_boxed_answer` here as well as
    at parse time, because a run finished before that normalisation existed
    still has LaTeX spacing macros in its stored results and would otherwise be
    scored on a string the model did not mean.
    """
    if result.get("contract_violation"):
        return "", None
    output = result.get("output") or result.get("raw_output") or ""
    probability = result.get("probability")
    if probability is None:
        probability = result.get("raw_probability")
    if not isinstance(probability, (int, float)) or isinstance(probability, bool):
        probability = None
    return normalize_boxed_answer(str(output)), probability


def build_summary_row(
    *,
    result: dict[str, Any],
    dataset_row: dict[str, Any],
    window: ForecastWindow,
    search_provider: str,
    max_tool_calls_configured: int | None,
    exported_at: str,
    effort: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one ``task_summary.jsonl`` row for the Milkyway scorer."""
    task_id = str(result.get("task_id") or "")
    prediction, probability = _submission_fields(result)
    submission_reason = result.get("submission_reason")

    # The result file's two counters are the floor; the trace adds model calls,
    # tokens and context size when it was read. Starting from the result keeps a
    # row well-formed for a task whose trace is missing or unparseable, which is
    # the case a live run hits constantly.
    metrics: dict[str, Any] = {
        "tool_calls": {"total": result.get("tool_count")},
        "num_turns": result.get("num_turns"),
    }
    if effort:
        metrics = {**metrics, **effort}

    row: dict[str, Any] = {
        "id": task_id,
        "prediction": prediction,
        "canonical_prediction": prediction,
        # The key must be present even when null: its presence is what selects
        # the summary branch in the scorer.
        "probability": probability,
        "status": "success" if result.get("reason") else "error",
        "ended_at": exported_at,
        "total_latency_seconds": result.get("task_elapsed_s"),
        "metrics": metrics,
        "max_num_tool_calls_configured": max_tool_calls_configured,
        "search_provider": search_provider,
        "canonical_status": "usable" if submission_reason is None else "unusable",
        "canonical_forecast_status": "valid" if submission_reason is None else "invalid",
        "market_age_days": _market_age_days(window),
        # Provenance for this file, not read by the scorer.
        "exporter": EXPORTER_ID,
        "axis_task_type": dataset_row.get("task_type"),
        "axis_submission_reason": submission_reason,
        "axis_submission_detail": result.get("submission_detail"),
        "axis_stop_reason": result.get("reason"),
    }
    # T_cut / delta_days / effective_delta_days / observation_time / temporal_policy.
    # The scorer reads effective_delta_days for by_delta_days; the rest make the
    # backtest reproducible, which is the reason they travel as one object.
    row.update(window.as_manifest_fields())
    return row


def export_notebooks(
    *,
    run_dir: Path,
    out_dir: Path,
    known_task_ids: set[str],
) -> dict[str, Any]:
    """Mirror each task's tool results into the notebook layout the audits read.

    Three kinds of file are written, and the split is load-bearing:

    ``art-NNNN.search.json``
        One search packet, carrying ``provenance.as_of``. The leakage page
        counts gates from these files **only**, so writing page-channel counters
        here would inflate our gate table relative to an arm whose page channel
        is not screened, and the two numbers would stop being comparable.

    ``art-NNNN.page.md``
        The extracted text a scrape returned — the bytes the model read. Scanned
        for post-cutoff dates like any other artifact.

    ``page_gates.json``
        The page-channel ``as_of`` counters, wrapped so that the scan's
        ``_strip_provenance`` removes the whole document: these are our own
        diagnostics, and a stamp's ``T_cut`` scanned as content would report one
        false post-cutoff date per page.

    Rejected and errored tool calls are skipped. A rejected call returned nothing
    to the model, so counting it would credit the screen with bytes that never
    existed.
    """
    trace_paths = sorted(run_dir.rglob("web-search-benchmark/*.json"))
    written = {"search": 0, "page": 0}
    tasks_written = 0
    skipped_unknown: list[str] = []
    unstamped_live = 0
    page_gates_total: dict[str, int] = {}

    for path in trace_paths:
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        task_id = _ATTEMPT_SUFFIX.sub("", str(trace.get("task_id") or ""))
        if not task_id:
            continue
        if task_id not in known_task_ids:
            skipped_unknown.append(task_id)
            continue

        notebook = out_dir / _task_dir_name(task_id) / "notebook"
        notebook.mkdir(parents=True, exist_ok=True)

        ordinal = 0
        page_stamps: list[dict[str, Any]] = []
        for call in trace.get("tool_calls") or []:
            if call.get("status") != "success":
                continue
            meta = call.get("metadata") or {}
            stamp = meta.get("as_of")
            tool = call.get("tool_name")
            ordinal += 1

            if tool == "web_search":
                if not isinstance(stamp, dict):
                    # A live search result that reached the model without the
                    # screen having run. The audit's headline check.
                    unstamped_live += 1
                    continue
                provenance: dict[str, Any] = {
                    "search_provider": "serper",
                    "engines_attempted": ["serper"],
                    "engines_used": ["serper"],
                    "retrieved_at": trace.get("started_at"),
                    "as_of": stamp,
                }
                for key in _PROVIDER_PROVENANCE_KEYS:
                    if key in meta:
                        provenance[key] = meta[key]
                if meta.get("as_of_blocked"):
                    provenance["as_of_blocked"] = meta["as_of_blocked"]
                packet = {
                    "query": (call.get("arguments") or {}).get("query"),
                    "source_cards": _search_cards(call.get("content")),
                    "provenance": provenance,
                }
                target = notebook / f"art-{ordinal:04d}.search.json"
                target.write_text(
                    json.dumps(packet, ensure_ascii=False, indent=1, default=_json_default) + "\n",
                    encoding="utf-8",
                )
                written["search"] += 1

            elif tool == "scrape_and_extract_info":
                if isinstance(stamp, dict):
                    page_stamps.append(stamp)
                    for key, value in stamp.items():
                        if isinstance(value, int) and not isinstance(value, bool):
                            page_gates_total[key] = page_gates_total.get(key, 0) + value
                elif str(call.get("content") or "").strip():
                    unstamped_live += 1
                body = str(call.get("content") or "")
                if not body.strip():
                    continue
                url = meta.get("url") or (call.get("arguments") or {}).get("url") or ""
                target = notebook / f"art-{ordinal:04d}.page.md"
                target.write_text(f"<!-- source: {url} -->\n\n{body}\n", encoding="utf-8")
                written["page"] += 1

        if page_stamps:
            (notebook.parent / "notebook" / "page_gates.json").write_text(
                json.dumps({"provenance": {"as_of_page_body": page_stamps}}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
        tasks_written += 1

    return {
        "n_traces": len(trace_paths),
        "n_tasks_with_notebook": tasks_written,
        "n_search_packets": written["search"],
        "n_page_artifacts": written["page"],
        "n_live_results_without_as_of_stamp": unstamped_live,
        "n_traces_not_in_summary": len(skipped_unknown),
        "page_channel_gate_totals": dict(sorted(page_gates_total.items())),
    }


def _search_cards(content: Any) -> list[dict[str, Any]]:
    """The model-visible cards from a ``web_search`` result payload."""
    if not isinstance(content, str):
        return []
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []
    cards = payload.get("organic") if isinstance(payload, dict) else None
    return cards if isinstance(cards, list) else []


def export_run(
    *,
    run_dir: Path,
    data_path: Path,
    out_dir: Path,
    temporal_policy: str,
    delta_days: int | None,
    observation_time: str | None,
    cutoff_override: str | None,
    search_provider: str,
    max_tool_calls_configured: int | None,
    emit_notebook: bool = False,
) -> dict[str, Any]:
    """Write ``<out_dir>/task_summary.jsonl`` and return a summary of what was written."""
    dataset_rows = _load_dataset_rows(data_path)
    results = _load_results(run_dir)
    effort = _load_effort(run_dir)
    exported_at = datetime.now(tz=timezone.utc).isoformat()

    rows: list[dict[str, Any]] = []
    missing_from_dataset: list[str] = []
    boundary_failures: dict[str, str] = {}
    reason_counts: dict[str, int] = {}

    for result in sorted(results, key=lambda item: str(item.get("task_id") or "")):
        task_id = str(result.get("task_id") or "")
        dataset_row = dataset_rows.get(task_id)
        if dataset_row is None:
            # Do not invent a boundary for a row the dataset does not contain.
            # The scorer counts these separately as run tasks outside the truth
            # set, which is a config mismatch worth seeing rather than hiding.
            missing_from_dataset.append(task_id)
            continue
        try:
            window = resolve_forecast_time(
                dataset_row.get("end_time"),
                observation_time=observation_time,
                delta_days=delta_days,
                start_time=dataset_row.get("start_time"),
                override=cutoff_override,
                policy=temporal_policy,
            )
        except TemporalPolicyError as exc:
            boundary_failures[task_id] = str(exc)
            continue
        rows.append(
            build_summary_row(
                result=result,
                dataset_row=dataset_row,
                window=window,
                search_provider=search_provider,
                max_tool_calls_configured=max_tool_calls_configured,
                exported_at=exported_at,
                effort=effort.get(task_id),
            )
        )
        reason = result.get("submission_reason")
        key = str(reason) if reason else "usable"
        reason_counts[key] = reason_counts.get(key, 0) + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "task_summary.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

    summary = {
        "out_path": str(out_path),
        "n_rows": len(rows),
        "n_dataset_rows": len(dataset_rows),
        "n_run_results": len(results),
        "n_missing_from_dataset": len(missing_from_dataset),
        "missing_from_dataset": missing_from_dataset[:20],
        "n_boundary_failures": len(boundary_failures),
        "boundary_failures": dict(list(boundary_failures.items())[:5]),
        "submission_reason_counts": dict(sorted(reason_counts.items())),
        "temporal_policy": temporal_policy,
        "delta_days": delta_days,
        "observation_time": observation_time,
        "search_provider": search_provider,
    }
    if emit_notebook:
        summary["notebook"] = export_notebooks(
            run_dir=run_dir,
            out_dir=out_dir,
            known_task_ids={str(row["id"]) for row in rows},
        )
    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an AxisAgentic forecast run for the Milkyway offline scorer.",
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="AxisAgentic run directory (holds benchmark_results.jsonl)")
    parser.add_argument("--data-path", required=True, type=Path, help="The benchmark JSONL the run was scored against")
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Where to write task_summary.jsonl. Keep it outside the dataset directory.",
    )
    parser.add_argument(
        "--temporal-policy",
        required=True,
        help="Declared boundary policy: 'strict' or 'disabled_control'. There is no default.",
    )
    parser.add_argument("--delta-days", type=int, default=None, help="Forecast horizon; required under 'strict'")
    parser.add_argument("--observation-time", default=None, help="YYYY-MM-DD; required under 'strict'")
    parser.add_argument("--cutoff-override", default=None, help="Pin T_cut directly instead of deriving it")
    parser.add_argument("--search-provider", default="serper")
    parser.add_argument("--max-tool-calls", type=int, default=None, help="Recorded for run-shape diagnostics")
    parser.add_argument(
        "--emit-notebook",
        action="store_true",
        help="Also mirror each task's tool results into <task>/notebook/, which is "
        "what render_leakage_report.py and audit_time_filtering.py read.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    summary = export_run(
        run_dir=args.run_dir,
        data_path=args.data_path,
        out_dir=args.out_dir,
        temporal_policy=args.temporal_policy,
        delta_days=args.delta_days,
        observation_time=args.observation_time,
        cutoff_override=args.cutoff_override,
        search_provider=args.search_provider,
        max_tool_calls_configured=args.max_tool_calls,
        emit_notebook=args.emit_notebook,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["n_boundary_failures"]:
        logger.error("%d task(s) had no derivable boundary and were not exported", summary["n_boundary_failures"])
        return 1
    notebook = summary.get("notebook") or {}
    if notebook.get("n_live_results_without_as_of_stamp"):
        # The boundary check that cannot be waived: a live retrieval reached the
        # model without the screen having run against this task's T_cut.
        logger.error(
            "%d live tool result(s) carry no as_of stamp",
            notebook["n_live_results_without_as_of_stamp"],
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
