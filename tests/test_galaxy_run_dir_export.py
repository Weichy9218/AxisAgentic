# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Executable spec for the export bridge to the Milkyway offline scorer.

The assertions here restate that scorer's read contract. If it changes, these
fail rather than the run silently exporting rows nothing can grade.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic.temporal import resolve_forecast_time
from recipe.web_search.runners.export_galaxy_run_dir import build_summary_row, export_run

DATASET = [
    {
        "task_id": "t-binary",
        "task_question": "Q\n\nOptions:\n- YES\n- NO\n",
        "task_type": "BINARY",
        "ground_truth": "YES",
        "start_time": "2026-06-29",
        "end_time": "2026-07-06",
        "source": "polymarket",
        "source_url": "https://example.invalid/1",
    },
    {
        "task_id": "t-numeric",
        "task_question": "Q",
        "task_type": "NUMERIC",
        "ground_truth": "845.47",
        "start_time": "2026-06-01",
        "end_time": "2026-07-06",
        "source": "futurex",
        "source_url": "https://example.invalid/2",
    },
]

RESULTS = [
    {
        "task_id": "t-binary",
        "output": "YES",
        "probability": 0.72,
        "raw_output": "YES",
        "raw_probability": 0.72,
        "contract_violation": False,
        "submission_reason": None,
        "reason": "assistant_completed",
        "task_elapsed_s": 12.5,
        "tool_count": 4,
        "num_turns": 6,
    },
    {
        "task_id": "t-numeric",
        "output": "857.64",
        "probability": None,
        "raw_output": "857.64",
        "raw_probability": None,
        "contract_violation": False,
        "submission_reason": None,
        "reason": "assistant_completed",
        "task_elapsed_s": 30.0,
        "tool_count": 9,
        "num_turns": 11,
    },
]


def _write_run(tmp_path: Path, results: list[dict]) -> tuple[Path, Path]:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir(parents=True)
    with (run_dir / "benchmark_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
    data_path = tmp_path / "data.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for row in DATASET:
            handle.write(json.dumps(row) + "\n")
    return run_dir, data_path


def _export(tmp_path: Path, results: list[dict], **overrides) -> list[dict]:
    run_dir, data_path = _write_run(tmp_path, results)
    kwargs = {
        "run_dir": run_dir,
        "data_path": data_path,
        "out_dir": tmp_path / "out",
        "temporal_policy": "strict",
        "delta_days": 7,
        "observation_time": "2026-09-01",
        "cutoff_override": None,
        "search_provider": "serper",
        "max_tool_calls_configured": 30,
    }
    kwargs.update(overrides)
    export_run(**kwargs)
    lines = (tmp_path / "out" / "task_summary.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_every_row_carries_the_probability_key_so_the_summary_branch_is_taken(tmp_path: Path) -> None:
    """Key *presence*, not truthiness, is what the scorer branches on.

    A numeric row's probability is null and must still be written, or the scorer
    falls through to a finalize event this runtime never produces.
    """
    rows = _export(tmp_path, RESULTS)
    assert len(rows) == 2
    for row in rows:
        assert "probability" in row
    numeric = next(row for row in rows if row["id"] == "t-numeric")
    assert numeric["probability"] is None


def test_the_scorer_reads_id_and_canonical_prediction(tmp_path: Path) -> None:
    rows = {row["id"]: row for row in _export(tmp_path, RESULTS)}
    assert rows["t-binary"]["canonical_prediction"] == "YES"
    assert rows["t-binary"]["prediction"] == "YES"
    assert rows["t-binary"]["probability"] == pytest.approx(0.72)


def test_latex_spacing_in_a_stored_result_is_expanded_at_export_time(tmp_path: Path) -> None:
    """The run finished before the parser learned this; the export must still fix it.

    `\\boxed{BK\\ Hacken}` was recorded verbatim by an earlier run, and the
    scorer compares the submission against the question's options as strings, so
    the spacing macro turns a correct answer into `output_not_in_options`.
    """
    results = [dict(RESULTS[0], output="YES\\ PLEASE", raw_output="YES\\ PLEASE"), RESULTS[1]]
    rows = {row["id"]: row for row in _export(tmp_path, results)}
    assert rows["t-binary"]["prediction"] == "YES PLEASE"
    assert rows["t-binary"]["canonical_prediction"] == "YES PLEASE"


def test_boundary_fields_travel_with_every_row(tmp_path: Path) -> None:
    """Without them the backtest cannot be reproduced or even located in time."""
    rows = {row["id"]: row for row in _export(tmp_path, RESULTS)}
    binary = rows["t-binary"]
    assert binary["T_cut"] == "2026-06-29"
    assert binary["t_cut_source"] == "derived"
    assert binary["temporal_policy"] == "strict"
    assert binary["delta_days"] == 7
    assert binary["effective_delta_days"] == 7
    assert binary["observation_time"] == "2026-09-01"
    # Posed and forecast on the same day.
    assert binary["market_age_days"] == 0
    # This one was created five weeks before its forecast time.
    assert rows["t-numeric"]["market_age_days"] == 28


def test_a_clamped_row_reports_the_horizon_it_actually_used(tmp_path: Path) -> None:
    rows = {row["id"]: row for row in _export(tmp_path, RESULTS, delta_days=30)}
    binary = rows["t-binary"]
    assert binary["t_cut_source"] == "derived_clamped_to_start"
    assert binary["delta_days"] == 30
    assert binary["effective_delta_days"] == 7


def test_disabled_control_exports_a_null_boundary_that_says_so(tmp_path: Path) -> None:
    rows = _export(tmp_path, RESULTS, temporal_policy="disabled_control", observation_time="2026-09-01")
    for row in rows:
        assert row["T_cut"] is None
        assert row["t_cut_source"] == "disabled_control"
        assert row["temporal_policy"] == "disabled_control"


def test_an_unreadable_submission_exports_an_empty_prediction_not_a_guess(tmp_path: Path) -> None:
    """An empty prediction lands in n_excluded. A filled one would bias the comparison."""
    results = [
        {
            **RESULTS[0],
            "output": None,
            "probability": None,
            "raw_output": None,
            "raw_probability": None,
            "submission_reason": "submission_has_no_answer",
        }
    ]
    rows = _export(tmp_path, results)
    assert rows[0]["prediction"] == ""
    assert rows[0]["probability"] is None
    assert rows[0]["canonical_forecast_status"] == "invalid"


def test_a_contract_violation_is_withheld_rather_than_forwarded(tmp_path: Path) -> None:
    """The answer text is the thing being refused, so it must not reach the scorer."""
    results = [
        {
            **RESULTS[0],
            "output": None,
            "probability": None,
            "raw_output": "YES, 58%",
            "raw_probability": 0.58,
            "contract_violation": True,
            "submission_reason": "no_valid_submission",
        }
    ]
    rows = _export(tmp_path, results)
    assert rows[0]["prediction"] == ""
    assert rows[0]["probability"] is None


def test_an_out_of_range_probability_is_forwarded_for_the_scorer_to_classify(tmp_path: Path) -> None:
    """The scorer owns the exclusion vocabulary; forwarding lets it say its own reason."""
    results = [
        {
            **RESULTS[0],
            "output": None,
            "probability": None,
            "raw_output": "YES",
            "raw_probability": 64.0,
            "contract_violation": False,
            "submission_reason": "probability_out_of_range",
        }
    ]
    rows = _export(tmp_path, results)
    assert rows[0]["prediction"] == "YES"
    assert rows[0]["probability"] == pytest.approx(64.0)


def test_a_task_absent_from_the_dataset_is_reported_not_invented(tmp_path: Path) -> None:
    run_dir, data_path = _write_run(tmp_path, [*RESULTS, {**RESULTS[0], "task_id": "t-ghost"}])
    summary = export_run(
        run_dir=run_dir,
        data_path=data_path,
        out_dir=tmp_path / "out",
        temporal_policy="strict",
        delta_days=7,
        observation_time="2026-09-01",
        cutoff_override=None,
        search_provider="serper",
        max_tool_calls_configured=30,
    )
    assert summary["n_missing_from_dataset"] == 1
    assert summary["missing_from_dataset"] == ["t-ghost"]
    assert summary["n_rows"] == 2


def test_build_summary_row_is_json_serializable() -> None:
    window = resolve_forecast_time("2026-07-06", observation_time="2026-09-01", delta_days=7, start_time="2026-06-29")
    row = build_summary_row(
        result=RESULTS[0],
        dataset_row=DATASET[0],
        window=window,
        search_provider="serper",
        max_tool_calls_configured=30,
        exported_at="2026-09-07T00:00:00+00:00",
    )
    assert json.loads(json.dumps(row))["T_cut"] == "2026-06-29"


# --- the notebook mirror -----------------------------------------------------
#
# These pin the layout that ``render_leakage_report.py`` and
# ``audit_time_filtering.py`` read. Both scripts live in another repository and
# are run unmodified, so their read contract can only be held here.

TRACE = {
    "task_id": "t-binary_attempt-1",
    "started_at": "2026-09-08T02:00:00+00:00",
    "tool_calls": [
        {
            "tool_name": "web_search",
            "status": "success",
            "arguments": {"query": "rate decision"},
            "content": json.dumps({"organic": [{"title": "T", "snippet": "S", "date": "2026-06-01"}]}),
            "metadata": {
                "as_of": {"T_cut": "2026-06-29", "scope": "search_cards", "units": 5, "kept": 4},
                "provider_temporal_filter_encoded": "cdr:1,cd_max:6/28/2026",
                "compiled_provider_end": "2026-06-28",
            },
        },
        {
            "tool_name": "scrape_and_extract_info",
            "status": "success",
            "arguments": {"url": "https://example.invalid/p"},
            "content": "The committee held rates.",
            "metadata": {
                "as_of": {"T_cut": "2026-06-29", "scope": "page_body", "chunks": 3, "kept": 2},
                "url": "https://example.invalid/p",
            },
        },
        {
            "tool_name": "web_search",
            "status": "rejected",
            "content": "over budget",
            "metadata": {},
        },
    ],
}


def _write_trace(run_dir: Path, trace: dict) -> None:
    traces = run_dir / "web-search-benchmark"
    traces.mkdir(parents=True, exist_ok=True)
    name = str(trace["task_id"]).replace("/", "_")
    (traces / f"{name}.json").write_text(json.dumps(trace), encoding="utf-8")


def _export_with_notebook(tmp_path: Path, trace: dict) -> Path:
    run_dir, data_path = _write_run(tmp_path, RESULTS)
    _write_trace(run_dir, trace)
    export_run(
        run_dir=run_dir,
        data_path=data_path,
        out_dir=tmp_path / "out",
        temporal_policy="strict",
        delta_days=7,
        observation_time="2026-09-01",
        cutoff_override=None,
        search_provider="serper",
        max_tool_calls_configured=50,
        emit_notebook=True,
    )
    return tmp_path / "out"


def test_the_notebook_directory_is_named_by_the_task_not_the_attempt(tmp_path: Path) -> None:
    """A trace file is per attempt; the audit joins on the summary row's id."""
    out = _export_with_notebook(tmp_path, TRACE)
    assert (out / "t-binary" / "notebook").is_dir()
    assert not (out / "t-binary_attempt-1").exists()


def test_a_search_packet_carries_the_gate_counters_where_the_audit_reads_them(tmp_path: Path) -> None:
    out = _export_with_notebook(tmp_path, TRACE)
    packets = sorted((out / "t-binary" / "notebook").glob("*.search.json"))
    assert len(packets) == 1
    packet = json.loads(packets[0].read_text(encoding="utf-8"))
    assert packet["provenance"]["as_of"]["T_cut"] == "2026-06-29"
    assert packet["provenance"]["as_of"]["scope"] == "search_cards"
    assert packet["provenance"]["compiled_provider_end"] == "2026-06-28"


def test_page_channel_counters_stay_out_of_the_search_packets(tmp_path: Path) -> None:
    """The leakage page's gate table sums ``*.search.json`` only.

    Our page channel is screened and the comparison arm's is not, so mixing the
    two here would make one arm's gate counts look larger for a reason that has
    nothing to do with what leaked.
    """
    out = _export_with_notebook(tmp_path, TRACE)
    for path in (out / "t-binary" / "notebook").glob("*.search.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["provenance"]["as_of"]["scope"] == "search_cards"
    gates = json.loads((out / "t-binary" / "notebook" / "page_gates.json").read_text(encoding="utf-8"))
    assert gates["provenance"]["as_of_page_body"][0]["scope"] == "page_body"


def test_page_gate_stamps_are_invisible_to_the_date_scan(tmp_path: Path) -> None:
    """``_strip_provenance`` drops any ``.json`` document's provenance block.

    ``page_gates.json`` is provenance and nothing else, so after stripping there
    is no text left. Without this the stamp's own ``T_cut`` would be counted as a
    post-cutoff date on every page — the audit script's own known false positive.
    """
    out = _export_with_notebook(tmp_path, TRACE)
    doc = json.loads((out / "t-binary" / "notebook" / "page_gates.json").read_text(encoding="utf-8"))
    assert set(doc) == {"provenance"}


def test_a_rejected_tool_call_is_not_mirrored(tmp_path: Path) -> None:
    """It returned nothing to the model, so it is not evidence the screen saw."""
    out = _export_with_notebook(tmp_path, TRACE)
    assert len(list((out / "t-binary" / "notebook").glob("*.search.json"))) == 1


def test_scraped_text_is_written_as_a_page_artifact(tmp_path: Path) -> None:
    out = _export_with_notebook(tmp_path, TRACE)
    pages = sorted((out / "t-binary" / "notebook").glob("*.page.md"))
    assert len(pages) == 1
    assert "The committee held rates." in pages[0].read_text(encoding="utf-8")


def test_an_unstamped_live_result_is_counted_and_not_mirrored(tmp_path: Path) -> None:
    """The one check that cannot be waived: retrieval reaching the model unscreened."""
    trace = json.loads(json.dumps(TRACE))
    trace["tool_calls"][0]["metadata"] = {}
    run_dir, data_path = _write_run(tmp_path, RESULTS)
    _write_trace(run_dir, trace)
    summary = export_run(
        run_dir=run_dir,
        data_path=data_path,
        out_dir=tmp_path / "out",
        temporal_policy="strict",
        delta_days=7,
        observation_time="2026-09-01",
        cutoff_override=None,
        search_provider="serper",
        max_tool_calls_configured=50,
        emit_notebook=True,
    )
    assert summary["notebook"]["n_live_results_without_as_of_stamp"] == 1
    assert summary["notebook"]["n_search_packets"] == 0


def test_an_overlong_task_id_is_truncated_the_way_galaxy_truncates_it(tmp_path: Path) -> None:
    """255 bytes, so both arms' leakage scans cover the same task set."""
    from recipe.web_search.runners.export_galaxy_run_dir import _task_dir_name

    short = "t-binary"
    assert _task_dir_name(short) == short
    long_id = "F" + "%7B%22a%22%3A%22b%22%2C" * 30
    assert len(long_id.encode("utf-8")) > 255
    assert len(_task_dir_name(long_id).encode("utf-8")) <= 255
    assert long_id.startswith(_task_dir_name(long_id))
