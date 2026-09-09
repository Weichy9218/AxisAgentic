# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Executable spec for per-task eval-result filenames.

This exists because a benchmark id is whatever the benchmark chose to call the
row, and one FutureX row's id is 190 characters of percent-encoded JSON. Writing
it straight to disk raised ``OSError: [Errno 36] File name too long`` and killed
four shards of a 1969-task run at once, roughly 32% in.
"""

from __future__ import annotations

from pathlib import Path

from agentic.evaluation.evaluator import BatchEvaluator, _safe_filename_stem

LONG_ID = (
    "FutureX_Wikimedia_article_pageviews_daily__wikimedia_articles_v1__"
    "%7B%22access%22%3A%22all-access%22%2C%22agent%22%3A%22user%22%2C"
    "%22canonical_title%22%3A%22%E5%94%90%E6%9C%9D%22%2C%22project%22%3A"
    "%22zh.wikipedia%22%2C%22utc_date%22%3A%222026-08-20%22%7D"
)


def test_an_ordinary_id_is_left_alone() -> None:
    """The common case must stay readable and greppable."""
    assert _safe_filename_stem("90tcyC2Pql") == "90tcyC2Pql"
    assert _safe_filename_stem("20260629110709444496") == "20260629110709444496"


def test_a_long_id_fits_the_filesystem_limit() -> None:
    stem = _safe_filename_stem(LONG_ID)
    # 255 bytes is the per-component cap; leave room for ".json".
    assert len(stem.encode("utf-8")) <= 200


def test_shortening_does_not_collide() -> None:
    """Truncation alone would map two ids sharing a prefix onto one file."""
    a = _safe_filename_stem(LONG_ID)
    b = _safe_filename_stem(LONG_ID + "_variant")
    assert a != b


def test_characters_a_path_cannot_hold_are_replaced() -> None:
    stem = _safe_filename_stem("a/b\\c:d*e?f")
    assert "/" not in stem
    assert "\\" not in stem


def test_the_full_id_survives_inside_the_file(tmp_path: Path) -> None:
    """The name is an index; the record is the data, and it keeps the real id."""
    evaluator = BatchEvaluator(eval_dir=tmp_path)
    path = evaluator.log_result(LONG_ID, eval_name="forecast_submission", score=1.0, prediction="YES")
    assert path is not None
    assert path.exists()
    import json

    assert json.loads(path.read_text(encoding="utf-8"))["task_id"] == LONG_ID


def test_two_long_ids_write_two_files(tmp_path: Path) -> None:
    evaluator = BatchEvaluator(eval_dir=tmp_path)
    first = evaluator.log_result(LONG_ID, eval_name="e", score=1.0)
    second = evaluator.log_result(LONG_ID + "_variant", eval_name="e", score=0.0)
    assert first != second
    assert len(list((tmp_path / "e").glob("*.json"))) == 2
