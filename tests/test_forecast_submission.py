# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Executable spec for reading a forecast submission out of a final message.

Every rejection here must carry a reason from the scorer's own vocabulary, so
this repository's excluded counts line up with the Milkyway ones task for task.
"""

from __future__ import annotations

import pytest

from recipe.web_search.eval.forecast_submission import (
    normalize_boxed_answer,
    parse_forecast_submission,
)


def test_binary_answer_with_confidence() -> None:
    result = parse_forecast_submission(
        "After weighing the reporting, the market looks likely to resolve up.\n"
        "\\boxed{New Jersey 5s}\n"
        "Confidence: 0.64\n",
        task_type="BINARY",
    )
    assert result.ok
    assert result.output == "New Jersey 5s"
    assert result.probability == pytest.approx(0.64)


def test_percent_suffix_is_the_one_accepted_conversion() -> None:
    """``64%`` is unambiguous. A bare ``64`` is not, and is refused below."""
    result = parse_forecast_submission("\\boxed{YES}\nConfidence: 64%", task_type="BINARY")
    assert result.ok
    assert result.probability == pytest.approx(0.64)


def test_a_bare_out_of_range_confidence_is_refused_not_rescaled() -> None:
    """Reading ``64`` as 0.64 is a guess. Guessing is what the validity principle forbids."""
    result = parse_forecast_submission("\\boxed{YES}\nConfidence: 64", task_type="BINARY")
    assert not result.ok
    assert result.reason == "probability_out_of_range"


def test_the_last_confidence_line_wins() -> None:
    """A model that reasons about confidence and then states it means the final one."""
    result = parse_forecast_submission(
        "Early on I would have said Confidence: 0.30\nNow:\n\\boxed{NO}\nConfidence: 0.82",
        task_type="BINARY",
    )
    assert result.probability == pytest.approx(0.82)


@pytest.mark.parametrize(
    "line",
    ["Confidence: 0.55", "**Confidence:** 0.55", "**Confidence**: 0.55", "  Confidence : 0.55", "> Confidence: 0.55"],
)
def test_markdown_emphasis_around_the_label_still_parses(line: str) -> None:
    result = parse_forecast_submission("\\boxed{NO}\n" + line, task_type="BINARY")
    assert result.ok, result.detail
    assert result.probability == pytest.approx(0.55)


def test_probability_zero_is_refused() -> None:
    """The bound is open: a probability of 0 says the submitted answer cannot be right."""
    result = parse_forecast_submission("\\boxed{YES}\nConfidence: 0", task_type="BINARY")
    assert result.reason == "probability_out_of_range"


def test_probability_one_is_allowed() -> None:
    result = parse_forecast_submission("\\boxed{YES}\nConfidence: 1.0", task_type="BINARY")
    assert result.ok
    assert result.probability == pytest.approx(1.0)


def test_choice_without_a_confidence_line_is_excluded_not_defaulted() -> None:
    result = parse_forecast_submission("\\boxed{Decrease}", task_type="MULTIPLE_CHOICE")
    assert not result.ok
    assert result.reason == "probability_missing"
    assert result.output == ""


def test_numeric_carries_no_probability_even_when_one_is_stated() -> None:
    """The contract has no correctness event for an open numeric target."""
    result = parse_forecast_submission("\\boxed{845.47}\nConfidence: 0.7", task_type="NUMERIC")
    assert result.ok
    assert result.output == "845.47"
    assert result.probability is None


@pytest.mark.parametrize("answer", ["857.64", "-12", "0.5", "+3.25", "1234567"])
def test_numeric_accepts_bare_numbers(answer: str) -> None:
    result = parse_forecast_submission("\\boxed{" + answer + "}", task_type="NUMERIC")
    assert result.ok
    assert result.output == answer


@pytest.mark.parametrize("answer", ["845.47 tons", "$1,200", "1.2e5", "about 900", "12%"])
def test_numeric_refuses_anything_the_scorer_might_not_read(answer: str) -> None:
    """This check is stricter than the scorer's reader, so what passes here passes there."""
    result = parse_forecast_submission("\\boxed{" + answer + "}", task_type="NUMERIC")
    assert not result.ok
    assert result.reason == "prediction_not_a_number"


def test_missing_boxed_payload_is_no_answer() -> None:
    result = parse_forecast_submission("I think it resolves NO.\nConfidence: 0.6", task_type="BINARY")
    assert result.reason == "submission_has_no_answer"


@pytest.mark.parametrize("payload", ["?", "...", "unknown"])
def test_blacklisted_non_answers_are_not_recorded_as_forecasts(payload: str) -> None:
    result = parse_forecast_submission("\\boxed{" + payload + "}\nConfidence: 0.5", task_type="BINARY")
    assert result.reason == "submission_has_no_answer"


def test_a_refusal_inside_the_box_fails_rather_than_scoring() -> None:
    """A refusal recorded as a forecast contaminates scoring more quietly than a gap."""
    result = parse_forecast_submission(
        "\\boxed{I cannot predict the future}\nConfidence: 0.5", task_type="BINARY"
    )
    assert result.reason == "no_valid_submission"
    assert "refusal" in result.detail


def test_a_probability_appended_to_the_answer_fails_rather_than_being_stripped() -> None:
    """"YES, 58%" matches neither legal option, so an otherwise-correct forecast would score wrong."""
    result = parse_forecast_submission("\\boxed{YES, 58%}\nConfidence: 0.58", task_type="BINARY")
    assert result.reason == "no_valid_submission"
    assert "probability appended" in result.detail


def test_a_nested_latex_wrapper_fails() -> None:
    result = parse_forecast_submission("\\boxed{\\boxed{YES}}\nConfidence: 0.6", task_type="BINARY")
    assert result.reason == "no_valid_submission"


def test_numeric_path_does_not_apply_the_trailing_number_check() -> None:
    """A bare number *is* the whole answer here; the inlined-probability rule would eat it."""
    result = parse_forecast_submission("\\boxed{0.75}", task_type="NUMERIC")
    assert result.ok
    assert result.output == "0.75"


def test_fill_in_is_graded_without_a_probability() -> None:
    result = parse_forecast_submission("\\boxed{a | b | c}", task_type="FILL_IN")
    assert result.ok
    assert result.output == "a | b | c"
    assert result.probability is None


# --- LaTeX typesetting inside the box ----------------------------------------
#
# `\boxed{}` is LaTeX, so the model can spell a correct option with a spacing
# macro. Observed on offline_0901: `\boxed{BK\ Hacken}` against the option
# `BK Hacken`, a right answer the scorer excluded as `output_not_in_options`.


def test_a_spacing_macro_does_not_cost_a_correct_option() -> None:
    result = parse_forecast_submission(
        "\\boxed{BK\\ Hacken}\nConfidence: 0.96", task_type="MULTIPLE_CHOICE"
    )
    assert result.ok
    assert result.output == "BK Hacken"


def test_a_text_wrapper_is_unwrapped_including_when_nested() -> None:
    assert normalize_boxed_answer("\\text{\\text{No change}}") == "No change"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("Estudiantes\\ de\\ La\\ Plata", "Estudiantes de La Plata"),
        ("a\\;b", "a;b"),
        ("1\\,000", "1,000"),
    ],
)
def test_the_three_spacing_macros_expand(payload: str, expected: str) -> None:
    assert normalize_boxed_answer(payload) == expected


@pytest.mark.parametrize("payload", ["120-139", "1. FC Kaiserslautern", "13.5 - 14M", "0.2-0.3%"])
def test_normalisation_removes_typesetting_and_nothing_else(payload: str) -> None:
    """The guard against adopting galaxy's `_cleanup_prediction_candidate`.

    That function strips enumeration prefixes and re-joins `;`-separated lists,
    which would rewrite bucket labels (`120-139` -> `139`) and a club whose name
    begins with `1.`. It has no caller on galaxy's scoring path; copying it
    would import a defect rather than align with anything.
    """
    assert normalize_boxed_answer(payload) == payload


def test_normalisation_is_idempotent() -> None:
    once = normalize_boxed_answer("\\text{BK\\ Hacken}")
    assert normalize_boxed_answer(once) == once == "BK Hacken"


def test_a_bare_backslash_is_left_alone() -> None:
    """In an answer string a lone backslash is more likely data than typesetting."""
    assert normalize_boxed_answer("AC\\DC") == "AC\\DC"


def test_a_leftover_boxed_wrapper_is_still_a_contract_violation() -> None:
    """Normalisation must not launder the one wrapper that means the format failed."""
    result = parse_forecast_submission("\\boxed{\\text{\\boxed{YES}}}\nConfidence: 0.6", task_type="BINARY")
    assert result.reason == "no_valid_submission"
