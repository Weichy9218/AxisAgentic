# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Parse a forecast submission out of the agent's final message.

AxisAgentic's orchestrator is built around ``\\boxed{}``: it drives retries off
:data:`~recipe.web_search.agent.prompts.FORMAT_ERROR_MESSAGE` and extracts the
final output with ``extract_boxed_content``. The scoring contract on the other
side wants a bare answer plus a probability. This module is the join: it reads
the boxed payload and the ``Confidence:`` line the forecast prompt asks for, and
applies the same three output checks the Milkyway scorer's own submission
contract applies (``galaxy/forecast/agent/forecast_output.py``), so both systems
reject the same strings rather than each accepting what the other refuses.

**Nothing is fabricated.** A submission that cannot be read yields a rejection
carrying one of the scorer's own exclusion reasons; it never yields a default
answer or a filled-in probability. That is the validity principle in
``docs/SCORING_SPEC.md``: a missing submission must reduce ``n_scored``, not
quietly become a wrong one.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final

from recipe.web_search.agent.prompts import extract_boxed_content

__all__ = [
    "DISCRETE_PROBABILITY_TASK_TYPES",
    "NUMERIC_TASK_TYPE",
    "ForecastSubmission",
    "normalize_boxed_answer",
    "parse_forecast_submission",
]

#: The scorer defines no probability event for an open numeric target, and its
#: submission contract requires ``probability`` to be null on this path.
NUMERIC_TASK_TYPE: Final = "NUMERIC"

#: Task types whose correctness is a decidable event, so a probability is required.
#: ``FILL_IN`` is deliberately absent: the scorer grades it by position overlap and
#: never reads a probability for it.
DISCRETE_PROBABILITY_TASK_TYPES: Final = frozenset({"BINARY", "MULTIPLE_CHOICE"})

# --- output contract checks, mirrored from the Milkyway submission schema -----

# ``\boxed{}`` is LaTeX, so its payload can carry LaTeX typesetting that is not
# part of the answer. Observed on this benchmark: ``\boxed{BK\ Hacken}`` for the
# option ``BK Hacken`` — the right club, a spacing macro away from matching. The
# scorer compares the submission against the question's options as strings, so
# the macro turns a correct answer into ``output_not_in_options``.
#
# Only the escapes whose expansion is unambiguous are undone: ``\text{}`` (a
# wrapper, never content) and the three spacing macros. A bare backslash is left
# alone, because in an answer string it is more likely data than typesetting;
# galaxy's own ``_cleanup_prediction_candidate`` strips those too, along with
# enumeration prefixes, but that function has no caller on its scoring path and
# would rewrite 90 of this run's bucket labels (``120-139`` -> ``139``) and the
# club name ``1. FC Kaiserslautern``. Copying it would import a defect, not
# alignment.
_LATEX_TEXT_RE = re.compile(r"\\text\s*\{([^{}]*)\}")
_LATEX_SPACING = (("\\;", ";"), ("\\,", ","), ("\\ ", " "))

_BOXED_SHELL_RE = re.compile(r"\\{0,2}boxed\s*\{", re.IGNORECASE)

# A probability restated inside the answer. Measured upstream against a live
# gateway: asked a YES/NO question, the model submitted ``"YES, 58%"`` alongside
# a correct probability field. For a question whose legal answers are YES / NO,
# "YES, 58%" matches neither, so an otherwise-correct forecast scores wrong.
# Stripping it silently would hide a systematic prompt problem behind a repair.
_INLINED_PROBABILITY_RE = re.compile(
    r"[,;:]\s*(?:~|about\s+|approximately\s+|p\s*=\s*|prob(?:ability)?\s*[:=]?\s*)?"
    r"(?:0?\.\d+|\d{1,3}\s*%|1\.0+)\s*$",
    re.IGNORECASE,
)

# A refusal contaminates scoring more quietly than a missing answer does, because
# it records as a submission. It fails here instead.
_REFUSAL_RE = re.compile(
    r"\b("
    r"no\s+(?:valid\s+)?forecast(?:\s+produced)?"
    r"|cannot\s+(?:produce\s+a\s+|provide\s+a\s+)?forecast"
    r"|unable\s+to\s+(?:produce\s+a\s+|provide\s+a\s+)?forecast"
    r"|cannot\s+(?:answer|determine|predict)"
    r"|insufficient\s+(?:evidence|information|data)\s+to\s+(?:answer|forecast|predict)"
    r"|i\s+(?:cannot|can\s*not|am\s+unable)"
    r")\b",
    re.IGNORECASE,
)

# ``Confidence: 0.64`` on its own line, optionally with a percent sign. The last
# match wins: a model that reasons about its confidence mid-message and then
# states it at the end means the final one. The character classes accept the
# Markdown emphasis models wrap the label in, on either side of the colon
# (``**Confidence:** 0.55`` and ``**Confidence**: 0.55``), and exclude newlines
# so a multiline match cannot span two lines.
_CONFIDENCE_RE = re.compile(r"(?im)^[ \t*_#>]*confidence[ \t*_]*[:：][ \t*_]*([0-9]*\.?[0-9]+)[ \t]*(%)?[ \t]*$")

# A bare number, which is all the numeric path may submit. Deliberately stricter
# than the scorer's unit-aware reader, so anything accepted here is also
# accepted there.
_BARE_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")

#: Exclusion reasons, using the scorer's vocabulary so the two systems' excluded
#: counts are directly comparable.
_REASON_NO_ANSWER: Final = "submission_has_no_answer"
_REASON_NO_PROBABILITY: Final = "probability_missing"
_REASON_PROBABILITY_RANGE: Final = "probability_out_of_range"
_REASON_NOT_A_NUMBER: Final = "prediction_not_a_number"
_REASON_INVALID: Final = "no_valid_submission"


@dataclass(frozen=True)
class ForecastSubmission:
    """A readable submission, or a rejection carrying the scorer's own reason.

    ``output``/``probability`` are populated only when :attr:`ok`. ``raw_output``
    and ``raw_probability`` carry whatever was found regardless, because the
    exporter forwards them for the rejections the scorer can classify better
    than this parser can: it, not this module, owns the exclusion vocabulary, and
    forwarding lets it say ``prediction_not_a_number`` or
    ``probability_out_of_range`` in its own words rather than having both sides
    guess separately. The one exception is a contract violation
    (``no_valid_submission``), where the answer itself is the thing being
    refused and must not be forwarded.
    """

    output: str = ""
    probability: float | None = None
    #: ``None`` when the submission is usable. Otherwise one of the scorer's
    #: exclusion reasons.
    reason: str | None = None
    #: Why, in this repository's own words. Diagnostics only; the scorer never
    #: sees it, and it must not be substituted for ``reason``.
    detail: str = ""
    #: The boxed payload as found, kept even on rejection.
    raw_output: str = ""
    #: The Confidence value as found, kept even when out of range.
    raw_probability: float | None = None
    #: Whether the answer itself failed the output contract, in which case it is
    #: withheld from the export rather than forwarded.
    contract_violation: bool = False

    @property
    def ok(self) -> bool:
        return self.reason is None


def _reject(
    reason: str,
    detail: str,
    *,
    raw_output: str = "",
    raw_probability: float | None = None,
    contract_violation: bool = False,
) -> ForecastSubmission:
    return ForecastSubmission(
        reason=reason,
        detail=detail,
        raw_output=raw_output,
        raw_probability=raw_probability,
        contract_violation=contract_violation,
    )


def normalize_boxed_answer(text: str) -> str:
    """Undo the LaTeX typesetting a ``\\boxed{}`` payload may carry.

    Unwraps ``\\text{...}`` (repeatedly, for nested wrappers) and expands the
    three spacing macros. Everything else is left verbatim — including bare
    backslashes and braces — so the function can only ever remove typesetting,
    never content. Idempotent: applying it twice changes nothing.

    Applied both when a submission is first read and again at export time, since
    a run finished before this existed still has the macros in its results.
    """
    value = str(text or "")
    previous = None
    while previous != value:
        previous = value
        value = _LATEX_TEXT_RE.sub(r"\1", value)
    for macro, expansion in _LATEX_SPACING:
        value = value.replace(macro, expansion)
    return value.strip()


def _read_confidence(text: str) -> tuple[float | None, str]:
    """Return ``(probability, detail)``; ``probability`` is ``None`` when unstated.

    A percent sign is the one unit conversion accepted, because it is
    unambiguous. A bare ``64`` is not converted: reading it as 0.64 is a guess
    about what the model meant, and guessing is what the validity principle
    forbids. It is reported out of range instead, which is visible in the
    excluded counts rather than silently shifting a score.
    """
    matches = list(_CONFIDENCE_RE.finditer(text))
    if not matches:
        return None, "no Confidence line"
    raw, percent = matches[-1].group(1), matches[-1].group(2)
    try:
        value = float(raw)
    except ValueError:
        return None, f"unparseable Confidence value {raw!r}"
    if percent:
        value /= 100.0
    return value, ""


def parse_forecast_submission(
    final_text: str,
    *,
    task_type: str,
    extracted_answer: str | None = None,
) -> ForecastSubmission:
    """Read ``(output, probability)`` out of the agent's final message.

    Args:
        final_text: The agent's last assistant message, verbatim. The
            ``Confidence:`` line lives here and nowhere else, so passing an
            already-unwrapped answer instead loses the probability.
        task_type: The row's declared type. Routes the probability requirement;
            the model is never told this label, it follows the question wording.
        extracted_answer: The answer the orchestrator already pulled out of the
            box, used only when *final_text* carries no ``\\boxed{}`` payload.
            Resuming from a persisted trace can reach here with the answer
            recovered and the surrounding message not.
    """
    text = final_text or ""
    normalized_type = str(task_type or "").strip().upper()

    output = extract_boxed_content(text).strip()
    if not output and extracted_answer:
        # Only the wrapper is missing; the probability, if stated, is still in
        # ``text`` and is read below from the same string.
        output = str(extracted_answer).strip()
    output = normalize_boxed_answer(output)
    if not output:
        # ``extract_boxed_content`` also returns "" for its blacklist of
        # non-answers ("?", "...", "unknown"), which belong here rather than
        # being recorded as a forecast.
        return _reject(_REASON_NO_ANSWER, "no usable \\boxed{} payload in the final message")

    if _BOXED_SHELL_RE.search(output):
        return _reject(
            _REASON_INVALID,
            f"answer still carries a LaTeX wrapper: {output[:120]!r}",
            raw_output=output,
            contract_violation=True,
        )
    if _REFUSAL_RE.search(output):
        return _reject(
            _REASON_INVALID,
            f"answer looks like a refusal: {output[:120]!r}",
            raw_output=output,
            contract_violation=True,
        )

    if normalized_type == NUMERIC_TASK_TYPE:
        # A bare number is the whole answer here, so the trailing-number check
        # would reject every valid submission.
        if not _BARE_NUMBER_RE.match(output):
            return _reject(
                _REASON_NOT_A_NUMBER,
                f"numeric answer is not digits only: {output[:120]!r}",
                raw_output=output,
            )
        value = float(output)
        if not math.isfinite(value):
            return _reject(
                _REASON_NOT_A_NUMBER,
                f"numeric answer is not finite: {output[:120]!r}",
                raw_output=output,
            )
        # The scorer requires null here; a stated confidence is discarded rather
        # than carried, because the contract has no event for it to describe.
        return ForecastSubmission(output=output, probability=None, raw_output=output)

    if _INLINED_PROBABILITY_RE.search(output):
        return _reject(
            _REASON_INVALID,
            f"answer has the probability appended to it: {output[:120]!r}",
            raw_output=output,
            contract_violation=True,
        )

    probability, detail = _read_confidence(text)
    if normalized_type not in DISCRETE_PROBABILITY_TASK_TYPES:
        # FILL_IN and anything else the scorer grades without a probability.
        return ForecastSubmission(output=output, probability=None, raw_output=output)
    if probability is None:
        return _reject(_REASON_NO_PROBABILITY, detail, raw_output=output)
    if not math.isfinite(probability) or not (0.0 < probability <= 1.0):
        return _reject(
            _REASON_PROBABILITY_RANGE,
            f"Confidence {probability!r} is outside (0, 1]",
            raw_output=output,
            raw_probability=probability,
        )
    return ForecastSubmission(
        output=output,
        probability=probability,
        raw_output=output,
        raw_probability=probability,
    )
