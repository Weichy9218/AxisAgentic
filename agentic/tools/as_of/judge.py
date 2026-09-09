# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Gate 3: a small model, asked a different question than the rules ask.

Ported from the Milkyway ``galaxy`` runtime. Every deterministic rule in this
system asks *does this text contain a date on or after ``T_cut``*. That is a
proxy for the thing anyone cares about, and it is a bad proxy in both directions.
Measured upstream on one 1,966-task run, of the 1,459 evidence strings that still
carried a late date after every rule had run, 1,173 were a single bare date with
no metadata key and no adjacent quantity — deliberately spared, because the rules
that would have caught them also caught page footers, and turning those rules up
was separately measured to move directional accuracy from 0.933 to 0.667. In the
other direction the proxy is blind by construction: a page published after
``T_cut`` that states the outcome in prose and prints no date at all matches
nothing, and cannot be counted.

So this gate asks:

    What is the earliest date on which the strongest *settled* factual claim in
    this passage could have been publicly known?

Four properties follow from asking it that way rather than asking "does this leak".

*It separates the two classes the regex conflates.* A footer reading "Updated
2026-09-01" above a 2024 revenue figure answers when that figure was published —
the claim is old, the render is new.

*"Settled" is load-bearing, and it was added after measurement.* An earlier
prompt asked for the date a forward-looking passage "was said". The model has no
way to know that, so it answered with its own idea of today, which is on or after
every benchmark ``T_cut``, so every undated schedule, deadline and analyst
opinion blocked.

*The decision boundary stays in code.* The model returns a date;
:mod:`agentic.tools.as_of.screen` compares it to ``T_cut``. The model never sees
``T_cut``, never sees the question, and never decides policy.

*The answer is cacheable forever.* It is a property of the passage, not of the
run asking, so one verdict serves every task and every future run that meets that
text again.

**Isolation.** This module returns dates and nothing else. There is no code path
by which a string the judge wrote can reach the forecasting model's context,
because the judge does not return strings — which matters, since the judge's own
training data postdates every ``T_cut`` in the benchmark and it therefore knows
many of the answers. It is allowed to know them. It is not allowed to speak.

**Failure is closed.** Under a cutoff, a judge that cannot be reached blocks every
unit routed to it. A network blip costing a search its evidence is a bad outcome;
a network blip silently converting a time-truncated arm into an open-book one,
while the run still reports itself as truncated, is the worst possible one — it
produces a number that looks valid.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from agentic.tools.as_of.cache import CachedVerdict
from agentic.tools.as_of.units import EvidenceUnit

logger = logging.getLogger(__name__)

__all__ = ["UNDATABLE", "JudgeUnavailable", "judge_units"]


class JudgeUnavailable(RuntimeError):
    """The judge could not return a usable verdict for a batch."""


#: Units per request. Held at 8 rather than 12 after a measured misalignment at
#: 12 — see :func:`_parse_verdicts`. The tradeoff is one extra request per ~8
#: units against a whole batch being retried and split.
_MAX_UNITS_PER_CALL: Final = 8

#: Per-unit budget for what the judge is shown. The unit is kept or blocked
#: whole — this only bounds what the judgement is made on, and a claim that is
#: not in the first 1,200 characters of a passage is not that passage's strongest
#: claim.
_MAX_CHARS_PER_UNIT: Final = 1200

_SYSTEM_PROMPT = """\
You date evidence passages. For each numbered passage, answer ONE question: when \
could its strongest SETTLED claim first have been publicly known?

A settled claim reports something that has already happened: a final result, a \
closing or released value, a completed event, an announced decision, a published \
figure. A passage that only schedules, forecasts, predicts, speculates, asks or \
describes settles nothing.

Answer with exactly one of three values:

1. "YYYY-MM-DD" — the passage settles something and you can date it.
2. null — the passage settles nothing. Schedules, deadlines, forecasts, \
predictions, analyst opinions, market questions, navigation text, boilerplate, \
definitions, methodology. These state no fact anyone had to wait for, so they \
have no knowable-from date. Do NOT date them by guessing when someone said them.
3. "unknown" — the passage settles something but gives you no way to date it \
("the merger has now closed", "the winner was sworn in earlier this week").

Dating rules:
- A claim about a period or an event is NEVER knowable before that period or \
event ends. "June 2026 CPI came in at 2.8%" is not knowable before 2026-06-30; \
"full-year 2023 revenue was $3.1bn" is not knowable before 2023-12-31.
- Ignore when the page was written, rendered, updated, crawled or accessed. A \
line reading "Last updated 2026-01-04" above a revised Q3 2026 figure does not \
make that figure knowable in January — the claim is about Q3, so answer Q3.
- Report the earliest date the claim could have been known, not the latest.
- If only a month or a year is determinable, report its LAST day.

Output a JSON array, one object per passage, in input order, and nothing else:
[{"i":0,"knowable_from":"2026-04-15"},{"i":1,"knowable_from":null},\
{"i":2,"knowable_from":"unknown"}]"""

_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Third answer shape: the passage settles something and the judge cannot date
#: it. Deliberately not folded into ``null``, and deliberately not a guess. A
#: settled claim that cannot be placed in time cannot be shown to predate the
#: boundary, so :func:`~agentic.tools.as_of.screen._blocks` blocks it — while
#: ``null`` (nothing settled) passes. Collapsing the two is what makes a judge
#: either leak undated outcomes or starve every forward-looking page.
UNDATABLE: Final = "unknown"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _request_timeout_seconds() -> float:
    try:
        return float(_env("AS_OF_JUDGE_TIMEOUT_SECONDS", "40") or "40")
    except ValueError:
        return 40.0


def _render_batch(units: Sequence[EvidenceUnit]) -> str:
    blocks = []
    for index, unit in enumerate(units):
        body = " ".join(unit.text.split())[:_MAX_CHARS_PER_UNIT]
        blocks.append(f"[{index}]\n{body}")
    return "\n\n".join(blocks)


def _parse_verdicts(content: str, expected: int) -> dict[int, str | None]:
    """Parse one judge response, rejecting anything that is not exactly aligned.

    Alignment is checked rather than trusted, because the failure it catches is
    silent and asymmetric. Measured upstream on a 12-unit batch: the model
    returned verdicts shifted by one, which put a settled 2004 date on a
    methodology note and left the passage that owned it unanswered — one wrong
    verdict and one piece of evidence blocked for a reason unrelated to its
    content. A batch that does not come back one-for-one, in order, is a batch
    the judge did not answer; the caller retries it and then splits it.
    """
    match = _ARRAY_RE.search(content or "")
    if match is None:
        msg = "judge returned no JSON array"
        raise JudgeUnavailable(msg)
    try:
        items = json.loads(match.group(0))
    except ValueError as exc:
        msg = f"judge returned unparseable JSON: {exc}"
        raise JudgeUnavailable(msg) from exc
    if not isinstance(items, list):
        msg = "judge returned a non-array"
        raise JudgeUnavailable(msg)
    if len(items) != expected:
        msg = f"judge returned {len(items)} verdicts for {expected} passages"
        raise JudgeUnavailable(msg)

    out: dict[int, str | None] = {}
    for position, item in enumerate(items):
        if not isinstance(item, Mapping):
            msg = "judge returned a non-object verdict"
            raise JudgeUnavailable(msg)
        try:
            index = int(item["i"])
        except (KeyError, TypeError, ValueError) as exc:
            msg = "judge verdict carries no usable index"
            raise JudgeUnavailable(msg) from exc
        if index != position:
            msg = f"judge verdict {position} is labelled {index}: batch is misaligned"
            raise JudgeUnavailable(msg)
        raw = item.get("knowable_from")
        if raw is None:
            out[index] = None
            continue
        if not isinstance(raw, str):
            msg = f"judge verdict {position} is not a string or null"
            raise JudgeUnavailable(msg)
        text = raw.strip()
        if _ISO_DATE_RE.match(text):
            out[index] = text
        elif text.lower() == UNDATABLE:
            out[index] = UNDATABLE
        else:
            msg = f"judge verdict {position} is neither a date, null nor {UNDATABLE!r}"
            raise JudgeUnavailable(msg)
    return out


_CLIENT: Any = None


def _client() -> Any:
    """One shared async client for the judge route.

    Its own base URL and key, separate from the forecasting model's, so judging
    cannot consume the capacity the rollout needs.
    """
    global _CLIENT  # noqa: PLW0603
    if _CLIENT is None:
        from openai import AsyncOpenAI

        base_url = _env("AS_OF_JUDGE_BASE_URL")
        if not base_url:
            msg = "AS_OF_JUDGE_BASE_URL is not set; the as-of judge has no endpoint."
            raise JudgeUnavailable(msg)
        _CLIENT = AsyncOpenAI(
            api_key=_env("AS_OF_JUDGE_API_KEY") or "dummy_key",
            base_url=base_url,
            timeout=_request_timeout_seconds(),
            max_retries=0,
        )
    return _CLIENT


def reset_client() -> None:
    """Drop the cached client. For tests that repoint the endpoint."""
    global _CLIENT  # noqa: PLW0603
    _CLIENT = None


async def _ask(units: Sequence[EvidenceUnit], *, model: str) -> dict[int, str | None]:
    response = await asyncio.wait_for(
        _client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _render_batch(units)},
            ],
            temperature=0.0,
            max_tokens=64 + 40 * len(units),
        ),
        timeout=_request_timeout_seconds(),
    )
    return _parse_verdicts(response.choices[0].message.content or "", len(units))


async def _judge_one_batch(units: Sequence[EvidenceUnit]) -> dict[str, CachedVerdict]:
    primary = _env("AS_OF_JUDGE_MODEL")
    if not primary:
        msg = "AS_OF_JUDGE_MODEL is not set; the as-of judge has no model."
        raise JudgeUnavailable(msg)

    # A second endpoint, tried once before the batch is split. Its purpose is to
    # keep a single gateway hiccup from blocking a whole search's evidence; it is
    # not a quality fallback and both models answer the same contract. It must
    # also be fast: a fallback slower than the request timeout is not a fallback.
    fallback = _env("AS_OF_JUDGE_FALLBACK_MODEL")
    attempts = [primary] + ([fallback] if fallback and fallback != primary else [])

    last_error: Exception | None = None
    for model in attempts:
        try:
            verdicts = await _ask(units, model=model)
        except Exception as exc:  # noqa: BLE001 - any failure moves to the next attempt
            last_error = exc
            continue
        return {units[index].digest: CachedVerdict(verdicts[index], model) for index in verdicts}

    # Both endpoints refused this batch. Halve it and try again: the common cause
    # is a batch the model could not keep aligned, and a batch of one cannot be
    # misaligned. Failure blocks evidence, so it is worth two more requests to
    # find out whether the batch was the problem.
    if len(units) > 1:
        midpoint = len(units) // 2
        halves = await asyncio.gather(
            _judge_one_batch(units[:midpoint]),
            _judge_one_batch(units[midpoint:]),
            return_exceptions=True,
        )
        merged: dict[str, CachedVerdict] = {}
        for half in halves:
            if isinstance(half, dict):
                merged.update(half)
        if merged:
            return merged

    msg = f"judge unreachable: {last_error}"
    raise JudgeUnavailable(msg)


async def judge_units(units: Sequence[EvidenceUnit]) -> dict[str, CachedVerdict]:
    """Date every unit, keyed by content digest.

    A digest missing from the result is a unit the judge did not answer for. The
    caller must treat that as a block, not a pass — see the module docstring on
    failing closed.
    """
    if not units:
        return {}
    batches = [units[start : start + _MAX_UNITS_PER_CALL] for start in range(0, len(units), _MAX_UNITS_PER_CALL)]
    results = await asyncio.gather(*(_judge_one_batch(batch) for batch in batches), return_exceptions=True)
    merged: dict[str, CachedVerdict] = {}
    failures: list[BaseException] = []
    for result in results:
        if isinstance(result, BaseException):
            failures.append(result)
            continue
        merged.update(result)
    if failures and not merged:
        msg = f"judge unreachable: {failures[0]}"
        raise JudgeUnavailable(msg)
    return merged
