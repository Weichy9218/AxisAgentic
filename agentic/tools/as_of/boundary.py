# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""The as-of stamp, and the existence check at the context edge.

Ported from the Milkyway ``galaxy`` runtime. The check here is *existential*, not
semantic: it asks whether the screen ran, against this boundary, on this payload.
Upstream that distinction was bought by measurement — in one run, 23 main-agent
search calls carried a cutoff while 24 sub-agent calls arrived without one, and
every post-cutoff card in that run came from the unarmed calls. No amount of
filter tuning finds that class of bug, because the filter was never invoked.

A blocked payload carries nothing from the original. A blocked page's own dates
are exactly the thing that must not reach context, and an earlier version leaked
them back through the diagnostic field.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Final

from agentic.temporal import parse_cutoff

__all__ = [
    "AS_OF_STAMP_KEY",
    "POST_CUTOFF_GUARDRAIL",
    "AsOfCutoffError",
    "AsOfScreeningError",
    "blocked_payload",
    "resolve_cutoff",
    "stamp",
]

#: What the model is told when a whole document was refused.
POST_CUTOFF_GUARDRAIL: Final = "post_cutoff_source_blocked"

#: Where the stamp lives in a tool result's metadata.
AS_OF_STAMP_KEY: Final = "as_of"


class AsOfCutoffError(ValueError):
    """A configured cutoff could not be parsed. Fatal: a typo must not disable filtering."""


class AsOfScreeningError(RuntimeError):
    """Evidence reached the context edge without having been screened."""


def resolve_cutoff(value: Any) -> date | None:
    """Parse a cutoff, or fail loudly. ``None`` only for a genuinely absent one."""
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    parsed = parse_cutoff(value)
    if parsed is None:
        msg = (
            f"cutoff {value!r} is not an unambiguous calendar day (expected YYYY-MM-DD). "
            "Refusing to continue: an uninterpretable cutoff would silently disable "
            "time truncation while the run still reported itself as truncated."
        )
        raise AsOfCutoffError(msg)
    return parsed


def stamp(counts: Mapping[str, int], *, T_cut: date, scope: str) -> dict[str, Any]:  # noqa: N803
    """Record that the screen ran, against which boundary, and what it decided.

    Emitted even when nothing was blocked: "yes, and it held nothing" is a
    different answer from silence, and only the first one proves the gate was
    armed for this call.
    """
    return {"T_cut": T_cut.isoformat(), "scope": scope, **dict(counts)}


def blocked_payload(*, tool_name: str, T_cut: date) -> dict[str, Any]:  # noqa: N803
    """What replaces a document refused whole. Carries nothing from the original."""
    return {
        "success": False,
        "guardrail": POST_CUTOFF_GUARDRAIL,
        "error": (
            "This source is dated on or after the information cutoff for this task and was "
            "withheld. Look for earlier reporting instead."
        ),
        AS_OF_STAMP_KEY: {"T_cut": T_cut.isoformat(), "scope": f"{tool_name}:blocked"},
    }


def assert_screened(*, tool_name: str, metadata: Mapping[str, Any] | None, T_cut: date | None) -> None:  # noqa: N803
    """Fail when a live-channel result reaches context unscreened.

    Args:
        tool_name: For the error message only.
        metadata: The tool result's metadata, where the stamp lives.
        T_cut: The boundary this call was supposed to enforce. ``None`` means the
            run declared no boundary, and there is nothing to check.

    Raises:
        AsOfScreeningError: no stamp, or a stamp naming a different boundary. A
            result screened for another task must not be replayed here.
    """
    if T_cut is None:
        return
    recorded = (metadata or {}).get(AS_OF_STAMP_KEY)
    if not isinstance(recorded, Mapping):
        msg = (
            f"{tool_name} returned evidence with no as-of stamp while T_cut={T_cut.isoformat()} "
            "was in force. This is an unwired channel, not a filter that let something through."
        )
        raise AsOfScreeningError(msg)
    if recorded.get("T_cut") != T_cut.isoformat():
        msg = (
            f"{tool_name} returned evidence stamped T_cut={recorded.get('T_cut')!r} while "
            f"{T_cut.isoformat()!r} was in force. A result screened for a different task "
            "must not be replayed here."
        )
        raise AsOfScreeningError(msg)
