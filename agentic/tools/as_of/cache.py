# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed store for Gate 3 verdicts.

Ported from the Milkyway ``galaxy`` runtime. The reuse win comes from one
property of the judge's contract: it is asked *when did this passage become
knowable*, not *does this passage leak given T_cut*. The first question has an
answer that belongs to the passage; the second has an answer that belongs to the
pair. So a verdict keys on the text alone and is valid for every task and every
future run that sees that text again, whereas a ``T_cut``-keyed cache would hit
only inside a single task, since ``T_cut = end_time - delta_days`` differs per
question.

That is also why the comparison against ``T_cut`` lives in
:mod:`agentic.tools.as_of.screen` and not in the model's output. The decision
boundary stays in code, where it can be audited and changed without re-spending
a single token.

On disk rather than in memory because a benchmark re-runs the same corpus many
times and tasks run concurrently. Writes are atomic (temp file + ``os.replace``)
so concurrent writers cannot tear an entry; reads treat any damaged entry as a
miss.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Final

__all__ = ["CACHE_SCHEMA_VERSION", "CachedVerdict", "VerdictCache", "get_verdict_cache"]

#: Bump when the judge prompt or its output contract changes in a way that makes
#: stored verdicts wrong. Entries from an older version are ignored, not deleted:
#: a concurrent reader on the old layout should miss, not crash.
CACHE_SCHEMA_VERSION: Final = 1

_MEMORY_ENTRIES: Final = 8192


class CachedVerdict:
    """One stored answer to "when did this passage become knowable"."""

    __slots__ = ("knowable_from", "model")

    def __init__(self, knowable_from: str | None, model: str) -> None:
        self.knowable_from = knowable_from
        self.model = model


def _default_cache_root() -> Path:
    override = os.environ.get("AS_OF_VERDICT_CACHE_DIR", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / ".cache" / "as_of_verdicts"


class VerdictCache:
    """Two tiers: a per-process LRU over an on-disk content-addressed store."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root is not None else _default_cache_root()
        self._memory: OrderedDict[str, CachedVerdict] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _path_for(self, digest: str) -> Path:
        return self._root / digest[:2] / f"{digest}.json"

    def get(self, digest: str) -> CachedVerdict | None:
        hit = self._memory.get(digest)
        if hit is not None:
            self._memory.move_to_end(digest)
            self.hits += 1
            return hit
        try:
            record = json.loads(self._path_for(digest).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.misses += 1
            return None
        if not isinstance(record, dict) or record.get("v") != CACHE_SCHEMA_VERSION:
            self.misses += 1
            return None
        knowable = record.get("knowable_from")
        if knowable is not None and not isinstance(knowable, str):
            self.misses += 1
            return None
        verdict = CachedVerdict(knowable, str(record.get("model") or ""))
        self._remember(digest, verdict)
        self.hits += 1
        return verdict

    def put(self, digest: str, verdict: CachedVerdict) -> None:
        self._remember(digest, verdict)
        path = self._path_for(digest)
        record = {
            "v": CACHE_SCHEMA_VERSION,
            "knowable_from": verdict.knowable_from,
            "model": verdict.model,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(path.parent),
                prefix=f".{path.name}.",
                delete=False,
            )
            try:
                json.dump(record, handle, ensure_ascii=False)
                handle.flush()
            finally:
                handle.close()
            os.replace(handle.name, path)  # noqa: PTH105
        except OSError:
            # A cache that cannot write is a cache that misses. It is never a
            # reason to fail a run, and never a reason to skip the judge.
            return

    def _remember(self, digest: str, verdict: CachedVerdict) -> None:
        self._memory[digest] = verdict
        self._memory.move_to_end(digest)
        while len(self._memory) > _MEMORY_ENTRIES:
            self._memory.popitem(last=False)


_CACHE: VerdictCache | None = None


def get_verdict_cache() -> VerdictCache:
    global _CACHE  # noqa: PLW0603
    if _CACHE is None:
        _CACHE = VerdictCache()
    return _CACHE
