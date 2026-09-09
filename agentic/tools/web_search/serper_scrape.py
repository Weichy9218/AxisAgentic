# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Provider-side page fetching, for hosts this machine cannot reach itself.

Jina Reader and a direct HTTP GET both fetch *from here*. On an egress-restricted
host that is fatal and unfixable by changing fetch libraries: measured on the
machine this was written for, ``r.jina.ai``, ``reuters.com``, ``en.wikipedia.org``,
``polymarket.com`` and ``manifold.markets`` all answer "Network is unreachable"
while ``google.serper.dev`` connects in under a second.

Serper's ``/scrape`` endpoint fetches from Serper's network and returns the text,
so a page is reachable whenever the search provider is. That is the whole reason
this backend exists; it is not a quality choice over Jina.

What comes back is ``{"text": ..., "metadata": {...}, "jsonld": ..., "credits": n}``.
Two things are worth knowing before relying on it:

* ``metadata`` carries **no publication date** on any page sampled — only
  OpenGraph title/description fields. ``jsonld`` sometimes carries
  ``datePublished``. So a page's own date is usually unavailable, and Gate 1 on
  this channel will mostly return "unknown", which is admitted by design.
* Each call costs 2 credits against the same Serper quota the search tool spends.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Final

import httpx

from agentic.tools.web_search._retry import RetryConfig, TimeoutConfig, coerce_timeout, default_scrape_retry
from agentic.tools.web_search._scrape_utils import empty_result
from agentic.tools.web_search._utils import is_banned_url

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_SERPER_SCRAPE_URL", "extract_serper_page_date", "scrape_with_serper"]

DEFAULT_SERPER_SCRAPE_URL: Final = "https://scrape.serper.dev"

#: JSON-LD keys that name when a document was first published. ``dateModified``
#: is deliberately absent: it is when the page was last re-rendered, which on a
#: live page is today, and treating it as the publication date would block every
#: page a site still maintains.
_JSONLD_DATE_KEYS: Final = ("datePublished", "dateCreated", "uploadDate")

#: Metadata keys some sites do expose, checked as a bonus. None of the sampled
#: pages carried any of these, so this is opportunistic rather than relied upon.
_METADATA_DATE_KEYS: Final = (
    "article:published_time",
    "og:published_time",
    "datePublished",
    "date",
    "pubdate",
    "publish-date",
)


def _walk_jsonld(node: Any) -> Any:
    """Yield every mapping in a JSON-LD document, which may be a graph or a list."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_jsonld(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_jsonld(item)


def extract_serper_page_date(payload: dict[str, Any]) -> str | None:
    """The page's own claimed publication date, or ``None`` when it states none.

    ``None`` is the common case and is not a failure. It feeds Gate 1 as
    "undated", which is admitted and labelled rather than dropped, because
    dropping undated evidence costs a third of all real cards and buys no safety.
    """
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in _METADATA_DATE_KEYS:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    jsonld = payload.get("jsonld")
    if jsonld is not None:
        for node in _walk_jsonld(jsonld):
            for key in _JSONLD_DATE_KEYS:
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


async def scrape_with_serper(
    url: str,
    *,
    client: httpx.AsyncClient,
    api_key: str,
    base_url: str = DEFAULT_SERPER_SCRAPE_URL,
    max_chars: int,
    timeout: TimeoutConfig | None = None,
    retry: RetryConfig | None = None,
) -> dict[str, Any]:
    """Fetch *url* through Serper and return the same shape the other backends do."""
    if not url or not url.strip():
        return empty_result("URL cannot be empty")
    if is_banned_url(url):
        return empty_result("This URL is blocked (benchmark data source).")
    if not api_key:
        return empty_result("SERPER_API_KEY not set")

    resolved_retry = retry or default_scrape_retry()
    resolved_timeout = coerce_timeout(timeout)
    request_kwargs: dict[str, Any] = {
        "json": {"url": url},
        "headers": {"X-API-KEY": api_key, "Content-Type": "application/json"},
    }
    if resolved_timeout is not None:
        request_kwargs["timeout"] = resolved_timeout.to_httpx()

    last_error: str = "serper scrape failed"
    for attempt in range(1, resolved_retry.max_attempts + 1):
        try:
            response = await client.post(base_url, **request_kwargs)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            last_error = f"HTTP {status}"
            if resolved_retry.is_retryable_status(status) and attempt < resolved_retry.max_attempts:
                continue
            return empty_result(f"serper scrape failed: {last_error}")
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_error = type(exc).__name__
            if attempt < resolved_retry.max_attempts:
                continue
            return empty_result(f"serper scrape failed: {last_error}")
        except (ValueError, json.JSONDecodeError) as exc:
            return empty_result(f"serper scrape returned unparseable JSON: {exc}")

        text = str(payload.get("text") or "")
        if not text.strip():
            return empty_result("No content retrieved from the page.")
        truncated = len(text) > max_chars
        content = text[:max_chars] if truncated else text
        return {
            "success": True,
            "content": content,
            "error": "",
            "total_chars": len(text),
            "total_lines": text.count("\n") + 1,
            "truncated": truncated,
            "page_date": extract_serper_page_date(payload),
            "credits": payload.get("credits"),
        }

    return empty_result(f"serper scrape failed: {last_error}")
