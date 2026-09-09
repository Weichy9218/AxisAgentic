"""Pin the site: fallback ladder in web_search.

Measured cause: models paste whole URLs after ``site:``. Google matches that
token against indexed page URLs, so ``site:host/api/v1/x.json`` is not a narrow
restriction, it is an unsatisfiable one. The ladder gives back the smallest
piece of intent first (the path), then the operator, and only when the query
already returned nothing.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx

from agentic.tools.web_search import search

if TYPE_CHECKING:
    import pytest


def test_site_token_rewrites_keep_the_rest_of_the_query() -> None:
    q = 'site:earthquake.usgs.gov/earthquakes/feed/v1.0/summary/ 4.5+ quakes "June 2026"'
    assert search._site_token_has_path(q) is True
    assert search._site_query_bare_domain(q) == 'site:earthquake.usgs.gov 4.5+ quakes "June 2026"'
    assert search._site_query_dropped(q) == '4.5+ quakes "June 2026"'

    bare = "site:manifold.markets new fish species"
    assert search._site_token_has_path(bare) is False
    assert search._site_query_bare_domain(bare) == bare
    assert search._site_query_dropped(bare) == "new fish species"

    assert search._site_token_has_path("no operator here") is False
    assert search._site_query_dropped("no operator here") == "no operator here"


class _QueryStubClient:
    """Stands in for the provider client: answers per query, records the order.

    The repo's convention for this seam is ``create_async_client`` (see
    tests/test_web_search_retry_policy.py); ``_do_search`` and the request
    helper under it are closures and cannot be patched directly.
    """

    def __init__(self, answers: dict[str, list[dict[str, Any]]]) -> None:
        self._answers = answers
        self.queries: list[str] = []

    async def post(self, _url: str, *, json: dict[str, Any], **_kw: Any) -> httpx.Response:
        query = json["q"]
        self.queries.append(query)
        body = {"organic": list(self._answers.get(query, [])), "searchParameters": {"q": query}}
        return httpx.Response(200, json=body, request=httpx.Request("POST", _url))

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> "_QueryStubClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


def _stub_do_search(monkeypatch: pytest.MonkeyPatch, answers: dict[str, list[dict]]) -> list[str]:
    client = _QueryStubClient(answers)
    monkeypatch.setattr(search, "create_async_client", lambda *_a, **_k: client)
    return client.queries


def _run(tool_fn, query: str):
    return asyncio.run(tool_fn(query=query))


def test_empty_site_query_falls_back_to_bare_domain_then_drops_the_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = "site:legulegu.com/stockdata/charts/827 dividend yield"
    seen = _stub_do_search(monkeypatch, {"site:legulegu.com dividend yield": [{"link": "https://legulegu.com/x"}]})

    tool = search.create_web_search_tool(serper_api_key="key")
    result = _run(tool._fn, query)

    assert seen == [query, "site:legulegu.com dividend yield"]
    assert result.metadata["site_retry_used"] == "bare_domain"
    assert result.metadata["request_count"] == 2


def test_ladder_reaches_the_dropped_operator_when_the_bare_domain_is_still_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = "site:polymarket.com/event/xrp-price-june XRP price June"
    seen = _stub_do_search(monkeypatch, {"XRP price June": [{"link": "https://example.com"}]})

    tool = search.create_web_search_tool(serper_api_key="key")
    result = _run(tool._fn, query)

    assert seen == [
        query,
        "site:polymarket.com XRP price June",
        "XRP price June",
    ]
    assert result.metadata["site_retry_used"] == "dropped"
    assert result.metadata["request_count"] == 3


def test_a_query_that_returns_results_still_costs_exactly_one_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = "site:bls.gov Employment Situation May 2026"
    seen = _stub_do_search(monkeypatch, {query: [{"link": "https://bls.gov/x"}]})

    tool = search.create_web_search_tool(serper_api_key="key")
    result = _run(tool._fn, query)

    assert seen == [query]
    assert result.metadata["site_retry_used"] is None
    assert result.metadata["request_count"] == 1


def test_quote_retry_runs_before_the_site_ladder_and_both_are_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = 'site:manifold.markets/manifold "exact market title" trillionaire'
    stripped = "site:manifold.markets/manifold exact market title trillionaire"
    bare = "site:manifold.markets exact market title trillionaire"
    seen = _stub_do_search(monkeypatch, {bare: [{"link": "https://manifold.markets/x"}]})

    tool = search.create_web_search_tool(serper_api_key="key")
    result = _run(tool._fn, query)

    assert seen == [query, stripped, bare]
    assert result.metadata["quote_retry_used"] is True
    assert result.metadata["site_retry_used"] == "bare_domain"
