"""Tests for app/ctgov/client.py.

All HTTP is mocked with respx, so these run offline and deterministically.
Each test pins one of the client's four jobs: paging, the cap, retrying, and
caching.

Every test gets its own cache directory via tmp_path. Sharing one would let an
earlier test's cached response satisfy a later test's request, and the suite
would pass without ever exercising the code under test.
"""

import httpx
import pytest
import respx

from app.ctgov.client import MAX_PAGE_SIZE, CTGovClient

BASE_URL = "https://api.example.test/studies"
PARAMS = {"query.intr": "pembrolizumab", "countTotal": "true"}


def study(n: int) -> dict:
    return {"protocolSection": {"identificationModule": {"nctId": f"NCT{n:08d}"}}}


def page(count: int, first: int = 0, total: int | None = None, token: str | None = None) -> dict:
    body: dict = {"studies": [study(first + i) for i in range(count)]}
    if total is not None:
        body["totalCount"] = total
    if token is not None:
        body["nextPageToken"] = token
    return body


def client(tmp_path) -> CTGovClient:
    return CTGovClient(base_url=BASE_URL, cache_dir=str(tmp_path))


# --- paging ---------------------------------------------------------------


async def test_follows_the_next_page_token(tmp_path):
    """Two pages, 1500 studies, and the second request must carry the token."""
    with respx.mock:
        route = respx.get(BASE_URL).mock(
            side_effect=[
                httpx.Response(200, json=page(1000, total=1500, token="page-2")),
                httpx.Response(200, json=page(500, first=1000)),
            ]
        )
        async with client(tmp_path) as c:
            result = await c.fetch(PARAMS, max_studies=2000)

    assert len(result.studies) == 1500
    assert result.total_count == 1500
    assert result.truncated is False  # we got everything
    assert route.call_count == 2

    first, second = route.calls
    assert "pageToken" not in first.request.url.params
    assert second.request.url.params["pageToken"] == "page-2"


async def test_stops_when_there_is_no_next_token(tmp_path):
    with respx.mock:
        route = respx.get(BASE_URL).mock(
            return_value=httpx.Response(200, json=page(42, total=42))
        )
        async with client(tmp_path) as c:
            result = await c.fetch(PARAMS, max_studies=2000)

    assert route.call_count == 1
    assert len(result.studies) == 42
    assert result.truncated is False


async def test_total_count_is_kept_from_the_first_page(tmp_path):
    """countTotal only comes back on page one; later pages must not reset it."""
    with respx.mock:
        respx.get(BASE_URL).mock(
            side_effect=[
                httpx.Response(200, json=page(10, total=25, token="page-2")),
                httpx.Response(200, json=page(15, first=10)),  # no totalCount
            ]
        )
        async with client(tmp_path) as c:
            result = await c.fetch(PARAMS, max_studies=2000)

    assert result.total_count == 25


async def test_falls_back_to_the_received_count_when_the_api_gives_no_total(tmp_path):
    """Without countTotal the only number we can stand behind is what arrived."""
    with respx.mock:
        respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={"studies": [study(1)]}))
        async with client(tmp_path) as c:
            result = await c.fetch(PARAMS, max_studies=2000)

    assert result.total_count == 1
    assert result.truncated is False


# --- the cap --------------------------------------------------------------


async def test_cap_stops_paging_and_sets_truncated(tmp_path):
    """The cap is what stops one question pulling 300,000 trials. When it
    bites, `truncated` must say so - otherwise the answer silently describes a
    slice while looking like it describes everything."""
    with respx.mock:
        route = respx.get(BASE_URL).mock(
            return_value=httpx.Response(200, json=page(10, total=5000, token="page-2"))
        )
        async with client(tmp_path) as c:
            result = await c.fetch(PARAMS, max_studies=10)

    assert route.call_count == 1  # stopped rather than following the token
    assert len(result.studies) == 10
    assert result.total_count == 5000
    assert result.truncated is True


async def test_page_size_never_exceeds_what_is_still_wanted(tmp_path):
    """Asking for 1000 when 50 are wanted wastes a large response."""
    with respx.mock:
        route = respx.get(BASE_URL).mock(
            return_value=httpx.Response(200, json=page(50, total=50))
        )
        async with client(tmp_path) as c:
            await c.fetch(PARAMS, max_studies=50)

    assert route.calls[0].request.url.params["pageSize"] == "50"


async def test_page_size_never_exceeds_the_api_maximum(tmp_path):
    """The API rejects a pageSize above 1000, so the cap must be clamped."""
    with respx.mock:
        route = respx.get(BASE_URL).mock(
            return_value=httpx.Response(200, json=page(5, total=5))
        )
        async with client(tmp_path) as c:
            await c.fetch(PARAMS, max_studies=5000)

    assert route.calls[0].request.url.params["pageSize"] == str(MAX_PAGE_SIZE)


async def test_compiler_params_are_passed_through_untouched(tmp_path):
    """The client adds paging keys and changes nothing else."""
    with respx.mock:
        route = respx.get(BASE_URL).mock(
            return_value=httpx.Response(200, json=page(1, total=1))
        )
        async with client(tmp_path) as c:
            await c.fetch({**PARAMS, "filter.advanced": "AREA[Phase]PHASE3"}, max_studies=10)

    sent = route.calls[0].request.url.params
    assert sent["query.intr"] == "pembrolizumab"
    assert sent["filter.advanced"] == "AREA[Phase]PHASE3"
    assert sent["countTotal"] == "true"


# --- retrying -------------------------------------------------------------


async def test_retries_a_503_then_succeeds(tmp_path):
    """A server-side failure is worth retrying; the caller sees only success."""
    with respx.mock:
        route = respx.get(BASE_URL).mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(200, json=page(3, total=3)),
            ]
        )
        async with client(tmp_path) as c:
            result = await c.fetch(PARAMS, max_studies=10)

    assert route.call_count == 2
    assert len(result.studies) == 3


async def test_retries_a_timeout(tmp_path):
    with respx.mock:
        route = respx.get(BASE_URL).mock(
            side_effect=[
                httpx.TimeoutException("too slow"),
                httpx.Response(200, json=page(1, total=1)),
            ]
        )
        async with client(tmp_path) as c:
            result = await c.fetch(PARAMS, max_studies=10)

    assert route.call_count == 2
    assert len(result.studies) == 1


async def test_gives_up_after_three_attempts(tmp_path):
    """Three tries, then the real error surfaces rather than tenacity's
    wrapper, so the API layer can turn it into a clear 502."""
    with respx.mock:
        route = respx.get(BASE_URL).mock(return_value=httpx.Response(503))
        with pytest.raises(httpx.HTTPStatusError):
            async with client(tmp_path) as c:
                await c.fetch(PARAMS, max_studies=10)

    assert route.call_count == 3


async def test_does_not_retry_a_client_error(tmp_path):
    """A 400 means our request was wrong. Retrying would just repeat it."""
    with respx.mock:
        route = respx.get(BASE_URL).mock(return_value=httpx.Response(400))
        with pytest.raises(httpx.HTTPStatusError):
            async with client(tmp_path) as c:
                await c.fetch(PARAMS, max_studies=10)

    assert route.call_count == 1


# --- caching --------------------------------------------------------------


async def test_a_second_identical_fetch_makes_no_request(tmp_path):
    """The cache is what makes the demo and the example runs instant."""
    with respx.mock:
        route = respx.get(BASE_URL).mock(
            return_value=httpx.Response(200, json=page(7, total=7))
        )
        async with client(tmp_path) as c:
            first = await c.fetch(PARAMS, max_studies=10)
        async with client(tmp_path) as c:
            second = await c.fetch(PARAMS, max_studies=10)

    assert route.call_count == 1  # served from disk the second time
    assert second.studies == first.studies
    assert second.total_count == first.total_count


async def test_param_order_does_not_change_the_cache_key(tmp_path):
    """Same search written differently is the same search."""
    with respx.mock:
        route = respx.get(BASE_URL).mock(
            return_value=httpx.Response(200, json=page(2, total=2))
        )
        async with client(tmp_path) as c:
            await c.fetch({"a": "1", "b": "2"}, max_studies=10)
            await c.fetch({"b": "2", "a": "1"}, max_studies=10)

    assert route.call_count == 1


async def test_a_different_cap_is_a_different_cache_entry(tmp_path):
    """A 10-study answer must not be served to a request that wanted 1000,
    or `truncated` and the counts would describe the wrong fetch."""
    with respx.mock:
        route = respx.get(BASE_URL).mock(
            return_value=httpx.Response(200, json=page(10, total=5000, token="page-2"))
        )
        async with client(tmp_path) as c:
            small = await c.fetch(PARAMS, max_studies=10)
            larger = await c.fetch(PARAMS, max_studies=20)

    # 1 call for the cap of 10, then 2 more for the cap of 20 (the mock returns
    # 10 per page with a next token, so reaching 20 takes two pages). The point
    # is that the second fetch went to the network at all rather than being
    # served the 10-study answer from disk.
    assert route.call_count == 3
    assert len(small.studies) == 10
    assert len(larger.studies) == 20
    assert small.truncated is True


async def test_different_params_are_different_cache_entries(tmp_path):
    with respx.mock:
        route = respx.get(BASE_URL).mock(
            return_value=httpx.Response(200, json=page(1, total=1))
        )
        async with client(tmp_path) as c:
            await c.fetch({"query.cond": "diabetes"}, max_studies=10)
            await c.fetch({"query.cond": "melanoma"}, max_studies=10)

    assert route.call_count == 2


async def test_a_corrupt_cache_file_is_ignored_not_fatal(tmp_path):
    """A half-written file from an interrupted run must not break a request."""
    with respx.mock:
        route = respx.get(BASE_URL).mock(
            return_value=httpx.Response(200, json=page(4, total=4))
        )
        async with client(tmp_path) as c:
            path = c._cache_path(PARAMS, 10)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{ truncated json")
            result = await c.fetch(PARAMS, max_studies=10)

    assert route.call_count == 1
    assert len(result.studies) == 4
