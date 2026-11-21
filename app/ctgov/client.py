"""The only thing in the project that talks to the network.

Four jobs, each small:

- page through results until we have enough,
- stop at a cap so one question cannot pull 300,000 trials,
- retry the failures that are worth retrying,
- cache responses on disk so repeated questions and demo runs are instant.

Keeping all of that here means every other module is pure and testable without
a network, and the tests for this file mock HTTP with respx.
"""

import hashlib
import json
from pathlib import Path

import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.settings import settings

# The API's own maximum page size. Asking for more is an error.
MAX_PAGE_SIZE = 1000

# Statuses worth retrying: rate limiting and the server's own failures. A 400
# or 404 means our request was wrong, so retrying would just repeat it.
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
RETRY_ATTEMPTS = 6


class FetchResult(BaseModel):
    """What one search returned."""

    studies: list[dict]
    total_count: int  # trials matching the search, per the API
    truncated: bool  # True if the cap stopped us short of total_count

    @property
    def studies_analyzed(self) -> int:
        return len(self.studies)

    @property
    def truncation_note(self) -> str | None:
        """The sentence the pipeline puts in notes.assumptions when capped.

        It lives here because this is the only place that knows both numbers.
        The point is to be explicit that a capped result is the API's first N
        studies - not a sample - so nobody reads a 400-trial chart of a
        604-trial search as representative. (The count intent sidesteps the
        problem by answering from total_count instead.)
        """
        if not self.truncated:
            return None
        return (
            f"Counted the first {len(self.studies)} of {self.total_count} "
            "trials in API order; not a random sample."
        )


def _is_retryable(exc: BaseException) -> bool:
    """Timeouts, connection failures, and the server-side status codes."""
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRY_STATUS_CODES
    return False


class CTGovClient:
    """Async client for ClinicalTrials.gov API v2.

    Used as a context manager so the HTTP connection pool is closed properly:

        async with CTGovClient() as client:
            result = await client.fetch(params, max_studies=500)
    """

    def __init__(self, base_url: str | None = None, cache_dir: str | None = None):
        self.base_url = base_url or settings.ctgov_base_url
        self.cache_dir = Path(cache_dir or settings.cache_dir)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "CTGovClient":
        self._client = httpx.AsyncClient(timeout=30)
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # --- caching ----------------------------------------------------------

    def _cache_path(self, params: dict, max_studies: int) -> Path:
        """One file per distinct search.

        The key is a hash of the sorted params plus the cap, so the same
        question always lands on the same file regardless of dict ordering,
        and a different cap is a different file (a 50-study answer must not be
        served to a request that asked for 2000).
        """
        key = json.dumps({"params": params, "max_studies": max_studies}, sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, path: Path) -> FetchResult | None:
        if not path.exists():
            return None
        try:
            return FetchResult.model_validate_json(path.read_text())
        except Exception:
            # A truncated or stale cache file must never break a request.
            return None

    def _write_cache(self, path: Path, result: FetchResult) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json())

    # --- one page ---------------------------------------------------------

    @retry(
        retry=retry_if_exception(_is_retryable),
        # ClinicalTrials.gov rate-limits bursts (HTTP 429, about 50 requests a
        # minute), and a 10,000-study pull is 10 pages, so back off for real.
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,  # surface the real error, not tenacity's wrapper
    )
    async def _get_page(self, params: dict) -> dict:
        """One HTTP GET, retried up to three times on retryable failures."""
        if self._client is None:
            # Not an assert: python -O would strip it and we would get an
            # AttributeError on None instead of a sentence that says what to do.
            raise RuntimeError("use CTGovClient as an async context manager")
        response = await self._client.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()

    # --- the public call --------------------------------------------------

    async def fetch(self, params: dict, max_studies: int | None = None) -> FetchResult:
        """Page through a search until `max_studies` or the results run out.

        `params` comes from the compiler and is passed through untouched; only
        the paging keys are added here.
        """
        cap = max_studies or settings.max_studies
        cache_path = self._cache_path(params, cap)

        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        studies: list[dict] = []
        total_count = 0
        page_token: str | None = None

        while len(studies) < cap:
            page_params = {
                **params,
                # Never ask for more than we still want, or more than the API allows.
                "pageSize": min(MAX_PAGE_SIZE, cap - len(studies)),
            }
            if page_token:
                page_params["pageToken"] = page_token

            page = await self._get_page(page_params)

            # countTotal only comes back on the first page; keep the first value.
            total_count = total_count or page.get("totalCount", 0)
            studies.extend(page.get("studies", []))

            page_token = page.get("nextPageToken")
            if not page_token:
                break

        # If the API reported no total, the only number we can stand behind is
        # what we actually received.
        total_count = total_count or len(studies)

        result = FetchResult(
            studies=studies,
            total_count=total_count,
            truncated=total_count > len(studies),
        )
        self._write_cache(cache_path, result)
        return result
