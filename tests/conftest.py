"""Pytest fixtures shared across test files. Loaders live in tests/helpers.py."""

import pytest
from tenacity import wait_none

from app.ctgov.client import CTGovClient

from tests.helpers import fixture_studies


@pytest.fixture
def pembrolizumab_studies() -> list[dict]:
    return fixture_studies("pembrolizumab")


@pytest.fixture
def nsclc_studies() -> list[dict]:
    return fixture_studies("nsclc")


@pytest.fixture(autouse=True)
def no_retry_wait():
    """Real backoff waits up to 30s per retry; tests check retry counts, not timing."""
    original = CTGovClient._get_page.retry.wait
    CTGovClient._get_page.retry.wait = wait_none()
    yield
    CTGovClient._get_page.retry.wait = original
