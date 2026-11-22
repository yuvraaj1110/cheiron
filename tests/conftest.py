"""Pytest fixtures shared across test files. Loaders live in tests/helpers.py."""

import pytest
from tenacity import wait_none

from app.ctgov.client import CTGovClient
from app.settings import settings

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


@pytest.fixture(autouse=True)
def no_ambient_api_key(monkeypatch):
    """Tests must never reach the real OpenAI API.

    settings reads .env, so a developer with a real OPENAI_API_KEY would
    otherwise have the pipeline tests build a live client, call out, and fall
    back to the rule planner - passing, but slowly and over the network. Tests
    that exercise the LLM planner set the key themselves and inject a fake
    client.
    """
    monkeypatch.setattr(settings, "openai_api_key", "")
