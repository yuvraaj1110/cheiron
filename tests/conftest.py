"""Pytest fixtures shared across test files. Loaders live in tests/helpers.py."""

import pytest

from tests.helpers import fixture_studies


@pytest.fixture
def pembrolizumab_studies() -> list[dict]:
    return fixture_studies("pembrolizumab")


@pytest.fixture
def nsclc_studies() -> list[dict]:
    return fixture_studies("nsclc")
