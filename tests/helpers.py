"""Loading the saved API fixtures.

Kept out of conftest.py because conftest is pytest's own module and importing
it directly is discouraged; these are plain functions any test can import.
"""

import json
from functools import lru_cache
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_NAMES = ["pembrolizumab", "breast_cancer", "diabetes_recruiting", "nsclc"]


@lru_cache(maxsize=None)
def load_fixture(name: str) -> dict:
    """The whole saved response. Cached, since these are ~0.5 MB each."""
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text())


def fixture_studies(name: str) -> list[dict]:
    """Just the studies from one fixture."""
    return load_fixture(name)["studies"]


def all_fixture_studies() -> list[dict]:
    """Every study across all four fixtures: 400 real records."""
    return [study for name in FIXTURE_NAMES for study in fixture_studies(name)]
