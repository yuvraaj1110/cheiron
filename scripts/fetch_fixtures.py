"""Download a few real ClinicalTrials.gov responses and save them as test fixtures.

Run once: python scripts/fetch_fixtures.py

Every unit test reads these saved files instead of calling the live API, so the
test suite is fast, offline and gives the same answer every time.
"""

import json
from pathlib import Path

import httpx

from app.settings import settings

FIXTURE_DIR = Path("tests/fixtures")

# Ask the API for only the modules we actually use. Without this it returns
# every field of every study, which makes the fixtures huge and slow to load.
FIELDS = ",".join(
    [
        "protocolSection.identificationModule",
        "protocolSection.statusModule",
        "protocolSection.designModule",
        "protocolSection.sponsorCollaboratorsModule",
        "protocolSection.conditionsModule",
        "protocolSection.armsInterventionsModule.interventions",
        "protocolSection.contactsLocationsModule.locations",
    ]
)

# One entry per fixture file: the filename, and the search that produces it.
# Chosen to cover the cases the code has to handle: a single drug, a broad
# condition, a status filter, and a condition with a messy name.
FIXTURES = {
    "pembrolizumab.json": {"query.intr": "pembrolizumab"},
    "breast_cancer.json": {"query.cond": "breast cancer"},
    "diabetes_recruiting.json": {
        "query.cond": "diabetes",
        "filter.overallStatus": "RECRUITING",
    },
    "nsclc.json": {"query.cond": "non-small cell lung cancer"},
}


def fetch(search_params: dict) -> dict:
    """Call the API once and return the parsed JSON response."""
    params = {
        **search_params,
        "fields": FIELDS,
        "pageSize": 100,
        "countTotal": "true",
    }
    response = httpx.get(settings.ctgov_base_url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def key_tree(value, depth: int, indent: int = 0) -> list[str]:
    """Return the shape of a JSON value as indented key names, no values.

    Used to confirm the real API field paths before any code depends on them.
    """
    if depth == 0:
        return []
    pad = "  " * indent
    lines = []
    if isinstance(value, dict):
        for key, child in value.items():
            lines.append(f"{pad}{key}")
            lines += key_tree(child, depth - 1, indent + 1)
    elif isinstance(value, list) and value:
        lines.append(f"{pad}[0]")
        lines += key_tree(value[0], depth - 1, indent + 1)
    return lines


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    for filename, search_params in FIXTURES.items():
        data = fetch(search_params)
        (FIXTURE_DIR / filename).write_text(json.dumps(data, indent=2))
        # Never print the data itself - these responses are thousands of lines.
        print(f"{filename}: totalCount={data['totalCount']} saved={len(data['studies'])}")

    print("\nField paths of one study (keys only):")
    one_study = json.loads((FIXTURE_DIR / "pembrolizumab.json").read_text())["studies"][0]
    print("\n".join(key_tree(one_study, depth=4)))


if __name__ == "__main__":
    main()
