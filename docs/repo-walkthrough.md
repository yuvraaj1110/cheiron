# Cheiron Repo Walkthrough

Nov 21, 2026 · @uv

> Historical snapshot, written at batch B2 (commit 118923f) when most of the
> pipeline had not been built yet. Kept for the design reasoning; for the
> finished repo read [`../README.md`](../README.md) and [`decisions.md`](decisions.md).

The repo at commit 118923f holds the project contract, config, and four real API fixtures; no pipeline code exists yet.

## At a glance

Ranked by importance to the project through B2. Five of the top seven don't exist yet.

| Rank | File | Job | Status |
| --- | --- | --- | --- |
| 1 | `CLAUDE.md` | Contract every Claude Code session builds from | Written |
| 2 | `app/api/schemas.py` | Request and response formats the frontend relies on | B1 step 3 |
| 3 | `app/planner/plan.py` | Plan format + allowed fields that constrain the LLM | B1 step 4 |
| 4 | `app/ctgov/normalize.py` | Raw study JSON → one clean record per trial | B2 |
| 5 | `tests/fixtures/*.json` | 4 saved API responses, the base for every test | Written |
| 6 | `app/ctgov/compiler.py` | Plan → API query parameters | B2 |
| 7 | `app/ctgov/client.py` | Paging, cap, retry, disk cache | B2 |
| 8 | `tests/test_*.py` | Proof the above works | B1–B2 |
| 9 | `scripts/fetch_fixtures.py` | Re-creates the fixtures | Written |
| 10 | `app/settings.py` | All config in one place | Written |
| 11 | `pyproject.toml` | Dependencies, packaging, pytest config | Written |
| 12 | `.env.example`, `.gitignore`, `__init__.py`, READMEs | Setup and housekeeping | Written |

## CLAUDE.md (61 lines)

This is the project's single source of truth. Claude Code loads it at the start of every session, so no session has to re-derive the design, and it doubles as the outline of your README.

| Section | Lines | What it says | Why it matters |
| --- | --- | --- | --- |
| Core rule | 5–6 | The LLM only fills a QueryPlan; all numbers come from code | Your answer to the "avoid hallucination" criterion (20% of the grade) |
| Pipeline | 8–9 | The 9 stages from request to response | Fixes the order modules call each other |
| Files | 11–21 | Which file owns which stage | Stops Claude from inventing new files or putting logic in the wrong place |
| Contract | 23–32 | Intents, dimensions, operators, viz types, the intent → viz table, network/citation/meta shapes | The vocabulary every module shares; changing it later means touching several files |
| Data rules | 34–40 | How phases, dates, countries and missing modules are handled | Encodes what the fixtures revealed about messy real data |
| Field paths | 42–51 | Exact JSON paths verified against the live API | The normalizer reads exactly these paths; citations point at them |
| API | 53–54 | Query params, `filter.advanced` syntax, page size, cap | What the compiler and client implement |
| Conventions | 56–61 | Python 3.11, Pydantic v2, async httpx, pytest, commit names, simplicity rule | Keeps code style consistent across sessions |

**Key decisions recorded here after checking real data (lines 37–40):**

- Observational studies have no `phases` key at all, so they go in the "Not applicable" bucket.
- Partial dates like `2019-05` are common, so year is parsed from whatever is present.
- Countries are full names, so they're used verbatim and citation excerpts stay exact.
- Any module can be missing, so the normalizer never assumes a key exists.

**To fix:** lines 50–51 (drug-name and override rules) sit under Field paths but belong in Data rules, and the Files list is missing `app/settings.py` and `scripts/fetch_fixtures.py`.

## scripts/fetch\_fixtures.py (93 lines)

A one-time script that downloads 4 real API responses and saves them as test data, so the test suite never depends on the live API. Run it with `pip install -e ".[dev]" && python scripts/fetch_fixtures.py`; without the install, the `import app` on line 14 fails.

**1. Field projection (lines 18–30).** Asks the API for only the 7 modules the project uses. Without it every study comes back with every field, and the files would be several times larger.

```python
FIELDS = ",".join([
    "protocolSection.identificationModule",
    "protocolSection.statusModule",
    "protocolSection.designModule",
    ...
])
```

**2. The fixture list (lines 35–43).** Maps each filename to the search that produces it. Each one covers a different case: one drug (`query.intr`), a broad condition (`query.cond`), a status filter, and a long condition name.

```python
FIXTURES = {
    "pembrolizumab.json": {"query.intr": "pembrolizumab"},
    "diabetes_recruiting.json": {"query.cond": "diabetes",
                                 "filter.overallStatus": "RECRUITING"},
    ...
}
```

**3. `fetch()` (lines 46–56).** One GET request: the search params plus the projection, `pageSize=100` and `countTotal=true` (so each file records the true total, e.g. 3,008 for pembrolizumab). `raise_for_status()` makes a bad response fail loudly instead of saving an error page as a fixture.

**4. `key_tree()` (lines 59–75).** A small recursive helper that prints the shape of the JSON (key names only, 4 levels deep, first list item only). This is how the field paths in CLAUDE.md were verified without dumping thousands of lines.

**5. `main()` (lines 78–89).** Writes each file, prints one summary line per fixture, then prints the key tree of one study.

**To fix:** line 3's run instruction needs the install step, and line 83's `indent=2` doubles the file size (4.2 MB total); dropping it halves that.

## tests/fixtures/\*.json (4 files, 4.2 MB)

Each file is page 1 (100 studies) of a real API search, saved exactly as returned. Every unit test in B2–B4 reads these instead of calling the API, so tests are fast, offline and repeatable.

**Shape of one file (trimmed):**

```json
{
  "totalCount": 3008,
  "nextPageToken": "...",
  "studies": [
    {"protocolSection": {
      "identificationModule": {"nctId": "NCT0...", "briefTitle": "..."},
      "statusModule": {"overallStatus": "RECRUITING",
                       "startDateStruct": {"date": "2019-05"}},
      "designModule": {"studyType": "INTERVENTIONAL",
                       "phases": ["PHASE1", "PHASE2"],
                       "enrollmentInfo": {"count": 120}},
      "sponsorCollaboratorsModule": {"leadSponsor": {"name": "...", "class": "INDUSTRY"}},
      "conditionsModule": {"conditions": ["..."]},
      "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "Pembrolizumab"}]},
      "contactsLocationsModule": {"locations": [{"country": "France", "city": "..."}]}
    }}
  ]
}
```

**What each covers** (counts out of 100 studies, checked Nov 21):

| Fixture | Total trials on CT.gov | No phases | Multi-phase | Partial dates | 2+ drugs | No locations |
| --- | --- | --- | --- | --- | --- | --- |
| pembrolizumab | 3,008 | 9 | 11 | 9 | 49 | 2 |
| breast\_cancer | 17,038 | 17 | 6 | 30 | 26 | 9 |
| diabetes\_recruiting | 1,975 | 21 | 1 | 1 | 21 | 0 |
| nsclc | 8,679 | 14 | 8 | 39 | 37 | 8 |

Every edge case the normalizer must handle shows up at least once. pembrolizumab and nsclc have the most multi-drug trials, so they drive the network tests. Never `cat` these files; inspect them with `python` or `jq`.

## app/settings.py + .env.example

All configuration lives in one class, so no module reads environment variables directly. That makes it easy to change a setting in tests and to document every knob in one place.

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    planner_model: str = "claude-sonnet-5-5"
    ctgov_base_url: str = "https://clinicaltrials.gov/api/v2/studies"
    max_studies: int = 2000
    cache_dir: str = ".cache"

settings = Settings()
```

- `BaseSettings` (from pydantic-settings) reads each field from the environment or `.env`, matching names case-insensitively (`MAX_STUDIES` → `max_studies`) and converting types (`"2000"` → `2000`).
- `extra="ignore"` means unrelated variables in `.env` don't crash startup.
- The empty `anthropic_api_key` default is deliberate: with no key, the app falls back to the rule planner instead of failing.
- `settings = Settings()` builds one shared instance at import time; every other file does `from app.settings import settings`.

`.env.example` lists the same 5 variables with a one-line comment each. A grader copies it to `.env`, adds a key, and runs. The real `.env` is gitignored, so the key never gets committed.

## pyproject.toml, .gitignore, \_\_init\_\_.py

**pyproject.toml (34 lines)** declares the project in 4 parts:

| Block | Lines | What it does |
| --- | --- | --- |
| `dependencies` | 6–14 | Runtime libraries: FastAPI + uvicorn (server), httpx (API calls), pydantic + pydantic-settings (validation, config), anthropic (LLM planner), tenacity (retries) |
| `dev` extras | 16–21 | Test-only: pytest, pytest-asyncio (async tests), respx (fakes httpx responses) |
| `build-system` + `packages.find` | 25–30 | Makes `pip install -e .` work and ships only the `app` package |
| `tool.pytest.ini_options` | 32–34 | Tests live in `tests/`; `asyncio_mode = "auto"` lets async tests run without a decorator on each one |

To fix: only pydantic has a minimum version. Add floors like `fastapi>=0.110` and `httpx>=0.27` so a grader installing months from now gets compatible versions.

**.gitignore** keeps the virtualenv, caches, build output, `.DS_Store` and `.env` out of git. `.cache/` matters most: the API client's disk cache will land there.

**\_\_init\_\_.py files** (8, all empty) mark `app/` and its 7 subfolders plus `tests/` as Python packages, so `from app.ctgov.client import ...` works.

**tests/fixtures/README.md, examples/README.md** are one-line placeholders. The fixtures one still says fixtures "land here"; update it to describe what each file covers.

## Not written yet (through B2)

These files don't exist yet. Each entry says what its main code will be, so you can check Claude Code's output against it.

**app/api/schemas.py (B1 step 3).** Pydantic models for the API contract. `VisualizationRequest` validates input: `query` is required at 3–500 characters, years fall between 1990 and 2035 with start ≤ end, and unknown fields are rejected. `VizSpec` is a discriminated union on `type`, so a `network_graph` must have nodes and edges and a `bar_chart` must have x and y. This file is what a frontend engineer reads.

```python
class VisualizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=3, max_length=500)
    drug_name: str | None = None
    start_year: int | None = Field(None, ge=1990, le=2035)
    ...
```

**app/planner/plan.py (B1 step 4).** The `Intent` and `Dimension` enums, the `DIMENSIONS` catalog (each dimension's kind, API path, and whether a trial can have several values), and `QueryPlan`. Its JSON schema becomes the LLM's tool schema, so the LLM can only choose listed values.

**app/ctgov/normalize.py (B2).** `normalize_study(raw) -> TrialRecord`: one pure function that reads the verified field paths with safe `.get()` chains, maps phases to labels, parses partial dates, strips dose text from drug names, dedupes countries, and stores the API path + raw value of each field for citations.

**app/ctgov/compiler.py (B2).** `compile_plan(plan) -> dict`: drug → `query.intr`, condition → `query.cond`, phase and years → `filter.advanced`, plus a `fields` projection limited to what the plan needs. `compile_groups()` makes one query per side of a comparison.

**app/ctgov/client.py (B2).** `CTGovClient.fetch(params, max_studies)`: an async loop that follows `nextPageToken` until the cap, retries 429/5xx with backoff, and caches each request on disk keyed by a hash of its params. It returns the studies, the true total, and whether the result was truncated.

**tests (B1–B2).** `test_schemas.py`, `test_plan.py`, and `test_ctgov_normalize/compiler/client.py`, all run against the fixtures, with respx faking HTTP.
