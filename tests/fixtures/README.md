# Test fixtures

Real ClinicalTrials.gov API v2 responses, saved so unit tests never call the
live API. Tests stay fast, work offline, and give the same answer every run —
which matters most for the verbatim-citation test, whose whole job is to prove
no text was invented. A guard whose input shifts underneath it guards nothing.

Regenerate with `python scripts/fetch_fixtures.py`. Each file is one response:
the search that produced it, `totalCount` for the whole search, and the first
100 studies.

| File | Search | totalCount | Studies | Size |
| --- | --- | --- | --- | --- |
| `pembrolizumab.json` | `query.intr=pembrolizumab` | 3,008 | 100 | 0.7 MB |
| `breast_cancer.json` | `query.cond=breast cancer` | 17,038 | 100 | 0.4 MB |
| `diabetes_recruiting.json` | `query.cond=diabetes`, `filter.overallStatus=RECRUITING` | 1,975 | 100 | 0.6 MB |
| `nsclc.json` | `query.cond=non-small cell lung cancer` | 8,679 | 100 | 0.8 MB |

Chosen to cover the four cases the code has to survive: a single drug, a broad
condition, a status filter, and a condition with a messy name.

## Why these four are enough

Between them they exercise every awkward case in the real data:

- **Missing `designModule.phases`** — 9–21% of studies per file (observational
  studies have no phase at all, which is different from `phases: ["NA"]`).
- **Partial `YYYY-MM` start dates** — 1% to 39% depending on the file.
- **Missing start date entirely** — 0 to 2 studies per file.
- **No locations** — 0 to 9 studies per file.
- **Phase combinations** — `PHASE1|PHASE2` and `PHASE2|PHASE3` both appear, so
  the "Phase 1/2" bucketing rule has real input.

## Reading them

These are single-line JSON, hundreds of thousands of characters wide. Never
`cat` one. Inspect with Python or `jq`:

```bash
jq '.studies[0].protocolSection.designModule' tests/fixtures/pembrolizumab.json
```
