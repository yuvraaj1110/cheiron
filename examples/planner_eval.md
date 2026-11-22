# Planner evaluation

`python scripts/eval_planners.py` over 15 fixed questions, covering all nine
intents. Plans only: no ClinicalTrials.gov calls, so this is a measure of the
planners and nothing downstream of them.

Each question is scored only on the plan fields its intent's operator reads
(plus the filters it asserts), because a field the chart never reads cannot
change the answer. Expectations come from the contract in `CLAUDE.md`.

## Totals

| Planner | Questions | Intent correct | Checks passed | Exact plans |
| --- | --- | --- | --- | --- |
| `rule` | 15 | 13/15 (87%) | 39/47 (83%) | 10/15 (67%) |
| `llm` | — | not run (OPENAI_API_KEY not set) | — | — |

- **Intent correct** — the right intent, and so the right chart type.
- **Checks passed** — every expectation, intent and filters included.
- **Exact plans** — questions where nothing at all was missed.

## Per question

| # | Question | `rule` |
| --- | --- | --- |
| 01 | How many recruiting Alzheimer's disease trials are there? | pass |
| 02 | What is the total number of completed trials sponsored by Pfizer? | 1/3 — intent, sponsor |
| 03 | List phase 3 trials for pembrolizumab | pass |
| 04 | How are breast cancer trials distributed across phases? | pass |
| 05 | Which sponsors run the most obesity trials? | pass |
| 06 | What share of melanoma trials are run by industry versus government? | 2/3 — group_by |
| 07 | How many trials for pembrolizumab started each year since 2015? | pass |
| 08 | Has interest in gene therapy trials grown over time? | pass |
| 09 | Compare phases for trials involving pembrolizumab vs nivolumab | pass |
| 10 | How do these two drugs compare by trial phase? | 2/3 — group_by |
| 11 | Which countries have the most recruiting trials for diabetes? | pass |
| 12 | Show a network of sponsors and drugs for non-small cell lung cancer trials | pass |
| 13 | Which drugs frequently co-occur with pembrolizumab in combination studies? | pass |
| 14 | What does the distribution of enrollment look like for Alzheimer's trials? | 2/3 — condition |
| 15 | Is enrollment related to study duration in obesity trials? | 1/4 — intent, numeric_x, numeric_y |

A cell shows `pass`, or the score with the checks that failed. Full detail,
including each planner's own interpretation line, is in
[`planner_eval.json`](planner_eval.json).
