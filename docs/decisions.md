# Decision log

Observations about the data and design decisions, with the evidence for each.
Newest first. Mirrors the working log kept alongside development.

| Date | Area | Observation | Decision | Evidence |
| --- | --- | --- | --- | --- |
| Nov 21 | Planner | Without a drug/condition field, the rule planner searched the whole registry (612,892 trials for "breast cancer by phase") | Rule planner reads unquoted names with simple patterns (flagged low confidence) and says so when it finds none | Live example run |
| Nov 21 | Scale | ClinicalTrials.gov returns HTTP 429 on bursts; a 10,000-trial pull is 10 pages | Exponential backoff, up to 6 attempts; final failure becomes a 502 with the status code | Live example run |
| Nov 21 | Scale | A capped pull counts only the first N trials in API order (a count said 400 when the API total was 604) | `count` answers from the API's `totalCount`; cap raised to 10,000; when still capped, notes say "first N of M in API order; not a random sample" | Live end-to-end check |
| Nov 21 | Network | The drug dimension included every intervention ("Laboratory Biomarker Analysis" appeared as a drug) | Drug = interventions of type DRUG, BIOLOGICAL or COMBINATION_PRODUCT only | `operators.py` |
| Nov 21 | Citations | 268 citations across 6 live queries: every excerpt found verbatim in its source record | Verbatim-excerpt design confirmed on live data | Live end-to-end check |
| Nov 21 | Planner | Available key is OpenAI | `gpt-4.1-mini`, structured outputs, temperature 0. Strict mode rejects some schema keywords, so a relaxed copy is sent and the reply re-validated with the full model | `llm_planner.py` |
| Nov 21 | Schemas | 8 chart types share title/encoding/options but need different channels and data shapes | One base class with one required-channel validator, 8 small subclasses, a union keyed on `type` | `schemas.py` |
| Nov 21 | Data: phases | Fixtures contain PHASE2+PHASE3 and EARLY_PHASE1 too | 8 buckets: Early Phase 1, Phase 1, Phase 1/2, Phase 2, Phase 2/3, Phase 3, Phase 4, Not applicable | Fixture profile |
| Nov 21 | Schemas | Request, plan and API could drift on phase/status spellings | Phase and Status enums use the API's own spellings, shared by request and plan | `schemas.py`, `plan.py` |
| Nov 21 | Data: phases | Observational studies have no `phases` key (9–21 of 100 per fixture) | "Not applicable" bucket, explained in notes | Fixture profile |
| Nov 21 | Data: dates | Partial `YYYY-MM` start dates in up to 39 of 100 | Year from whatever is present; missing dates excluded and counted | Fixture profile |
| Nov 21 | Data: countries | API returns full names, not ISO codes | Names used verbatim so citation excerpts stay exact | Live API |
| Nov 21 | Data: countries | One trial lists sites in many countries | Counted once per distinct country | `CLAUDE.md` |
| Nov 21 | Data: missing modules | Interventions, locations, enrollment absent on some studies | Normalizer never assumes a key exists | Fixture profile |
| Nov 21 | Data: combinations | 21–49 of 100 trials list 2+ drugs | Drug ↔ drug co-occurrence network is viable | Fixture profile |
| Nov 21 | Testing | — | Unit tests read 4 saved API responses (400 studies), never the live API | `tests/fixtures/` |
| Nov 21 | Code style | — | Simplicity is a hard requirement: code a person can explain out loud | `CLAUDE.md` |
| Nov 20 | Output | Some questions ("how many", "list") don't need a chart | Always return a visualization; `stat` and `table` types plus `chart_recommended` and `viz_rationale` | Design |
| Nov 20 | AI design | LLMs hallucinate numbers | LLM only fills a QueryPlan; all numbers, data and citations come from code | Design |
| Nov 20 | AI design | — | Chart type chosen by a fixed intent → chart table, not the LLM | Design |
| Nov 20 | AI design | — | Rule-based planner fallback when there's no key or the LLM fails; one repair retry first | Design |
| Nov 20 | Inputs | — | Request's structured fields override LLM-extracted filters; each override reported | Design |
| Nov 20 | Citations | — | Excerpts copied verbatim from the API record; a test checks each appears in the source | Design |
