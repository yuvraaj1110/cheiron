# ctgov-viz

Backend that turns a clinical-trials question into a visualization spec backed by ClinicalTrials.gov API v2 data. Take-home; README, schemas and example outputs are graded.

## Core rule
The LLM ONLY fills a QueryPlan (enums + filter strings). It never produces numbers, chart data or citations. Everything after the plan is deterministic, tested Python.

## Pipeline
request -> planner (LLM or rule fallback) -> validator/merger -> compiler (plan -> API params) -> client (paging, cap, cache) -> normalizer (raw -> TrialRecord) -> operator -> viz builder (rule table) -> citation attacher -> response

## Files
- app/main.py: FastAPI; POST /v1/visualize, GET /health, GET /v1/schema
- app/api/schemas.py: VisualizationRequest, VisualizationResponse, VizSpec, Encoding, Citation, Meta, Notes
- app/planner/plan.py: QueryPlan, Dimension/Intent/Metric enums, DIMENSIONS catalog
- app/planner/rule_planner.py, llm_planner.py (OpenAI structured outputs), validator.py
- app/ctgov/compiler.py, client.py, normalize.py
- app/analytics/operators.py
- app/viz/builder.py
- app/citations/attach.py
- app/settings.py: every config value, read from .env (pydantic-settings)
- scripts/fetch_fixtures.py: one-off script that downloads the fixtures below
- tests/fixtures/*.json: saved API responses (large; inspect with python/jq, never cat)
- examples/: real request/response pairs

## Contract
Intents: count, list, distribution, trend, comparison, geo, network, histogram, correlation
Dimensions: phase, status, start_year, sponsor, sponsor_class, country, intervention_type, drug, condition, study_type, enrollment (numeric), duration_months (numeric)
Operators: count, list, count_by, time_series, compare, top_k, histogram, scatter, network. Every operator returns rows AND contributors: dict[datum_key, set[nct_id]].
Viz types: stat, table, bar_chart, grouped_bar_chart, time_series, histogram, scatter_plot, network_graph
Intent -> viz: count->stat, list->table, distribution->bar_chart, trend->time_series, comparison->grouped_bar_chart, geo->bar_chart (horizontal, country names exactly as the API returns them), network->network_graph, histogram->histogram, correlation->scatter_plot
Network data: {nodes:[{id,label,type,weight}], edges:[{source,target,weight,citations}]}
Citation: {nct_id, field (API path), value, excerpt (verbatim from API record), url}
meta: filters_applied, plan, total_matched, studies_analyzed, truncated, chart_recommended, viz_rationale, source, timings_ms
notes: interpretation, assumptions[], overrides[]

## Data rules
(Confirmed against the saved fixtures - see "Field paths" below.)
- Phases: eight buckets, in this sort order: Early Phase 1, Phase 1, Phase 1/2, Phase 2, Phase 2/3, Phase 3, Phase 4, Not applicable
- Phases: a combination becomes its own bucket (PHASE1|PHASE2 -> "Phase 1/2", PHASE2|PHASE3 -> "Phase 2/3"), so each trial lands in exactly one bucket and counts sum to the trial total
- Phases: observational studies have NO designModule.phases key at all (9-21% of a typical result). Same bucket as NA, "Not applicable"; say so in notes.assumptions
- Dates: partial "YYYY-MM" is common (up to 39% of a result), so handle it; year = start date year; missing date = excluded, counted in notes
- Countries: the API returns full names ("France", "United States"), NOT ISO codes. Use the name verbatim - no lookup table, so citation excerpts stay verbatim. A trial counts once per distinct country
- Any module can be absent: armsInterventionsModule, contactsLocationsModule.locations and designModule.enrollmentInfo.count are all missing on real studies. Normalizer must never assume a key exists
- Drug names: lowercase key, strip dose/route text, keep display label
- User structured fields override LLM-extracted filters; log each override in notes.overrides

## Field paths (verified against the live API, Nov 2026)
- protocolSection.identificationModule.nctId / .briefTitle
- protocolSection.statusModule.overallStatus / .startDateStruct.date / .completionDateStruct.date
- protocolSection.designModule.studyType / .phases (list) / .enrollmentInfo.count
- protocolSection.sponsorCollaboratorsModule.leadSponsor.name / .class
- protocolSection.conditionsModule.conditions (list of str)
- protocolSection.armsInterventionsModule.interventions[].type / .name
- protocolSection.contactsLocationsModule.locations[].country / .city / .facility

## API
Base https://clinicaltrials.gov/api/v2/studies. Params: query.intr, query.cond, query.spons, query.locn, filter.overallStatus, filter.advanced (AREA[Phase]PHASE3, AREA[StartDate]RANGE[2015-01-01,MAX]), fields (projection), pageSize=1000, pageToken, countTotal=true. Cap at MAX_STUDIES (default 10000).

## Conventions
- Python 3.11+, Pydantic v2, httpx async, type hints everywhere, no print debugging left in
- LLM: OpenAI structured outputs (AsyncOpenAI, model from settings.planner_model, temperature 0).
  The response_format IS the QueryPlan model, so the SDK validates the shape before we see it.
  No other provider SDK in the project.
- Tests: pytest, run `pytest -q -x`; unit tests use fixtures, never the live API
- Keep functions small and pure in analytics/ and viz/
- Commit at the end of each batch: "B<n>: <summary>"
- Simplicity is a hard requirement: prefer the most readable version a human can explain out loud. No clever tricks, no premature abstraction, no metaclasses/decorators where a plain function works.
