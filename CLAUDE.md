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
- app/planner/rule_planner.py, llm_planner.py, validator.py
- app/ctgov/compiler.py, client.py, normalize.py
- app/analytics/operators.py
- app/viz/builder.py
- app/citations/attach.py
- tests/fixtures/*.json: saved API responses (large; inspect with python/jq, never cat)
- examples/: real request/response pairs

## Contract
Intents: count, list, distribution, trend, comparison, geo, network, histogram, correlation
Dimensions: phase, status, start_year, sponsor, sponsor_class, country, intervention_type, drug, condition, study_type, enrollment (numeric), duration_months (numeric)
Operators: count, list, count_by, time_series, compare, top_k, histogram, scatter, network. Every operator returns rows AND contributors: dict[datum_key, set[nct_id]].
Viz types: stat, table, bar_chart, grouped_bar_chart, time_series, histogram, scatter_plot, network_graph
Intent -> viz: count->stat, list->table, distribution->bar_chart, trend->time_series, comparison->grouped_bar_chart, geo->bar_chart (horizontal, ISO-3166 alpha-2 codes), network->network_graph, histogram->histogram, correlation->scatter_plot
Network data: {nodes:[{id,label,type,weight}], edges:[{source,target,weight,citations}]}
Citation: {nct_id, field (API path), value, excerpt (verbatim from API record), url}
meta: filters_applied, plan, total_matched, studies_analyzed, truncated, chart_recommended, viz_rationale, source, timings_ms
notes: interpretation, assumptions[], overrides[]

## Data rules
- Phases: PHASE1|PHASE2 -> "Phase 1/2" as its own bucket; NA -> "Not applicable"; fixed sort order
- Dates: partial "YYYY-MM" ok; year = start date year; missing date = excluded, counted in notes
- Countries: a trial counts once per distinct country
- Drug names: lowercase key, strip dose/route text, keep display label
- User structured fields override LLM-extracted filters; log each override in notes.overrides

## API
Base https://clinicaltrials.gov/api/v2/studies. Params: query.intr, query.cond, query.spons, query.locn, filter.overallStatus, filter.advanced (AREA[Phase]PHASE3, AREA[StartDate]RANGE[2015-01-01,MAX]), fields (projection), pageSize=1000, pageToken, countTotal=true. Cap at MAX_STUDIES (default 2000).

## Conventions
- Python 3.11+, Pydantic v2, httpx async, type hints everywhere, no print debugging left in
- Tests: pytest, run `pytest -q -x`; unit tests use fixtures, never the live API
- Keep functions small and pure in analytics/ and viz/
- Commit at the end of each batch: "B<n>: <summary>"
- Simplicity is a hard requirement: prefer the most readable version a human can explain out loud. No clever tricks, no premature abstraction, no metaclasses/decorators where a plain function works.
