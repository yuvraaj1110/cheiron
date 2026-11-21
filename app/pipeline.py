"""One request in, one VisualizationResponse out.

The whole pipeline, top to bottom, in the order CLAUDE.md lists it:

    plan -> compile -> fetch -> normalize -> compute -> build -> cite

Every stage is timed into meta.timings_ms. The only stage that may call an LLM
is `plan`; everything after it is deterministic code over the fetched records.
"""

import asyncio
import time

from openai import AsyncOpenAI

from app.analytics import operators as op
from app.analytics.operators import OperatorResult
from app.api.schemas import Meta, Notes, VisualizationRequest, VisualizationResponse
from app.citations.attach import attach
from app.ctgov.client import CTGovClient, FetchResult
from app.ctgov.compiler import compile_groups
from app.ctgov.normalize import TrialRecord, normalize_studies
from app.planner.plan import Dimension, Intent, QueryPlan
from app.planner.validator import make_plan
from app.viz.builder import build_spec

PHASE_NOTE = "'Not applicable' phase includes observational trials, which have no phase."
NO_MATCH_NOTE = "No trials matched these filters. Try a broader condition, drug or year range."
DEFAULT_GEO_TOP_K = 15


class Timer:
    """Collects how long each named stage took, in milliseconds."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._start = time.perf_counter()

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        self.timings[stage] = round((now - self._start) * 1000, 1)
        self._start = now


def compute(
    plan: QueryPlan,
    records: list[TrialRecord],
    groups: dict[str, list[TrialRecord]],
    total_count: int,
) -> OperatorResult:
    """Run the one operator the plan's intent calls for."""
    intent = plan.intent
    if intent is Intent.count:
        return op.count(records, total_count=total_count)
    if intent is Intent.list:
        return op.list_trials(records)
    if intent is Intent.trend:
        start = plan.time.start if plan.time else plan.filters.start_year
        end = plan.time.end if plan.time else plan.filters.end_year
        return op.time_series(records, start=start, end=end, series_by=plan.series_by)
    if intent is Intent.comparison:
        return op.compare(groups, plan.group_by or Dimension.phase)
    if intent is Intent.geo:
        return op.top_k(records, plan.group_by or Dimension.country, plan.top_k or DEFAULT_GEO_TOP_K)
    if intent is Intent.network:
        return op.network(records, plan.network.node_dims, min_edge_weight=plan.network.min_edge_weight)
    if intent is Intent.histogram:
        return op.histogram(records, plan.numeric_x)
    if intent is Intent.correlation:
        return op.scatter(records, plan.numeric_x, plan.numeric_y)
    # distribution
    dim = plan.group_by or Dimension.phase
    if plan.top_k:
        return op.top_k(records, dim, plan.top_k)
    return op.count_by(records, dim)


async def fetch_all(
    client: CTGovClient, searches: list[tuple[str, dict]], max_studies: int
) -> list[FetchResult]:
    """One fetch per search, concurrently (comparisons have several)."""
    return await asyncio.gather(*(client.fetch(params, max_studies) for _, params in searches))


def uses_phase(plan: QueryPlan) -> bool:
    return Dimension.phase in (plan.group_by, plan.series_by) or (
        plan.intent is Intent.comparison and plan.group_by is None
    )


async def run(
    request: VisualizationRequest,
    client: CTGovClient,
    llm_client: AsyncOpenAI | None = None,
) -> VisualizationResponse:
    timer = Timer()

    plan, overrides, plan_notes = await make_plan(request, llm_client)
    timer.lap("plan")

    searches = compile_groups(plan)
    results = await fetch_all(client, searches, request.max_studies)
    timer.lap("fetch")

    groups = {label: normalize_studies(r.studies) for (label, _), r in zip(searches, results)}
    records = [record for group in groups.values() for record in group]
    total_matched = sum(r.total_count for r in results)
    studies_analyzed = sum(r.studies_analyzed for r in results)
    timer.lap("normalize")

    assumptions = list(plan.assumptions) + plan_notes
    assumptions += [r.truncation_note for r in results if r.truncation_note]

    # Nothing matched: answer "0" honestly rather than draw an empty chart.
    if not records and plan.intent is not Intent.count:
        plan = plan.model_copy(update={"intent": Intent.count})
        assumptions.append(NO_MATCH_NOTE)

    result = compute(plan, records, groups, total_matched)
    timer.lap("compute")

    built = build_spec(plan, result, request.viz_hint)
    timer.lap("build")

    spec = attach(built.spec, result, {r.nct_id: r for r in records}, plan)
    timer.lap("cite")

    if uses_phase(plan):
        assumptions.append(PHASE_NOTE)
    assumptions += built.notes

    filters_applied = (
        searches[0][1] if len(searches) == 1 else {label: params for label, params in searches}
    )

    return VisualizationResponse(
        visualization=spec,
        meta=Meta(
            filters_applied=filters_applied,
            plan=plan.model_dump(mode="json", exclude_none=True),
            total_matched=total_matched,
            studies_analyzed=studies_analyzed,
            truncated=total_matched > studies_analyzed,
            chart_recommended=built.chart_recommended,
            viz_rationale=built.rationale,
            timings_ms=timer.timings,
        ),
        notes=Notes(
            interpretation=plan.interpretation,
            assumptions=list(dict.fromkeys(assumptions)),  # de-duplicated, order kept
            overrides=overrides,
        ),
    )
