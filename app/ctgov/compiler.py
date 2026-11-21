"""Turn a QueryPlan into ClinicalTrials.gov API query parameters.

Pure translation, no I/O: plan in, dict of params out. That is what makes it
testable by comparing against an exact expected dict, which is how the tests
here are written.

Paging parameters (pageSize, pageToken) are deliberately absent. They belong
to the client, which is the only thing that knows how many studies are still
wanted. This file answers "what are we asking for", the client answers "how do
we fetch it".
"""

from app.planner.plan import (
    DIMENSIONS,
    Dimension,
    Intent,
    PlanFilters,
    QueryPlan,
)

# --------------------------------------------------------------------------
# Which modules to ask for
# --------------------------------------------------------------------------
# The API returns everything unless you name a `fields` projection, which makes
# responses several times larger and slower. We ask only for the modules this
# plan can actually use.

IDENTIFICATION = "protocolSection.identificationModule"
STATUS = "protocolSection.statusModule"

# Needed by every answer: the NCT ID and title for citations and links, and the
# start date, which orders the citations we pick ("most recent first").
BASE_MODULES = (IDENTIFICATION, STATUS)


def module_for(dim: Dimension) -> str:
    """The API module a dimension is read from.

    Derived from the catalog's source path rather than a second hardcoded
    table, so a corrected path cannot leave this mapping stale.
    `duration_months` is computed from two dates, so it needs the status module.
    """
    path = DIMENSIONS[dim].source_path
    if path == "derived":
        return STATUS
    section, module, *_ = path.split(".")
    return f"{section}.{module}"


ALL_MODULES = tuple(sorted({module_for(dim) for dim in Dimension} | set(BASE_MODULES)))


def referenced_dimensions(plan: QueryPlan) -> list[Dimension]:
    """Every dimension this plan reads out of a record.

    Filters are not included: those are applied by the API server-side, so we
    never need their fields back in the response.
    """
    dims = [plan.group_by, plan.series_by, plan.numeric_x, plan.numeric_y]
    if plan.network:
        dims += plan.network.node_dims
    return [dim for dim in dims if dim is not None]


def modules_for_plan(plan: QueryPlan) -> list[str]:
    """The `fields` projection for this plan.

    A `list` intent is the one case that needs everything: its table shows
    phase, sponsor and status, which are not "dimensions" of the question but
    are still columns the user reads.
    """
    if plan.intent is Intent.list:
        return list(ALL_MODULES)
    modules = set(BASE_MODULES)
    modules.update(module_for(dim) for dim in referenced_dimensions(plan))
    return sorted(modules)


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
# Entity filters map to the API's own search params. Phase and the year window
# have no dedicated param, so they go through filter.advanced.

ENTITY_PARAMS: dict[str, str] = {
    "drug": "query.intr",
    "condition": "query.cond",
    "sponsor": "query.spons",
    "country": "query.locn",
}


def _phase_clause(phases: list) -> str:
    """AREA[Phase]PHASE3, or AREA[Phase](PHASE2 OR PHASE3) for several."""
    values = [p.value for p in phases]
    if len(values) == 1:
        return f"AREA[Phase]{values[0]}"
    return f"AREA[Phase]({' OR '.join(values)})"


def _start_date_clause(start_year: int | None, end_year: int | None) -> str:
    """AREA[StartDate]RANGE[2015-01-01,MAX].

    MIN and MAX are the API's own open-ended bounds, so a one-sided window
    needs no special case here.
    """
    start = f"{start_year}-01-01" if start_year else "MIN"
    end = f"{end_year}-12-31" if end_year else "MAX"
    return f"AREA[StartDate]RANGE[{start},{end}]"


def compile_filters(filters: PlanFilters) -> dict[str, str]:
    """Just the filter half of the params, so comparison groups can reuse it."""
    params: dict[str, str] = {}

    for filter_name, param_name in ENTITY_PARAMS.items():
        value = getattr(filters, filter_name)
        if value:
            params[param_name] = value

    if filters.status:
        params["filter.overallStatus"] = ",".join(s.value for s in filters.status)

    # filter.advanced takes several clauses joined by AND.
    advanced = []
    if filters.phase:
        advanced.append(_phase_clause(filters.phase))
    if filters.start_year or filters.end_year:
        advanced.append(_start_date_clause(filters.start_year, filters.end_year))
    if advanced:
        params["filter.advanced"] = " AND ".join(advanced)

    return params


def merge_filters(base: PlanFilters, override: PlanFilters) -> PlanFilters:
    """Overlay one set of filters on another, field by field.

    Used for comparison groups: the plan's own filters are the shared context
    ("NSCLC trials") and each group narrows it ("...with pembrolizumab").
    Only fields the override actually set take effect.
    """
    merged = base.model_dump()
    merged.update(override.model_dump(exclude_none=True))
    return PlanFilters(**merged)


# --------------------------------------------------------------------------
# The two entry points
# --------------------------------------------------------------------------


def compile_plan(plan: QueryPlan, filters: PlanFilters | None = None) -> dict[str, str]:
    """Plan -> API params for one search.

    `filters` overrides the plan's own, which is how a comparison group reuses
    everything else about the plan.
    """
    params = compile_filters(plan.filters if filters is None else filters)
    params["fields"] = ",".join(modules_for_plan(plan))
    params["countTotal"] = "true"  # so meta.total_matched is the real total
    return params


def compile_groups(plan: QueryPlan) -> list[tuple[str, dict[str, str]]]:
    """(label, params) per comparison group, or one unlabelled search.

    A comparison is several separate searches rather than one search split up
    afterwards, because "pembrolizumab vs nivolumab" is two different queries
    to the API - a single search could not tell which trials belong to which
    side once the results came back.
    """
    if not plan.compare_groups:
        return [("all", compile_plan(plan))]

    return [
        (group.label, compile_plan(plan, merge_filters(plan.filters, group.filters)))
        for group in plan.compare_groups
    ]
