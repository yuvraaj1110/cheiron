"""Between the planner and the API: merge the caller's fields, check the plan.

Two jobs, both deterministic:

1. Merge. If the caller set a structured field (drug_name, phase, ...), it wins
   over whatever the planner extracted. A field the planner left empty is just
   filled in; a field it got different is replaced AND logged, so the response
   says exactly what was overridden.
2. Check. Each intent needs certain parts of the plan (a network needs two node
   dimensions, a histogram needs a numeric axis, ...). Where a missing part can
   be rebuilt from the request, it is, with a note. Where it cannot, the plan
   is rejected with PlanIncompatible.

`make_plan` is the entry point the API calls. It tries the LLM, allows one
repair attempt, and falls back to the rule planner, so a question always gets
a plan, and the notes say which path produced it.
"""

from enum import Enum
from typing import Any

import openai
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.api.schemas import VisualizationRequest
from app.planner.llm_planner import (
    PlannerError,
    ask_llm,
    first_messages,
    make_client,
    parse_plan,
    repair_messages,
)
from app.planner.plan import (
    REQUEST_FIELD_TO_PLAN_FILTER,
    CompareGroup,
    Intent,
    PlanFilters,
    QueryPlan,
    TimeSpec,
)
from app.planner.rule_planner import rule_plan

# The same table plan.py defines, under the name this module's logic reads best with.
REQUEST_TO_PLAN_FIELD: dict[str, str] = REQUEST_FIELD_TO_PLAN_FILTER

PlanResult = tuple[QueryPlan, list[str], list[str]]  # (plan, overrides, notes)


class PlanIncompatible(ValueError):
    """The plan lacks something its intent needs, and it cannot be rebuilt."""


def display(value: Any) -> Any:
    """Enums (and lists of them) as plain values, so messages read well."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [display(item) for item in value]
    return value


# --------------------------------------------------------------------------
# 1. Merge the request's structured fields into the plan
# --------------------------------------------------------------------------
def merge_request_fields(plan: QueryPlan, request: VisualizationRequest, source: str) -> list[str]:
    """Copy request fields into plan.filters (in place). Returns override messages."""
    overrides: list[str] = []
    for request_field, plan_field in REQUEST_TO_PLAN_FIELD.items():
        new = getattr(request, request_field)
        if new is None:
            continue
        old = getattr(plan.filters, plan_field)
        if old is not None and old != new:
            overrides.append(
                f"{plan_field}: {source} said {display(old)!r}, request said {display(new)!r}; used {display(new)!r}"
            )
        setattr(plan.filters, plan_field, new)
    return overrides


# --------------------------------------------------------------------------
# 2. One small check per intent. Each edits the plan in place and returns a
#    note (or None), or raises PlanIncompatible.
# --------------------------------------------------------------------------
def check_network(plan: QueryPlan, request: VisualizationRequest) -> str | None:
    if plan.network is None:
        raise PlanIncompatible("network plan has no network.node_dims")
    return None


def check_histogram(plan: QueryPlan, request: VisualizationRequest) -> str | None:
    if plan.numeric_x is None:
        raise PlanIncompatible("histogram plan has no numeric_x")
    return None


def check_correlation(plan: QueryPlan, request: VisualizationRequest) -> str | None:
    if plan.numeric_x is None or plan.numeric_y is None:
        raise PlanIncompatible("correlation plan needs both numeric_x and numeric_y")
    return None


def check_trend(plan: QueryPlan, request: VisualizationRequest) -> str | None:
    if plan.time is not None:
        return None
    start, end = plan.filters.start_year, plan.filters.end_year
    plan.time = TimeSpec(start=start, end=end)
    if start is None and end is None:
        return "Trend window: no window given; using all years."
    return "Trend window taken from start_year/end_year."


def check_comparison(plan: QueryPlan, request: VisualizationRequest) -> str | None:
    if plan.compare_groups is not None and len(plan.compare_groups) >= 2:
        return None
    main = plan.filters.drug or plan.filters.condition
    names = ([main] if main else []) + (request.compare_with or [])
    if len(names) < 2:
        raise PlanIncompatible("comparison needs at least two things to compare")
    plan.compare_groups = [CompareGroup(label=name, filters=PlanFilters(drug=name)) for name in names]
    return "Comparison groups built from the main entity and compare_with: " + ", ".join(names) + "."


INTENT_CHECKS = {
    Intent.network: check_network,
    Intent.histogram: check_histogram,
    Intent.correlation: check_correlation,
    Intent.trend: check_trend,
    Intent.comparison: check_comparison,
}


def validate_and_merge(plan: QueryPlan, request: VisualizationRequest, source: str = "LLM") -> PlanResult:
    """Return (plan, overrides, notes). The inputs are never changed."""
    work = plan.model_copy(deep=True)
    overrides = merge_request_fields(work, request, source)

    notes: list[str] = []
    check = INTENT_CHECKS.get(work.intent)
    if check is not None:
        note = check(work, request)
        if note:
            notes.append(note)

    # Rebuild so every QueryPlan constraint runs again on the merged values.
    return QueryPlan.model_validate(work.model_dump()), overrides, notes


# --------------------------------------------------------------------------
# The orchestrator
# --------------------------------------------------------------------------
def with_rule_planner(request: VisualizationRequest, notes: list[str]) -> PlanResult:
    plan, overrides, plan_notes = validate_and_merge(rule_plan(request), request, source="rule planner")
    return plan, overrides, notes + plan_notes


async def plan_with_llm(request: VisualizationRequest, client: AsyncOpenAI) -> PlanResult:
    """Ask the model; if its plan is bad, show it the error and ask once more."""
    content = await ask_llm(client, first_messages(request))
    try:
        return validate_and_merge(parse_plan(content), request)
    except (ValidationError, PlanIncompatible) as error:
        repair = repair_messages(request, content, str(error)[:1000])
        return validate_and_merge(parse_plan(await ask_llm(client, repair)), request)


def short_reason(error: Exception) -> str:
    return f"{type(error).__name__}: {str(error).splitlines()[0][:100] if str(error) else ''}".rstrip(": ")


async def make_plan(request: VisualizationRequest, client: AsyncOpenAI | None = None) -> PlanResult:
    """The plan for a request: LLM first, rule planner if that does not work out."""
    client = client or make_client()
    if client is None:
        return with_rule_planner(request, ["No OpenAI key configured; used the rule planner."])
    try:
        return await plan_with_llm(request, client)
    except (ValidationError, PlanIncompatible, PlannerError, openai.OpenAIError, TimeoutError) as error:
        return with_rule_planner(request, [f"LLM planner failed ({short_reason(error)}); used the rule planner."])
