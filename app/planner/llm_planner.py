"""The LLM planner: asks an OpenAI model to fill in a QueryPlan.

The model never answers the question. It only chooses an intent, a grouping,
some filters and a sentence of interpretation. Every number in the final
response is computed later by deterministic code.

Three small pieces, deliberately separate so the validator can orchestrate
retries without this file knowing about them:

- `ask_llm` sends messages and returns the raw JSON text,
- `parse_plan` turns that text into a QueryPlan (and raises if it is invalid),
- `llm_plan` is the two of them together for the simple, first-attempt case.

OpenAI's strict structured-output mode rejects some JSON-schema keywords that
our QueryPlan uses (maxLength, minimum, default, ...). So the model is shown a
copy of the schema with those keywords removed (`llm_schema`), and its answer
is then checked against the REAL QueryPlan, which applies every constraint.
"""

from typing import Any

from openai import AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema

from app.api.schemas import VisualizationRequest
from app.planner.plan import (
    DIMENSIONS,
    CompareGroup,
    Dimension,
    Intent,
    NetworkSpec,
    PlanFilters,
    QueryPlan,
    TimeSpec,
)
from app.settings import settings
from app.viz.builder import INTENT_TO_VIZ

# Keywords strict mode refuses. The real QueryPlan re-checks all of them.
UNSUPPORTED_KEYWORDS = {"default", "minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems", "pattern"}

PHASE_BUCKETS = ["Early Phase 1", "Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Phase 4", "Not applicable"]


class PlannerError(Exception):
    """The model refused, or returned nothing usable."""


def strip_keywords(node: Any) -> Any:
    """A copy of a JSON schema with UNSUPPORTED_KEYWORDS removed at every level."""
    if isinstance(node, dict):
        return {key: strip_keywords(value) for key, value in node.items() if key not in UNSUPPORTED_KEYWORDS}
    if isinstance(node, list):
        return [strip_keywords(item) for item in node]
    return node


def llm_schema() -> dict[str, Any]:
    """The QueryPlan JSON schema in the form OpenAI strict mode accepts."""
    return strip_keywords(to_strict_json_schema(QueryPlan))


LLM_SCHEMA = llm_schema()


# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------
# Each example is built as a QueryPlan, not typed as JSON, so a typo in an
# example fails at import time instead of teaching the model a bad plan.
def few_shot_examples() -> list[tuple[str, QueryPlan]]:
    return [
        (
            "How many recruiting trials are there for lung cancer?",
            QueryPlan(
                intent=Intent.count,
                filters=PlanFilters(condition="lung cancer", status=["RECRUITING"]),
                interpretation="Count of recruiting lung cancer trials.",
            ),
        ),
        (
            "List phase 3 trials for melanoma",
            QueryPlan(
                intent=Intent.list,
                filters=PlanFilters(condition="melanoma", phase=["PHASE3"]),
                interpretation="Phase 3 melanoma trials, shown as a table.",
            ),
        ),
        (
            "What phases are diabetes trials in?",
            QueryPlan(
                intent=Intent.distribution,
                filters=PlanFilters(condition="diabetes"),
                group_by=Dimension.phase,
                interpretation="Diabetes trials grouped by phase.",
            ),
        ),
        (
            "How has the number of trials for pembrolizumab changed since 2015?",
            QueryPlan(
                intent=Intent.trend,
                filters=PlanFilters(drug="pembrolizumab", start_year=2015),
                time=TimeSpec(start=2015),
                interpretation="Pembrolizumab trials started per year since 2015.",
            ),
        ),
        (
            "Compare pembrolizumab and nivolumab trials by phase",
            QueryPlan(
                intent=Intent.comparison,
                group_by=Dimension.phase,
                compare_groups=[
                    CompareGroup(label="pembrolizumab", filters=PlanFilters(drug="pembrolizumab")),
                    CompareGroup(label="nivolumab", filters=PlanFilters(drug="nivolumab")),
                ],
                interpretation="Phase breakdown of pembrolizumab trials next to nivolumab trials.",
            ),
        ),
        (
            "Which countries run the most breast cancer trials?",
            QueryPlan(
                intent=Intent.geo,
                filters=PlanFilters(condition="breast cancer"),
                group_by=Dimension.country,
                interpretation="Breast cancer trials counted per country.",
            ),
        ),
        (
            "Which sponsors work on which drugs in Alzheimer's disease?",
            QueryPlan(
                intent=Intent.network,
                filters=PlanFilters(condition="Alzheimer's disease"),
                network=NetworkSpec(node_dims=[Dimension.sponsor, Dimension.drug]),
                interpretation="Network linking sponsors to the drugs they test in Alzheimer's disease.",
                assumptions=["Used the default minimum edge weight of 2."],
            ),
        ),
        (
            "What is the distribution of enrollment size in asthma trials?",
            QueryPlan(
                intent=Intent.histogram,
                filters=PlanFilters(condition="asthma"),
                numeric_x=Dimension.enrollment,
                interpretation="Histogram of enrollment in asthma trials.",
            ),
        ),
        (
            "Is enrollment related to trial duration?",
            QueryPlan(
                intent=Intent.correlation,
                numeric_x=Dimension.enrollment,
                numeric_y=Dimension.duration_months,
                interpretation="Scatter of enrollment against duration in months.",
            ),
        ),
    ]


def build_system_prompt() -> str:
    """Everything the model needs to know, as one string."""
    lines = [
        "You turn a question about clinical trials into a QueryPlan (JSON).",
        "You only choose how to answer. Another program fetches the data and computes every number.",
        "",
        "Dimensions you may use:",
    ]
    for dim, info in DIMENSIONS.items():
        lines.append(f"- {dim.value} ({info.kind.value}): {info.description}")

    lines += ["", "Each intent produces one chart type:"]
    for intent, viz in INTENT_TO_VIZ.items():
        lines.append(f"- {intent.value} -> {viz.value}")

    lines += [
        "",
        "Phases are reported in eight buckets: " + ", ".join(PHASE_BUCKETS) + ".",
        "In filters.phase use the API enum values: EARLY_PHASE1, PHASE1, PHASE2, PHASE3, PHASE4, NA.",
        "",
        "Rules:",
        "- Use only the enum values listed above.",
        "- Copy entity names (drugs, conditions, sponsors, countries) exactly as written in the question.",
        "- Never output numbers, data, counts or citations. Only fill the plan.",
        "- Put anything you were unsure about in assumptions.",
        "",
        "Examples (question -> plan):",
    ]
    for question, plan in few_shot_examples():
        lines.append(f"Q: {question}")
        lines.append(f"A: {plan.model_dump_json(exclude_none=True)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Calling the model
# --------------------------------------------------------------------------
def make_client() -> AsyncOpenAI | None:
    """A client, or None when no API key is configured (rule planner is used)."""
    if not settings.openai_api_key:
        return None
    return AsyncOpenAI(api_key=settings.openai_api_key)


def first_messages(request: VisualizationRequest) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": request.model_dump_json(exclude_none=True)},
    ]


def repair_messages(request: VisualizationRequest, previous: str, error: str) -> list[dict[str, str]]:
    """The first conversation plus the bad answer and what was wrong with it."""
    return first_messages(request) + [
        {"role": "assistant", "content": previous},
        {"role": "user", "content": f"That plan was invalid: {error}. Return a corrected plan."},
    ]


async def ask_llm(client: AsyncOpenAI, messages: list[dict[str, str]]) -> str:
    """Send the messages and return the model's raw JSON text."""
    response = await client.chat.completions.create(
        model=settings.planner_model,
        temperature=0,
        timeout=20,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "QueryPlan", "strict": True, "schema": LLM_SCHEMA},
        },
    )
    message = response.choices[0].message
    if message.refusal:
        raise PlannerError(f"model refused: {message.refusal}")
    if not message.content:
        raise PlannerError("model returned no content")
    return message.content


def parse_plan(content: str) -> QueryPlan:
    """Check the raw JSON against the real QueryPlan. Raises ValidationError."""
    return QueryPlan.model_validate_json(content)


async def llm_plan(
    request: VisualizationRequest,
    client: AsyncOpenAI | None = None,
    repair_error: str | None = None,
    previous: str | None = None,
) -> QueryPlan:
    """Ask once and parse. Give `repair_error` and `previous` to ask for a fix."""
    client = client or make_client()
    if client is None:
        raise PlannerError("no OpenAI API key configured")
    if repair_error is not None and previous is not None:
        messages = repair_messages(request, previous, repair_error)
    else:
        messages = first_messages(request)
    return parse_plan(await ask_llm(client, messages))
