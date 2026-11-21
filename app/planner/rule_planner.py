"""The fallback planner: a QueryPlan built from keywords, with no LLM.

It is used when there is no OpenAI key, or when the LLM fails twice. It reads
the question with a handful of regular expressions and fills the same
QueryPlan the LLM would, so everything after the planner is identical.

Two rules keep it honest:

- Intent is the first cue that matches, checked in the order of INTENT_CUES.
  The order is the design: "how many" beats everything, "distribution" is only
  the default when nothing matched.
- Entities (drug, condition, ...) come ONLY from the request's own fields or
  from text the user put in quotes. A rule planner that guessed that
  "Alzheimer" in an unquoted sentence is a condition would sometimes be wrong,
  and the filter would silently return the wrong trials.

Whenever it had to guess (a default grouping, a quoted word classified as a
drug), it says so in `assumptions`, so the answer carries its own
"low confidence" label. It does no I/O and never raises: the plan it returns
always has the parts its intent needs.
"""

import re

from app.api.schemas import Phase, Status, VisualizationRequest
from app.planner.plan import (
    DIMENSIONS,
    CompareGroup,
    Dimension,
    DimensionKind,
    Intent,
    NetworkSpec,
    PlanFilters,
    QueryPlan,
    TimeSpec,
    is_numeric,
)

# Checked top to bottom; the first pattern that matches the lowercased question
# decides the intent. Comparison and distribution have extra logic in rule_plan.
INTENT_CUES: list[tuple[Intent, str]] = [
    (Intent.count, r"how many"),
    (Intent.list, r"\blist\b|which trials|show trials"),
    (Intent.network, r"network|co-occur|relationship|<->|together|combination"),
    (Intent.comparison, r" vs |versus|compare"),
    (Intent.trend, r"per year|over time|trend|since \d{4}|each year"),
    (Intent.geo, r"countr|geograph|\bwhere\b|location"),
    (Intent.histogram, r"distribution of (enrollment|size|duration)"),
]

# Words that point at a dimension. Longer phrases are tried first (see
# mentioned_dimensions), so "sponsor type" is sponsor_class, not sponsor.
DIMENSION_SYNONYMS: dict[Dimension, list[str]] = {
    Dimension.phase: ["phase"],
    Dimension.sponsor: ["sponsor", "company", "companies"],
    Dimension.sponsor_class: ["sponsor type", "sponsor class", "industry vs"],
    Dimension.country: ["country", "countries"],
    Dimension.drug: ["drug", "drugs", "intervention name"],
    Dimension.condition: ["condition", "disease", "indication"],
    Dimension.status: ["status", "recruiting status"],
    Dimension.intervention_type: ["intervention type"],
    Dimension.study_type: ["study type", "observational", "interventional"],
    Dimension.enrollment: ["enrollment", "size", "participants"],
    Dimension.duration_months: ["duration", "length"],
}

PHASE_BY_NUMBER: dict[str, Phase] = {
    "1": Phase.PHASE1, "i": Phase.PHASE1,
    "2": Phase.PHASE2, "ii": Phase.PHASE2,
    "3": Phase.PHASE3, "iii": Phase.PHASE3,
    "4": Phase.PHASE4, "iv": Phase.PHASE4,
}  # fmt: skip

# "not recruiting" and "not yet recruiting" are different statuses, so the
# lookbehinds keep them from being read as RECRUITING.
STATUS_CUES: list[tuple[Status, str]] = [
    (Status.RECRUITING, r"(?<!not )(?<!yet )recruiting"),
    (Status.COMPLETED, r"completed"),
    (Status.TERMINATED, r"terminated"),
]

PHASE_PATTERN = r"\bphase\s*(iv|iii|ii|i|[1-4])(?:\s*(?:/|-|and)\s*(iv|iii|ii|i|[1-4]))?\b"
YEAR = r"(199\d|20[0-3]\d)"
YEAR_WINDOWS: list[str] = [
    rf"since {YEAR}()",  # end group is empty: no end year
    rf"from {YEAR} to {YEAR}",
    rf"between {YEAR} and {YEAR}",
    rf"{YEAR}\s*-\s*{YEAR}",
]
QUOTED = re.compile(r'"([^"]+)"|(?<!\w)\'([^\']+)\'|“([^”]+)”')


def mentioned_dimensions(query: str) -> list[Dimension]:
    """Dimensions named in the text, in order of first appearance.

    Multi-word synonyms are matched first and their characters are used up, so
    the "sponsor" inside "sponsor type" cannot also count as a sponsor.
    """
    text = query.lower()
    pairs = [(word, dim) for dim, words in DIMENSION_SYNONYMS.items() for word in words]
    pairs.sort(key=lambda pair: -len(pair[0]))

    first_position: dict[Dimension, int] = {}
    for word, dim in pairs:
        for match in re.finditer(rf"\b{re.escape(word)}s?\b", text):
            start = match.start()
            text = text[:start] + "#" * (match.end() - start) + text[match.end() :]
            first_position[dim] = min(start, first_position.get(dim, start))
    return sorted(first_position, key=lambda dim: first_position[dim])


def categorical_dimensions(query: str) -> list[Dimension]:
    return [d for d in mentioned_dimensions(query) if DIMENSIONS[d].kind == DimensionKind.categorical]


def numeric_dimensions(query: str) -> list[Dimension]:
    return [d for d in mentioned_dimensions(query) if is_numeric(d)]


def find_intent(text: str) -> Intent | None:
    """The first intent whose cue appears in the lowercased question."""
    for intent, pattern in INTENT_CUES:
        if re.search(pattern, text):
            # "enrollment vs duration" is two numbers plotted together, not A vs B.
            if intent == Intent.comparison and len(numeric_dimensions(text)) >= 2:
                return Intent.correlation
            return intent
    return None


def find_phases(text: str) -> list[Phase] | None:
    phases: list[Phase] = []
    for match in re.finditer(PHASE_PATTERN, text):
        for number in match.groups():
            if number and PHASE_BY_NUMBER[number] not in phases:
                phases.append(PHASE_BY_NUMBER[number])
    return phases or None


def find_statuses(text: str) -> list[Status] | None:
    found = [status for status, pattern in STATUS_CUES if re.search(pattern, text)]
    return found or None


def find_year_window(text: str) -> tuple[int | None, int | None]:
    """(start, end) from "since 2015", "from 2010 to 2020", "2010-2020", ..."""
    for pattern in YEAR_WINDOWS:
        match = re.search(pattern, text)
        if match:
            start, end = match.groups()
            return int(start), int(end) if end else None
    return None, None


def find_quoted(query: str) -> list[tuple[str, str]]:
    """Quoted strings as (name, kind). Kind is "condition" when the quote
    follows "in " or "for ", otherwise "drug"."""
    found: list[tuple[str, str]] = []
    for match in QUOTED.finditer(query):
        name = next(group for group in match.groups() if group).strip()
        before = query[: match.start()].lower()
        kind = "condition" if before.endswith(("in ", "for ")) else "drug"
        found.append((name, kind))
    return found


def group_for(name: str, kind: str = "drug") -> CompareGroup:
    return CompareGroup(label=name, filters=PlanFilters(**{kind: name}))


def build_compare_groups(request: VisualizationRequest, quoted: list[tuple[str, str]]) -> list[CompareGroup]:
    """Each quoted name is a group; with compare_with, the main entity plus
    those names are the groups instead."""
    if request.compare_with:
        main = request.drug_name or (quoted[0][0] if quoted else None)
        names = ([main] if main else []) + request.compare_with
        return [group_for(name) for name in names]
    return [group_for(name, kind) for name, kind in quoted]


def rule_plan(request: VisualizationRequest) -> QueryPlan:
    """Read the question with keywords and return a QueryPlan. Pure, no I/O."""
    text = request.query.lower()
    assumptions: list[str] = []

    intent = find_intent(text)
    if intent is None:
        intent = Intent.distribution
        assumptions.append("Rule planner: no intent keyword found; treated as a distribution (low confidence).")

    start_year, end_year = find_year_window(text)
    quoted = find_quoted(request.query)
    groups = build_compare_groups(request, quoted) if intent == Intent.comparison else []
    if intent == Intent.comparison and len(groups) < 2:
        intent = Intent.distribution
        groups = []
        assumptions.append("Rule planner: a comparison needs two named things in quotes; showed a distribution instead (low confidence).")

    filters = PlanFilters(
        drug=request.drug_name,
        condition=request.condition,
        sponsor=request.sponsor,
        country=request.country,
        phase=find_phases(text),
        status=find_statuses(text),
        start_year=start_year,
        end_year=end_year,
    )
    if intent != Intent.comparison:
        # Quoted names are filters, but only where the request did not already set one.
        for name, kind in quoted:
            if getattr(filters, kind) is None:
                setattr(filters, kind, name)
                assumptions.append(f'Rule planner: read "{name}" as a {kind} (low confidence).')

    plan = QueryPlan(intent=intent, filters=filters, interpretation="")
    categorical = categorical_dimensions(text)
    numeric = numeric_dimensions(text)

    if intent in (Intent.distribution, Intent.comparison):
        if categorical:
            plan.group_by = categorical[0]
        else:
            plan.group_by = Dimension.phase
            assumptions.append("Rule planner: no grouping named; defaulted to phase (low confidence).")
    if intent == Intent.geo:
        plan.group_by = Dimension.country
    if intent == Intent.comparison:
        plan.compare_groups = groups
    if intent == Intent.trend:
        window_start = start_year or request.start_year
        window_end = end_year or request.end_year
        plan.time = TimeSpec(start=window_start, end=window_end)
    if intent == Intent.network:
        plan.network = NetworkSpec(node_dims=choose_node_dims(categorical, assumptions))
    if intent == Intent.histogram:
        plan.numeric_x = numeric[0] if numeric else Dimension.enrollment
    if intent == Intent.correlation:
        plan.numeric_x, plan.numeric_y = numeric[0], numeric[1]

    plan.interpretation = describe(plan)
    plan.assumptions = assumptions[:10]
    return QueryPlan.model_validate(plan.model_dump())


def choose_node_dims(categorical: list[Dimension], assumptions: list[str]) -> list[Dimension]:
    if len(categorical) >= 2:
        return categorical[:2]
    if len(categorical) == 1:
        assumptions.append(f"Rule planner: one dimension named; linked {categorical[0].value} with itself (low confidence).")
        return [categorical[0], categorical[0]]
    assumptions.append("Rule planner: no dimensions named; defaulted to sponsor and drug (low confidence).")
    return [Dimension.sponsor, Dimension.drug]


def describe(plan: QueryPlan) -> str:
    """One short sentence saying what the rule planner understood."""
    sentence = f"Rule-based reading: {plan.intent.value} of trials"
    if plan.group_by:
        sentence += f" by {plan.group_by.value}"
    if plan.compare_groups:
        sentence += " for " + " vs ".join(group.label for group in plan.compare_groups)
    for name in ("drug", "condition", "sponsor", "country"):
        value = getattr(plan.filters, name)
        if value:
            sentence += f", {name} {value}"
    return sentence + "."
