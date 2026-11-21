"""Turn a plan plus an operator result into a VizSpec.

Everything here is a lookup table or a string template. There is no LLM in
this file and no arithmetic: the numbers arrived already computed from the
operators, and this decides only how to draw them and what to call them.

The one judgement it makes is whether to honour a caller's `viz_hint`. A hint
is accepted only when it actually fits the data; otherwise the default wins
and the reason is written into the notes, because silently ignoring a request
is worse than refusing it out loud.
"""

from pydantic import BaseModel, TypeAdapter

from app.analytics.operators import OperatorResult
from app.api.schemas import VizSpec, VizType
from app.ctgov.normalize import PHASE_ORDER
from app.planner.plan import DIMENSIONS, Dimension, DimensionKind, Intent, QueryPlan

SPEC = TypeAdapter(VizSpec)

# --------------------------------------------------------------------------
# The rule table
# --------------------------------------------------------------------------
# One chart per intent. This is the whole "which visualization" decision, and
# it is a dict rather than a prompt so it cannot drift between runs.
INTENT_TO_VIZ: dict[Intent, VizType] = {
    Intent.count: VizType.stat,
    Intent.list: VizType.table,
    Intent.distribution: VizType.bar_chart,
    Intent.trend: VizType.time_series,
    Intent.comparison: VizType.grouped_bar_chart,
    Intent.geo: VizType.bar_chart,
    Intent.network: VizType.network_graph,
    Intent.histogram: VizType.histogram,
    Intent.correlation: VizType.scatter_plot,
}

# A viz_hint is honoured only if it appears here for the plan's intent.
# `table` is allowed almost everywhere because any list of rows can be shown
# as a table; the reverse is not true, which is why stat and network have no
# alternatives at all - a single number has no rows, and a graph is not rows.
ALTERNATIVE_VIZ: dict[Intent, set[VizType]] = {
    Intent.count: set(),
    Intent.list: set(),
    Intent.distribution: {VizType.table},
    Intent.trend: {VizType.table, VizType.bar_chart},
    Intent.comparison: {VizType.table},
    Intent.geo: {VizType.table},
    Intent.network: set(),
    Intent.histogram: {VizType.table},
    Intent.correlation: {VizType.table},
}

# Charts that are a number or a list rather than a picture. The response says
# so in meta.chart_recommended, so a frontend need not invent a chart for them.
NOT_A_CHART = {VizType.stat, VizType.table}

# --------------------------------------------------------------------------
# Naming things
# --------------------------------------------------------------------------

DIMENSION_TITLES: dict[Dimension, str] = {
    Dimension.phase: "Phase",
    Dimension.status: "Status",
    Dimension.start_year: "Start year",
    Dimension.sponsor: "Sponsor",
    Dimension.sponsor_class: "Sponsor type",
    Dimension.country: "Country",
    Dimension.intervention_type: "Intervention type",
    Dimension.drug: "Drug",
    Dimension.condition: "Condition",
    Dimension.study_type: "Study type",
    Dimension.enrollment: "Participants",
    Dimension.duration_months: "Duration (months)",
}

# How a dimension reads inside a sentence, as opposed to on an axis label.
MEASURE_NOUNS: dict[Dimension, str] = {
    Dimension.enrollment: "enrollment",
    Dimension.duration_months: "duration",
}

# Plurals for network titles: "sponsors and drugs", not "Sponsor and Drug".
DIMENSION_PLURALS: dict[Dimension, str] = {
    Dimension.sponsor: "sponsors",
    Dimension.drug: "drugs",
    Dimension.condition: "conditions",
    Dimension.country: "countries",
    Dimension.intervention_type: "intervention types",
    Dimension.sponsor_class: "sponsor types",
    Dimension.phase: "phases",
    Dimension.status: "statuses",
    Dimension.study_type: "study types",
}

DIMENSION_UNITS: dict[Dimension, str] = {
    Dimension.enrollment: "participants",
    Dimension.duration_months: "months",
}

TRIAL_COUNT_UNIT = "trials"


def channel_type(dim: Dimension) -> str:
    """How a frontend should treat this dimension's values.

    Derived from the catalog's `kind`, with phase as the single exception: it
    is categorical but genuinely ordered, so it is ordinal rather than nominal
    and a frontend will keep Phase 1 before Phase 2.
    """
    if dim is Dimension.phase:
        return "ordinal"
    kind = DIMENSIONS[dim].kind
    if kind is DimensionKind.numeric:
        return "quantitative"
    if kind is DimensionKind.temporal:
        return "temporal"
    return "nominal"


def channel(dim: Dimension, sort: str | list[str] | None = None) -> dict:
    """An encoding channel for a dimension, named after the operator's field."""
    return {
        "field": dim.value,
        "type": channel_type(dim),
        "title": DIMENSION_TITLES[dim],
        "sort": sort,
        "unit": DIMENSION_UNITS.get(dim),
    }


def count_channel(field: str = "trial_count", title: str = "Trials") -> dict:
    return {
        "field": field,
        "type": "quantitative",
        "title": title,
        "unit": TRIAL_COUNT_UNIT,
    }


def category_sort(dim: Dimension) -> str | list[str]:
    """How to order a category axis.

    Phase gets its explicit chart order, since neither alphabetical nor
    by-count puts "Phase 1/2" between 1 and 2 or "Not applicable" last.
    Everything else is biggest-first, which is how a bar chart is read.
    """
    if dim is Dimension.phase:
        return PHASE_ORDER
    return "-trial_count"


# --------------------------------------------------------------------------
# Titles
# --------------------------------------------------------------------------
# Built from the plan's filters by a template. No LLM: a title is a claim
# about what the chart shows, so it has to follow from the filters mechanically.


def capitalize_entity(text: str) -> str:
    """"breast cancer" -> "Breast cancer", "alzheimer" -> "Alzheimer".

    Only the first letter: the rest is left alone so "BRCA1" and "Pfizer Inc"
    survive, which str.title() would mangle into "Brca1" and "Pfizer Inc".
    """
    return text[0].upper() + text[1:] if text else text


def subject_phrase(plan: QueryPlan) -> str:
    """What the trials in this chart have in common, e.g. "recruiting
    Pembrolizumab trials in breast cancer".

    The entity the question is about leads the phrase, so a title reads
    "Breast cancer trials by phase" rather than "Trials in breast cancer by
    phase". When both a drug and a condition are filtered the drug leads,
    because "Pembrolizumab trials in breast cancer" is how people say it.
    """
    filters = plan.filters
    parts = []

    if filters.status and len(filters.status) == 1:
        parts.append(filters.status[0].value.replace("_", " ").lower())
    if filters.phase and len(filters.phase) == 1:
        parts.append(filters.phase[0].value.replace("PHASE", "phase ").lower().strip())

    lead = filters.drug or filters.condition
    if lead:
        parts.append(capitalize_entity(lead))

    subject = " ".join(parts + ["trials"])

    if filters.drug and filters.condition:
        subject += f" in {filters.condition}"
    if filters.sponsor:
        subject += f" from {filters.sponsor}"
    if filters.country:
        subject += f" in {filters.country}"
    return subject


def window_phrase(plan: QueryPlan) -> str:
    """", 2015-2024" or " since 2015", or nothing."""
    start = (plan.time.start if plan.time else None) or plan.filters.start_year
    end = (plan.time.end if plan.time else None) or plan.filters.end_year
    if start and end:
        return f", {start}-{end}"
    if start:
        return f" since {start}"
    if end:
        return f" up to {end}"
    return ""


def build_title(plan: QueryPlan) -> str:
    """A sentence describing exactly what was counted."""
    subject = subject_phrase(plan)
    window = window_phrase(plan)
    group = DIMENSION_TITLES[plan.group_by].lower() if plan.group_by else ""

    if plan.intent is Intent.count:
        title = f"Number of {subject}{window}"
    elif plan.intent is Intent.list:
        # No .capitalize() here: it lowercases the rest of the string, which
        # would turn "BRCA1" into "Brca1". The first letter is raised below.
        title = f"{subject}{window}"
    elif plan.intent is Intent.trend:
        title = f"{subject} per year{window}"
    elif plan.intent is Intent.comparison:
        labels = [g.label for g in plan.compare_groups or []]
        versus = " vs ".join(labels)
        title = f"{subject} by {group}: {versus}{window}" if versus else f"{subject} by {group}"
    elif plan.intent is Intent.network:
        left, right = plan.network.node_dims
        if left is right:
            # Co-occurrence: the same kind of thing linked to itself.
            title = f"{DIMENSION_PLURALS[left]} studied together in {subject}{window}"
        else:
            title = (
                f"{DIMENSION_PLURALS[left]} and {DIMENSION_PLURALS[right]} "
                f"in {subject}{window}"
            )
    elif plan.intent is Intent.histogram:
        measure = MEASURE_NOUNS[plan.numeric_x]
        title = f"Distribution of {measure} across {subject}{window}"
    elif plan.intent is Intent.correlation:
        x = MEASURE_NOUNS[plan.numeric_x]
        y = MEASURE_NOUNS[plan.numeric_y]
        title = f"{x} against {y} in {subject}{window}"
    else:  # distribution and geo
        title = f"{subject} by {group}{window}"

    return title[0].upper() + title[1:] if title else "Trials"


# --------------------------------------------------------------------------
# What the builder returns
# --------------------------------------------------------------------------


def _dropped_note(dropped: int, viz_type: VizType) -> str:
    """Say how many trials are not in the chart, and why.

    Pluralised because "1 trials were left out" reads like a bug, and worded
    per chart because a trial missing from a network is missing for a
    different reason than one missing from a bar chart.
    """
    noun = "trial was" if dropped == 1 else "trials were"
    if viz_type is VizType.network_graph:
        reason = "they had fewer than two things to link, so they form no pair"
        if dropped == 1:
            reason = "it had fewer than two things to link, so it forms no pair"
        return f"{dropped} {noun} left out because {reason}."
    reason = "they had no value for the field being measured"
    if dropped == 1:
        reason = "it had no value for the field being measured"
    return f"{dropped} {noun} left out because {reason}."


class BuiltViz(BaseModel):
    """The spec plus the two things meta and notes need from this stage."""

    spec: VizSpec
    chart_recommended: bool
    rationale: str  # why this chart
    notes: list[str] = []  # anything the caller should know, e.g. a refused hint


# --------------------------------------------------------------------------
# One builder per viz type
# --------------------------------------------------------------------------


def _stat(plan: QueryPlan, result: OperatorResult, title: str) -> dict:
    return {
        "type": "stat",
        "title": title,
        "encoding": {"y": count_channel()},
        "data": result.rows,
    }


def _table(plan: QueryPlan, result: OperatorResult, title: str) -> dict:
    """Columns come from the operator when it named them, else from the row
    keys, so a table of any operator's output is still well described."""
    columns = result.extra.get("columns") or (list(result.rows[0]) if result.rows else ["nct_id"])
    return {
        "type": "table",
        "title": title,
        "encoding": {},
        "columns": [
            {
                "field": name,
                "type": "quantitative" if name in ("trial_count", "count") else "nominal",
                "title": name.replace("_", " ").capitalize(),
            }
            for name in columns
        ],
        "data": result.rows,
        "options": {"shown": result.extra.get("shown"), "available": result.extra.get("available")},
    }


def _category_dim(plan: QueryPlan) -> Dimension:
    """The dimension on the category axis, whatever the intent calls it."""
    return plan.group_by or Dimension.phase


def _bar(plan: QueryPlan, result: OperatorResult, title: str) -> dict:
    """One bar per category. Also the geo chart, drawn horizontally because
    country names are long and unreadable rotated under a vertical axis."""
    dim = _category_dim(plan)
    return {
        "type": "bar_chart",
        "title": title,
        "encoding": {"x": channel(dim, sort=category_sort(dim)), "y": count_channel()},
        "data": result.rows,
        "options": {"horizontal": plan.intent is Intent.geo},
    }


def _grouped_bar(plan: QueryPlan, result: OperatorResult, title: str) -> dict:
    dim = _category_dim(plan)
    return {
        "type": "grouped_bar_chart",
        "title": title,
        "encoding": {
            "x": channel(dim, sort=category_sort(dim)),
            "y": count_channel(),
            "series": {"field": "group", "type": "nominal", "title": "Group"},
        },
        "data": result.rows,
    }


def _time_series(plan: QueryPlan, result: OperatorResult, title: str) -> dict:
    """Time ascending, always. A trend read right to left is not a trend."""
    encoding = {
        "x": {"field": "year", "type": "temporal", "title": "Start year", "sort": "year"},
        "y": count_channel(),
    }
    if plan.series_by:
        encoding["series"] = {
            "field": "series",
            "type": "nominal",
            "title": DIMENSION_TITLES[plan.series_by],
        }
    return {"type": "time_series", "title": title, "encoding": encoding, "data": result.rows}


def _histogram(plan: QueryPlan, result: OperatorResult, title: str) -> dict:
    dim = plan.numeric_x or Dimension.enrollment
    return {
        "type": "histogram",
        "title": title,
        "encoding": {
            "x": {
                "field": "bin_start",
                "type": "quantitative",
                "title": DIMENSION_TITLES[dim],
                "unit": DIMENSION_UNITS.get(dim),
            },
            "y": count_channel("count", "Trials"),
        },
        "data": result.rows,
        # The frontend needs to know to use a log axis, or the bars will be
        # drawn at even widths and misrepresent the distribution.
        "options": {"log_scale": result.extra.get("log_scale", False)},
    }


def _scatter(plan: QueryPlan, result: OperatorResult, title: str) -> dict:
    x = plan.numeric_x or Dimension.enrollment
    y = plan.numeric_y or Dimension.duration_months
    encoding = {"x": channel(x), "y": channel(y)}
    if any("color" in row for row in result.rows):
        encoding["color"] = {"field": "color", "type": "nominal", "title": "Group"}
    return {"type": "scatter_plot", "title": title, "encoding": encoding, "data": result.rows}


def _network(plan: QueryPlan, result: OperatorResult, title: str) -> dict:
    """`color` and `size` name real node fields; the nodes and edges are the
    data, so there is no channel pointing at them."""
    return {
        "type": "network_graph",
        "title": title,
        "encoding": {
            "color": {"field": "type", "type": "nominal", "title": "Node kind"},
            "size": {"field": "weight", "type": "quantitative", "title": "Trials", "unit": TRIAL_COUNT_UNIT},
        },
        "data": {
            "nodes": result.extra.get("nodes", []),
            "edges": result.extra.get("edges", []),
        },
        "options": {"min_edge_weight": result.extra.get("min_edge_weight")},
    }


BUILDERS = {
    VizType.stat: _stat,
    VizType.table: _table,
    VizType.bar_chart: _bar,
    VizType.grouped_bar_chart: _grouped_bar,
    VizType.time_series: _time_series,
    VizType.histogram: _histogram,
    VizType.scatter_plot: _scatter,
    VizType.network_graph: _network,
}

RATIONALES: dict[VizType, str] = {
    VizType.stat: "The question asks for a single number, so there is nothing to plot.",
    VizType.table: "The question asks which trials, so the trials themselves are the answer.",
    VizType.bar_chart: "One bar per category compares counts across a categorical dimension.",
    VizType.grouped_bar_chart: "Grouped bars put the compared sides side by side in each category.",
    VizType.time_series: "A line over years shows how the count changes, including years with none.",
    VizType.histogram: "Binning a numeric field shows the shape of its distribution.",
    VizType.scatter_plot: "One point per trial shows whether the two measures move together.",
    VizType.network_graph: "A graph shows which entities appear alongside which, and how often.",
}


# --------------------------------------------------------------------------
# The entry point
# --------------------------------------------------------------------------


def choose_viz(intent: Intent, viz_hint: VizType | None = None) -> tuple[VizType, list[str]]:
    """The default chart for the intent, or an accepted hint, plus any note.

    The hint lives on the request, not the plan - it is the caller's
    preference, not the LLM's - so it is passed in rather than read off the
    plan. A refused hint is reported rather than ignored: the caller asked for
    something and is entitled to know why they did not get it.
    """
    default = INTENT_TO_VIZ[intent]
    if viz_hint is None or viz_hint is default:
        return default, []

    if viz_hint in ALTERNATIVE_VIZ[intent]:
        return viz_hint, [
            f"Used the requested {viz_hint.value} instead of the default {default.value}."
        ]

    return default, [
        f"Ignored the requested {viz_hint.value}: a {intent.value} result cannot be drawn "
        f"that way, so a {default.value} was used instead."
    ]


def build_spec(plan: QueryPlan, result: OperatorResult, viz_hint: VizType | None = None) -> BuiltViz:
    """Plan + numbers -> a validated VizSpec.

    The spec is validated here rather than at the API boundary, so a builder
    that forgets a required channel fails at the point of the mistake.
    """
    viz_type, notes = choose_viz(plan.intent, viz_hint)
    title = build_title(plan)
    spec = SPEC.validate_python(BUILDERS[viz_type](plan, result, title))

    # An empty graph deserves an explanation rather than a blank canvas.
    if viz_type is VizType.network_graph and not spec.data.edges:
        before = result.extra.get("edges_before_threshold", 0)
        notes.append(
            f"No pair appeared in at least {result.extra.get('min_edge_weight')} of the trials "
            f"analysed, so the graph is empty ({before} pairs occurred less often than that). "
            "A larger max_studies or a lower min_edge_weight would show more."
        )

    if result.dropped:
        notes.append(_dropped_note(result.dropped, viz_type))

    return BuiltViz(
        spec=spec,
        chart_recommended=viz_type not in NOT_A_CHART,
        rationale=RATIONALES[viz_type],
        notes=notes,
    )
