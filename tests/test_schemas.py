"""Tests for the request and response contract in app/api/schemas.py.

Two things are checked:

- a hand-built, realistic example of every viz type validates, and
- the mistakes that would produce a misleading chart are rejected.

The second half matters more. A request field that silently does nothing, or a
chart missing the channel it needs to be read, both end up as a confidently
wrong answer rather than an error.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from app.api.schemas import (
    Meta,
    VisualizationRequest,
    VisualizationResponse,
    VizSpec,
    VizType,
)

SPEC = TypeAdapter(VizSpec)


def channel(field: str, type_: str, title: str) -> dict:
    """Shorthand for one encoding channel."""
    return {"field": field, "type": type_, "title": title}


# One realistic spec per viz type, as the viz builder in B3 will produce them.
VALID_SPECS: dict[VizType, dict] = {
    VizType.stat: {
        "type": "stat",
        "title": "Recruiting Alzheimer's disease trials",
        "encoding": {"y": channel("trial_count", "quantitative", "Trials")},
        "data": [{"trial_count": 412}],
    },
    VizType.table: {
        "type": "table",
        "title": "Pembrolizumab trials",
        "encoding": {},
        "columns": [
            channel("nct_id", "nominal", "NCT ID"),
            channel("phase_label", "ordinal", "Phase"),
            channel("start_year", "temporal", "Start year"),
        ],
        "data": [{"nct_id": "NCT03871959", "phase_label": "Phase 1", "start_year": 2019}],
    },
    VizType.bar_chart: {
        "type": "bar_chart",
        "title": "Breast cancer trials by phase",
        "encoding": {
            "x": channel("phase_label", "ordinal", "Phase"),
            "y": channel("trial_count", "quantitative", "Trials"),
        },
        "data": [{"phase_label": "Phase 3", "trial_count": 49}],
    },
    VizType.grouped_bar_chart: {
        "type": "grouped_bar_chart",
        "title": "Phases: pembrolizumab vs nivolumab",
        "encoding": {
            "x": channel("phase_label", "ordinal", "Phase"),
            "y": channel("trial_count", "quantitative", "Trials"),
            "series": channel("group", "nominal", "Drug"),
        },
        "data": [{"phase_label": "Phase 3", "group": "pembrolizumab", "trial_count": 31}],
    },
    VizType.time_series: {
        "type": "time_series",
        "title": "Pembrolizumab trials per year since 2015",
        "encoding": {
            "x": channel("year", "temporal", "Year"),
            "y": channel("trial_count", "quantitative", "Trials"),
        },
        "data": [{"year": 2015, "trial_count": 118}],
    },
    VizType.histogram: {
        "type": "histogram",
        "title": "Enrollment size of breast cancer trials",
        "encoding": {
            "x": channel("bin_start", "quantitative", "Participants"),
            "y": channel("count", "quantitative", "Trials"),
        },
        "data": [{"bin_start": 0, "bin_end": 50, "count": 37}],
    },
    VizType.scatter_plot: {
        "type": "scatter_plot",
        "title": "Enrollment against duration",
        "encoding": {
            "x": channel("enrollment", "quantitative", "Participants"),
            "y": channel("duration_months", "quantitative", "Months"),
        },
        "data": [{"nct_id": "NCT03871959", "enrollment": 54, "duration_months": 33.0}],
    },
    VizType.network_graph: {
        "type": "network_graph",
        "title": "Sponsors and drugs in NSCLC trials",
        "encoding": {
            "color": channel("type", "nominal", "Node kind"),
            "size": channel("weight", "quantitative", "Trials"),
        },
        "data": {
            "nodes": [
                {"id": "sponsor:merck", "label": "Merck", "type": "sponsor", "weight": 12},
                {"id": "drug:pembrolizumab", "label": "Pembrolizumab", "type": "drug", "weight": 12},
            ],
            "edges": [
                {"source": "sponsor:merck", "target": "drug:pembrolizumab", "weight": 12}
            ],
        },
    },
}


def test_every_viz_type_has_an_example():
    """Guard on the test file itself: a ninth viz type must bring an example."""
    assert set(VALID_SPECS) == set(VizType)


@pytest.mark.parametrize("viz_type", list(VALID_SPECS), ids=lambda t: t.value)
def test_valid_spec_validates(viz_type):
    spec = SPEC.validate_python(VALID_SPECS[viz_type])
    assert spec.type is viz_type
    assert spec.title


# --- specs that must be rejected ------------------------------------------
# Each entry is (label, broken spec). The label says what is wrong.

def _without_channel(viz_type: VizType, channel_name: str) -> dict:
    spec = {**VALID_SPECS[viz_type]}
    spec["encoding"] = {k: v for k, v in spec["encoding"].items() if k != channel_name}
    return spec


INVALID_SPECS = [
    # A missing channel means the chart cannot be read: no value axis, or no
    # idea what the bars are split by.
    ("bar_chart with no y", _without_channel(VizType.bar_chart, "y")),
    ("bar_chart with no x", _without_channel(VizType.bar_chart, "x")),
    ("grouped_bar_chart with no series", _without_channel(VizType.grouped_bar_chart, "series")),
    ("stat with no y", _without_channel(VizType.stat, "y")),
    ("network_graph with no size", _without_channel(VizType.network_graph, "size")),
    ("network_graph with no color", _without_channel(VizType.network_graph, "color")),
    # Wrong data shape, both directions.
    ("network data given as rows", {**VALID_SPECS[VizType.network_graph], "data": [{"a": 1}]}),
    (
        "bar_chart data given as a network",
        {**VALID_SPECS[VizType.bar_chart], "data": {"nodes": [], "edges": []}},
    ),
    # A chart we do not build.
    ("unknown viz type", {**VALID_SPECS[VizType.bar_chart], "type": "pie_chart"}),
    # A channel with an invented field type.
    (
        "unknown channel type",
        {
            **VALID_SPECS[VizType.bar_chart],
            "encoding": {
                "x": channel("phase_label", "categorical", "Phase"),
                "y": channel("trial_count", "quantitative", "Trials"),
            },
        },
    ),
    ("missing title", {k: v for k, v in VALID_SPECS[VizType.bar_chart].items() if k != "title"}),
    ("table with no columns", {k: v for k, v in VALID_SPECS[VizType.table].items() if k != "columns"}),
]


@pytest.mark.parametrize("label,spec", INVALID_SPECS, ids=[label for label, _ in INVALID_SPECS])
def test_invalid_spec_rejected(label, spec):
    with pytest.raises(ValidationError):
        SPEC.validate_python(spec)


def test_orphan_network_edge_rejected():
    """An edge must connect nodes that exist in the same graph.

    Without this a frontend has to check every edge's ids resolve before it
    can draw them, or it silently drops edges. Cheap to make impossible.
    """
    orphan = {
        **VALID_SPECS[VizType.network_graph],
        "data": {
            "nodes": [{"id": "sponsor:merck", "label": "Merck", "type": "sponsor", "weight": 1}],
            "edges": [{"source": "sponsor:merck", "target": "drug:nothing", "weight": 1}],
        },
    }
    with pytest.raises(ValidationError):
        SPEC.validate_python(orphan)


# --- the full response ----------------------------------------------------


def test_full_response_round_trips():
    response = VisualizationResponse(
        visualization=VALID_SPECS[VizType.bar_chart],
        meta={
            "filters_applied": {"query.cond": "breast cancer"},
            "plan": {"intent": "distribution", "group_by": "phase"},
            "total_matched": 17038,
            "studies_analyzed": 2000,
            "truncated": True,
            "chart_recommended": True,
            "viz_rationale": "A distribution over a categorical dimension reads best as bars.",
        },
        notes={
            "interpretation": "Breast cancer trials grouped by phase.",
            "assumptions": ["Trials with no phase are shown as 'Not applicable'."],
        },
    )
    assert response.meta.truncated is True  # 17038 matched, 2000 analysed
    assert response.meta.source == "clinicaltrials.gov/api/v2"
    assert response.notes.overrides == []
    # The response must survive a JSON round trip, since that is how it ships.
    assert VisualizationResponse.model_validate_json(response.model_dump_json()) == response


def test_schema_generation_covers_every_viz_type():
    """GET /v1/schema serves this, so it has to spell out all eight shapes."""
    schema = VisualizationResponse.model_json_schema()
    visualization = schema["properties"]["visualization"]
    assert len(visualization["oneOf"]) == len(VizType)
    assert visualization["discriminator"]["propertyName"] == "type"


# --- the request ----------------------------------------------------------


def test_minimal_request():
    request = VisualizationRequest(query="How many recruiting Alzheimer's trials are there?")
    assert request.max_studies == 2000
    assert request.phase is None


def test_full_request():
    request = VisualizationRequest(
        query="How has the number of pembrolizumab trials changed per year since 2015?",
        drug_name="pembrolizumab",
        phase=["PHASE3"],
        status=["RECRUITING"],
        start_year=2015,
        end_year=2024,
        compare_with=["nivolumab"],
        viz_hint="time_series",
        max_studies=500,
    )
    assert request.phase[0].value == "PHASE3"
    assert request.viz_hint is VizType.time_series


def test_request_strips_whitespace():
    assert VisualizationRequest(query="   padded question   ").query == "padded question"


INVALID_REQUESTS = [
    ("query too short", {"query": "hi"}),
    ("query missing", {}),
    # The important one: a near-miss field name must not be ignored. `drug`
    # instead of `drug_name` would otherwise be dropped and the answer would
    # come back unfiltered, which looks like a correct answer.
    ("misspelt field name", {"query": "a valid question", "drug": "pembrolizumab"}),
    ("unknown phase", {"query": "a valid question", "phase": ["PHASE9"]}),
    ("unknown status", {"query": "a valid question", "status": ["GOING_WELL"]}),
    ("unknown viz hint", {"query": "a valid question", "viz_hint": "pie_chart"}),
    ("year before 1990", {"query": "a valid question", "start_year": 1850}),
    ("year after 2035", {"query": "a valid question", "end_year": 2100}),
    ("start year after end year", {"query": "a valid question", "start_year": 2024, "end_year": 2015}),
    ("too many comparisons", {"query": "a valid question", "compare_with": ["a", "b", "c", "d", "e"]}),
    ("max_studies above the cap", {"query": "a valid question", "max_studies": 9999}),
    ("max_studies zero", {"query": "a valid question", "max_studies": 0}),
]


@pytest.mark.parametrize(
    "label,payload", INVALID_REQUESTS, ids=[label for label, _ in INVALID_REQUESTS]
)
def test_invalid_request_rejected(label, payload):
    with pytest.raises(ValidationError):
        VisualizationRequest(**payload)


def test_equal_start_and_end_year_allowed():
    """A single-year window is a legitimate question, not an off-by-one."""
    request = VisualizationRequest(query="a valid question", start_year=2020, end_year=2020)
    assert request.start_year == request.end_year == 2020


def test_truncated_flag_must_match_the_counts():
    """A response claiming nothing was left out while analysing 2000 of 17038
    trials would understate the answer. Make it impossible, both ways."""
    meta = {
        "filters_applied": {},
        "plan": {},
        "total_matched": 17038,
        "studies_analyzed": 2000,
        "truncated": False,  # the lie
        "chart_recommended": True,
        "viz_rationale": "x",
    }
    with pytest.raises(ValidationError):
        Meta(**meta)
    with pytest.raises(ValidationError):
        Meta(**{**meta, "total_matched": 2000, "truncated": True})
    # Consistent either way is fine.
    assert Meta(**{**meta, "truncated": True}).truncated is True
    assert Meta(**{**meta, "total_matched": 2000, "truncated": False}).truncated is False
