"""Tests for the plan grammar in app/planner/plan.py.

The plan is the only thing the LLM produces, so this file is really a test of
the boundary itself: a sensible plan for each of the nine intents goes through,
and anything outside the grammar - an invented dimension, an extra key, a
numeric axis on a categorical field - is refused.

The extra-key tests are the ones worth reading. They are what stop a model
"helpfully" returning an answer alongside the plan.
"""

import json

import pytest
from pydantic import ValidationError

from app.planner.plan import (
    DIMENSIONS,
    Dimension,
    DimensionKind,
    Intent,
    Metric,
    QueryPlan,
    is_multi_valued,
    is_numeric,
)

# --- the catalog ----------------------------------------------------------


def test_catalog_covers_every_dimension():
    """A new Dimension without a catalog entry would crash the operators."""
    assert set(DIMENSIONS) == set(Dimension)


def test_catalog_entries_are_complete():
    for dim, info in DIMENSIONS.items():
        assert info.source_path, f"{dim.value} has no source path"
        assert info.description.endswith("."), f"{dim.value} description is not a sentence"


def test_numeric_dimensions():
    """Only these two can be binned or put on a continuous axis."""
    numeric = {d for d in Dimension if is_numeric(d)}
    assert numeric == {Dimension.enrollment, Dimension.duration_months}


def test_multi_valued_dimensions():
    """One trial can contribute several values to each of these, so a count by
    one of them sums to more than the number of trials."""
    multi = {d for d in Dimension if is_multi_valued(d)}
    assert multi == {
        Dimension.country,
        Dimension.intervention_type,
        Dimension.drug,
        Dimension.condition,
    }


def test_phase_is_not_multi_valued():
    """Deliberate, and the one that looks wrong at first glance.

    The API returns a list like ["PHASE1", "PHASE2"], but the data rules fold
    that into the single bucket "Phase 1/2", so every trial lands in exactly
    one phase bucket and phase counts do sum to the trial total.
    """
    assert is_multi_valued(Dimension.phase) is False


def test_start_year_is_the_only_temporal_dimension():
    temporal = {d for d in Dimension if DIMENSIONS[d].kind is DimensionKind.temporal}
    assert temporal == {Dimension.start_year}


# --- one plan per intent --------------------------------------------------
# Written the way the planner should fill them for these real questions.

VALID_PLANS: dict[Intent, dict] = {
    Intent.count: {
        "intent": "count",
        "filters": {"condition": "Alzheimer's disease", "status": ["RECRUITING"]},
        "interpretation": "Count trials for Alzheimer's disease that are currently recruiting.",
    },
    Intent.list: {
        "intent": "list",
        "filters": {"drug": "pembrolizumab", "phase": ["PHASE3"]},
        "top_k": 50,
        "interpretation": "List Phase 3 pembrolizumab trials.",
    },
    Intent.distribution: {
        "intent": "distribution",
        "filters": {"condition": "breast cancer"},
        "group_by": "phase",
        "interpretation": "Show how breast cancer trials are spread across phases.",
    },
    Intent.trend: {
        "intent": "trend",
        "filters": {"drug": "pembrolizumab", "start_year": 2015},
        "time": {"granularity": "year", "start": 2015},
        "interpretation": "Count pembrolizumab trials started in each year since 2015.",
    },
    Intent.comparison: {
        "intent": "comparison",
        "group_by": "phase",
        "compare_groups": [
            {"label": "pembrolizumab", "filters": {"drug": "pembrolizumab"}},
            {"label": "nivolumab", "filters": {"drug": "nivolumab"}},
        ],
        "interpretation": "Compare the phase mix of pembrolizumab and nivolumab trials.",
    },
    Intent.geo: {
        "intent": "geo",
        "filters": {"condition": "diabetes", "status": ["RECRUITING"]},
        "group_by": "country",
        "top_k": 15,
        "interpretation": "Rank countries by their number of recruiting diabetes trials.",
        "assumptions": ["A trial running in several countries is counted once per country."],
    },
    Intent.network: {
        "intent": "network",
        "filters": {"condition": "non-small cell lung cancer"},
        "network": {"node_dims": ["sponsor", "drug"], "min_edge_weight": 2},
        "interpretation": "Link sponsors to the drugs they study in NSCLC trials.",
    },
    Intent.histogram: {
        "intent": "histogram",
        "filters": {"condition": "breast cancer"},
        "numeric_x": "enrollment",
        "interpretation": "Show the spread of enrollment sizes for breast cancer trials.",
    },
    Intent.correlation: {
        "intent": "correlation",
        "filters": {"condition": "breast cancer"},
        "numeric_x": "enrollment",
        "numeric_y": "duration_months",
        "interpretation": "Plot enrollment against trial duration to see if larger trials run longer.",
    },
}


def test_every_intent_has_an_example():
    """Guard on the test file: a tenth intent must bring an example plan."""
    assert set(VALID_PLANS) == set(Intent)


@pytest.mark.parametrize("intent", list(VALID_PLANS), ids=lambda i: i.value)
def test_valid_plan_passes(intent):
    plan = QueryPlan(**VALID_PLANS[intent])
    assert plan.intent is intent
    assert plan.interpretation
    assert plan.metric is Metric.trial_count  # the only metric so far


def test_plan_defaults():
    plan = QueryPlan(intent="count", interpretation="Count everything.")
    assert plan.filters.drug is None  # filters default to an empty set, not None
    assert plan.assumptions == []
    assert plan.network is None


def test_co_occurrence_network_uses_the_same_dimension_twice():
    """drug <-> drug is a real question: which drugs appear in trials together."""
    plan = QueryPlan(
        intent="network",
        network={"node_dims": ["drug", "drug"]},
        interpretation="Show which drugs are studied together.",
    )
    assert plan.network.node_dims == [Dimension.drug, Dimension.drug]
    assert plan.network.min_edge_weight == 2  # default


# --- plans that must be rejected ------------------------------------------

INVALID_PLANS = [
    # Outside the closed vocabularies.
    ("unknown dimension", {"intent": "distribution", "group_by": "principal_investigator"}),
    ("unknown intent", {"intent": "forecast"}),
    ("unknown phase in filters", {"intent": "count", "filters": {"phase": ["PHASE7"]}}),
    ("unknown status in filters", {"intent": "count", "filters": {"status": ["GOING_WELL"]}}),
    # The hallucination guard: there is nowhere in the shape for an answer.
    ("extra key carrying data", {"intent": "count", "data": [{"trial_count": 412}]}),
    ("extra key carrying an answer", {"intent": "count", "answer": "There are 412 trials."}),
    ("extra key on filters", {"intent": "count", "filters": {"drug_name": "pembrolizumab"}}),
    ("extra key on the network spec", {"intent": "network", "network": {"node_dims": ["sponsor", "drug"], "layout": "force"}}),
    # A numeric axis has to name a numeric dimension.
    ("categorical dimension binned", {"intent": "histogram", "numeric_x": "country"}),
    ("temporal dimension on a scatter axis", {"intent": "correlation", "numeric_x": "enrollment", "numeric_y": "start_year"}),
    # Malformed network specs.
    ("network with one node dimension", {"intent": "network", "network": {"node_dims": ["sponsor"]}}),
    ("network with three node dimensions", {"intent": "network", "network": {"node_dims": ["sponsor", "drug", "phase"]}}),
    ("edge weight below one", {"intent": "network", "network": {"node_dims": ["drug", "drug"], "min_edge_weight": 0}}),
    # Only yearly granularity exists, because start dates are often month-accurate.
    ("monthly granularity", {"intent": "trend", "time": {"granularity": "month"}}),
    # Out-of-range numbers.
    ("year before 1990", {"intent": "count", "filters": {"start_year": 1850}}),
    ("top_k of zero", {"intent": "geo", "group_by": "country", "top_k": 0}),
]


@pytest.mark.parametrize("label,payload", INVALID_PLANS, ids=[label for label, _ in INVALID_PLANS])
def test_invalid_plan_rejected(label, payload):
    payload = {"interpretation": "A plausible-sounding interpretation.", **payload}
    with pytest.raises(ValidationError):
        QueryPlan(**payload)


def test_interpretation_is_required():
    """The plan must say how it read the question, so notes are never empty."""
    with pytest.raises(ValidationError):
        QueryPlan(intent="count")


# --- the schema handed to the LLM -----------------------------------------


def test_tool_schema_is_closed_and_small():
    """In B4 this schema is the forced tool call's input_schema, so it has to
    be closed (no extra keys allowed) and small enough to send every time."""
    schema = QueryPlan.model_json_schema()

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"intent", "interpretation"}
    assert len(json.dumps(schema)) < 20_000

    # The closed vocabularies must reach the model as enums, not free strings.
    assert len(schema["$defs"]["Intent"]["enum"]) == len(Intent)
    assert len(schema["$defs"]["Dimension"]["enum"]) == len(Dimension)


def test_tool_schema_has_no_free_form_fields():
    """A field the model could fill with arbitrary JSON would be a hole in the
    boundary. Every non-scalar field must point at a named model."""
    schema = QueryPlan.model_json_schema()
    for name, prop in schema["properties"].items():
        if prop.get("type") == "object":
            pytest.fail(f"{name} accepts a free-form object")
