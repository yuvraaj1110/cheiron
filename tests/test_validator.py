"""Merging request fields and the per-intent compatibility rules."""

import pytest

from app.api.schemas import Phase, VisualizationRequest
from app.planner.plan import (
    CompareGroup,
    Dimension,
    Intent,
    NetworkSpec,
    PlanFilters,
    QueryPlan,
    TimeSpec,
)
from app.planner.validator import PlanIncompatible, validate_and_merge


def request(**fields) -> VisualizationRequest:
    return VisualizationRequest(query="a question", **fields)


def plan(intent: Intent = Intent.count, **parts) -> QueryPlan:
    return QueryPlan(intent=intent, interpretation="test", **parts)


def test_missing_filter_is_set_silently():
    merged, overrides, _ = validate_and_merge(plan(), request(drug_name="aspirin"))
    assert merged.filters.drug == "aspirin"
    assert overrides == []


def test_conflicting_filter_is_logged_in_exact_format():
    original = plan(filters=PlanFilters(drug="ibuprofen"))
    merged, overrides, _ = validate_and_merge(original, request(drug_name="aspirin"))
    assert merged.filters.drug == "aspirin"
    assert overrides == ["drug: LLM said 'ibuprofen', request said 'aspirin'; used 'aspirin'"]


def test_enum_lists_are_shown_by_value():
    original = plan(filters=PlanFilters(phase=[Phase.PHASE2]))
    _, overrides, _ = validate_and_merge(original, request(phase=[Phase.PHASE3]), source="rule planner")
    assert overrides == ["phase: rule planner said ['PHASE2'], request said ['PHASE3']; used ['PHASE3']"]


def test_same_value_is_not_an_override():
    original = plan(filters=PlanFilters(drug="aspirin"))
    _, overrides, _ = validate_and_merge(original, request(drug_name="aspirin"))
    assert overrides == []


def test_inputs_are_not_mutated():
    original = plan(filters=PlanFilters(drug="ibuprofen"))
    req = request(drug_name="aspirin")
    validate_and_merge(original, req)
    assert original.filters.drug == "ibuprofen"
    assert req.drug_name == "aspirin"


def test_network_without_node_dims_raises():
    with pytest.raises(PlanIncompatible):
        validate_and_merge(plan(Intent.network), request())


def test_network_with_node_dims_passes():
    spec = NetworkSpec(node_dims=[Dimension.sponsor, Dimension.drug])
    merged, _, notes = validate_and_merge(plan(Intent.network, network=spec), request())
    assert merged.network == spec and notes == []


def test_histogram_without_numeric_x_raises():
    with pytest.raises(PlanIncompatible):
        validate_and_merge(plan(Intent.histogram), request())


def test_correlation_needs_both_axes():
    with pytest.raises(PlanIncompatible):
        validate_and_merge(plan(Intent.correlation, numeric_x=Dimension.enrollment), request())


def test_trend_time_defaults_from_years():
    merged, _, notes = validate_and_merge(plan(Intent.trend), request(start_year=2010, end_year=2020))
    assert (merged.time.start, merged.time.end) == (2010, 2020)
    assert notes == ["Trend window taken from start_year/end_year."]


def test_trend_without_any_window_uses_all_years():
    merged, _, notes = validate_and_merge(plan(Intent.trend), request())
    assert merged.time == TimeSpec()
    assert "all years" in notes[0]


def test_comparison_groups_built_from_compare_with_and_main_entity():
    original = plan(Intent.comparison, filters=PlanFilters(drug="aspirin"))
    merged, _, notes = validate_and_merge(original, request(compare_with=["ibuprofen"]))
    assert [g.label for g in merged.compare_groups] == ["aspirin", "ibuprofen"]
    assert merged.compare_groups[1].filters.drug == "ibuprofen"
    assert notes


def test_comparison_with_existing_groups_is_left_alone():
    groups = [CompareGroup(label=n, filters=PlanFilters(drug=n)) for n in ("a", "b")]
    merged, _, notes = validate_and_merge(plan(Intent.comparison, compare_groups=groups), request())
    assert merged.compare_groups == groups and notes == []


def test_comparison_with_fewer_than_two_groups_raises():
    with pytest.raises(PlanIncompatible):
        validate_and_merge(plan(Intent.comparison), request())
