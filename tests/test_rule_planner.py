"""The rule planner: ten questions, one per behaviour, plus the entity rules."""


from app.api.schemas import Phase, Status, VisualizationRequest
from app.planner.plan import Dimension, Intent
from app.planner.rule_planner import mentioned_dimensions, rule_plan


def plan_for(query: str, **fields):
    return rule_plan(VisualizationRequest(query=query, **fields))


def test_count_with_status():
    plan = plan_for("How many recruiting Alzheimer's trials?")
    assert plan.intent == Intent.count
    assert plan.filters.status == [Status.RECRUITING]


def test_list_with_phase_and_quoted_condition():
    plan = plan_for('List phase 3 trials for "melanoma"')
    assert plan.intent == Intent.list
    assert plan.filters.phase == [Phase.PHASE3]
    assert plan.filters.condition == "melanoma"
    assert any("condition" in a for a in plan.assumptions)


def test_network_of_sponsors_and_drugs():
    plan = plan_for("Network of sponsors and drugs")
    assert plan.intent == Intent.network
    assert plan.network.node_dims == [Dimension.sponsor, Dimension.drug]


def test_comparison_of_two_quoted_drugs():
    plan = plan_for('Compare "pembrolizumab" vs "nivolumab" by phase')
    assert plan.intent == Intent.comparison
    assert plan.group_by == Dimension.phase
    assert [g.label for g in plan.compare_groups] == ["pembrolizumab", "nivolumab"]
    assert plan.compare_groups[0].filters.drug == "pembrolizumab"


def test_two_numeric_dimensions_make_a_correlation():
    plan = plan_for("Enrollment vs duration")
    assert plan.intent == Intent.correlation
    assert (plan.numeric_x, plan.numeric_y) == (Dimension.enrollment, Dimension.duration_months)


def test_trend_since_year():
    plan = plan_for("Trials per year since 2015")
    assert plan.intent == Intent.trend
    assert plan.time.start == 2015
    assert plan.filters.start_year == 2015


def test_countries_question_is_geo():
    plan = plan_for("Which countries run the most trials?")
    assert plan.intent == Intent.geo
    assert plan.group_by == Dimension.country


def test_distribution_of_enrollment_is_histogram():
    plan = plan_for("Distribution of enrollment")
    assert plan.intent == Intent.histogram
    assert plan.numeric_x == Dimension.enrollment


def test_sponsor_type_is_sponsor_class_not_sponsor():
    plan = plan_for("Trials by sponsor type")
    assert plan.intent == Intent.distribution
    assert plan.group_by == Dimension.sponsor_class


def test_no_cue_defaults_to_phase_with_low_confidence_note():
    plan = plan_for("Breakdown of trials")
    assert plan.intent == Intent.distribution
    assert plan.group_by == Dimension.phase
    assert any("low confidence" in a and "phase" in a for a in plan.assumptions)


def test_unquoted_names_are_read_with_a_low_confidence_note():
    # Without this the fallback would search all ~600k registered trials.
    plan = plan_for("How many pembrolizumab trials in melanoma?")
    assert plan.filters.drug == "pembrolizumab"
    assert plan.filters.condition == "melanoma"
    assert any("low confidence" in note for note in plan.assumptions)


def test_no_name_found_is_reported():
    plan = plan_for("How are trials distributed across phases?")
    assert plan.filters.drug is None and plan.filters.condition is None
    assert any("searched all registered trials" in note for note in plan.assumptions)


def test_duplicate_compare_names_collapse():
    plan = plan_for("Compare pembrolizumab vs nivolumab", compare_with=["pembrolizumab", "nivolumab"])
    assert [g.label for g in plan.compare_groups] == ["pembrolizumab", "nivolumab"]


def test_request_fields_are_used_as_entities():
    plan = plan_for("How many trials?", drug_name="aspirin", country="France")
    assert plan.filters.drug == "aspirin"
    assert plan.filters.country == "France"


def test_comparison_with_compare_with_uses_main_entity():
    plan = plan_for("Compare phases", drug_name="aspirin", compare_with=["ibuprofen"])
    assert [g.label for g in plan.compare_groups] == ["aspirin", "ibuprofen"]


def test_comparison_without_two_groups_degrades_to_distribution():
    plan = plan_for("Compare phases")
    assert plan.intent == Intent.distribution


def test_mentioned_dimensions_in_order_of_appearance():
    assert mentioned_dimensions("status by country and phase") == [
        Dimension.status,
        Dimension.country,
        Dimension.phase,
    ]


def test_histogram_question_finds_the_condition_not_the_measure():
    plan = plan_for("What is the distribution of enrollment for Alzheimer's disease trials?")
    assert plan.filters.condition == "Alzheimer's disease"
