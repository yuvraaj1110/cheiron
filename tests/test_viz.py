"""Tests for app/viz/builder.py.

Three things are pinned here:

- every intent produces its chart with the channels that chart requires, so a
  frontend never receives a spec it cannot draw;
- a viz_hint is honoured only when it fits, and a refused hint is reported
  rather than silently dropped;
- the titles and sort orders, because a chart whose title overstates what was
  counted is wrong even when the numbers are right.
"""

import pytest

from app.analytics import operators as ops
from app.api.schemas import VizType
from app.ctgov.normalize import PHASE_ORDER, Intervention, TrialRecord
from app.planner.plan import Dimension, Intent, QueryPlan
from app.viz.builder import (
    ALTERNATIVE_VIZ,
    INTENT_TO_VIZ,
    NOT_A_CHART,
    build_spec,
    build_title,
    channel_type,
    choose_viz,
)


def record(nct_id="NCT1", **kwargs) -> TrialRecord:
    from app.ctgov.normalize import drug_key

    drugs = kwargs.pop("drugs", ["Pembrolizumab"])
    defaults = dict(
        title=f"Trial {nct_id}",
        phase_label="Phase 2",
        status="RECRUITING",
        study_type="INTERVENTIONAL",
        start_year=2020,
        sponsor="Merck",
        sponsor_class="INDUSTRY",
        countries=["France"],
        conditions=["Melanoma"],
        enrollment=100,
        duration_months=12.0,
    )
    defaults.update(kwargs)
    return TrialRecord(
        nct_id=nct_id,
        interventions=[Intervention(type="DRUG", name=d, key=drug_key(d)) for d in drugs],
        **defaults,
    )


RECORDS = [record(f"NCT{i}", start_year=2018 + i % 3) for i in range(6)]


def plan(**kwargs) -> QueryPlan:
    kwargs.setdefault("interpretation", "A test plan.")
    return QueryPlan(**kwargs)


# One plan and one matching operator result per intent, so every row shape the
# builder can receive is actually exercised.
def case_for(intent: Intent):
    if intent is Intent.count:
        return plan(intent="count"), ops.count(RECORDS)
    if intent is Intent.list:
        return plan(intent="list"), ops.list_trials(RECORDS)
    if intent is Intent.distribution:
        return plan(intent="distribution", group_by="phase"), ops.count_by(
            RECORDS, Dimension.phase
        )
    if intent is Intent.trend:
        return plan(intent="trend", time={"start": 2018}), ops.time_series(RECORDS, start=2018)
    if intent is Intent.comparison:
        groups = {"a": RECORDS[:3], "b": RECORDS[3:]}
        return (
            plan(
                intent="comparison",
                group_by="phase",
                compare_groups=[
                    {"label": "a", "filters": {"drug": "a"}},
                    {"label": "b", "filters": {"drug": "b"}},
                ],
            ),
            ops.compare(groups, Dimension.phase),
        )
    if intent is Intent.geo:
        return plan(intent="geo", group_by="country"), ops.count_by(RECORDS, Dimension.country)
    if intent is Intent.network:
        records = [record(f"N{i}", drugs=["A", "B"]) for i in range(3)]
        return (
            plan(intent="network", network={"node_dims": ["drug", "drug"]}),
            ops.network(records, [Dimension.drug, Dimension.drug], min_edge_weight=2),
        )
    if intent is Intent.histogram:
        return plan(intent="histogram", numeric_x="enrollment"), ops.histogram(
            RECORDS, Dimension.enrollment
        )
    if intent is Intent.correlation:
        return (
            plan(intent="correlation", numeric_x="enrollment", numeric_y="duration_months"),
            ops.scatter(RECORDS, Dimension.enrollment, Dimension.duration_months),
        )
    raise AssertionError(f"no case for {intent}")


# --- the rule table -------------------------------------------------------


def test_every_intent_has_a_chart():
    """An intent with no chart would be a runtime KeyError."""
    assert set(INTENT_TO_VIZ) == set(Intent)


def test_every_intent_has_an_alternatives_entry():
    assert set(ALTERNATIVE_VIZ) == set(Intent)


@pytest.mark.parametrize("intent", list(Intent), ids=lambda i: i.value)
def test_intent_produces_its_chart(intent):
    query_plan, result = case_for(intent)
    built = build_spec(query_plan, result)
    assert built.spec.type is INTENT_TO_VIZ[intent]


@pytest.mark.parametrize("intent", list(Intent), ids=lambda i: i.value)
def test_built_spec_has_a_title_and_a_rationale(intent):
    query_plan, result = case_for(intent)
    built = build_spec(query_plan, result)
    assert built.spec.title
    assert built.rationale


@pytest.mark.parametrize("intent", list(Intent), ids=lambda i: i.value)
def test_chart_recommended_is_false_only_for_a_number_or_a_list(intent):
    query_plan, result = case_for(intent)
    built = build_spec(query_plan, result)
    assert built.chart_recommended is (INTENT_TO_VIZ[intent] not in NOT_A_CHART)


# --- required channels ----------------------------------------------------
# The spec is validated inside build_spec, so a missing channel raises there.
# These assert the channels point at fields that exist in the rows, which
# validation cannot check.


def test_bar_chart_channels_name_real_row_fields():
    built = build_spec(*case_for(Intent.distribution))
    fields = {built.spec.encoding.x.field, built.spec.encoding.y.field}
    assert fields <= set(built.spec.data[0])


def test_grouped_bar_series_names_a_real_row_field():
    built = build_spec(*case_for(Intent.comparison))
    assert built.spec.encoding.series.field in built.spec.data[0]


def test_time_series_channels_name_real_row_fields():
    built = build_spec(*case_for(Intent.trend))
    assert built.spec.encoding.x.field == "year"
    assert "year" in built.spec.data[0]


def test_histogram_channels_name_real_row_fields():
    built = build_spec(*case_for(Intent.histogram))
    assert built.spec.encoding.x.field == "bin_start"
    assert built.spec.encoding.y.field == "count"
    assert {"bin_start", "bin_end", "count"} <= set(built.spec.data[0])


def test_scatter_channels_name_real_row_fields():
    built = build_spec(*case_for(Intent.correlation))
    assert built.spec.encoding.x.field == "enrollment"
    assert built.spec.encoding.y.field == "duration_months"
    assert {"enrollment", "duration_months"} <= set(built.spec.data[0])


def test_network_channels_name_real_node_fields():
    """color and size must point at NetworkNode fields, since the nodes and
    edges are the data and no channel can point at those."""
    built = build_spec(*case_for(Intent.network))
    node = built.spec.data.nodes[0].model_dump()
    assert built.spec.encoding.color.field in node
    assert built.spec.encoding.size.field in node


def test_table_columns_name_real_row_fields():
    built = build_spec(*case_for(Intent.list))
    column_fields = {c.field for c in built.spec.columns}
    assert column_fields <= set(built.spec.data[0])


# --- channel types --------------------------------------------------------


def test_phase_is_ordinal_not_nominal():
    """Phase is categorical but genuinely ordered, so a frontend must keep
    Phase 1 before Phase 2 rather than sorting it alphabetically."""
    assert channel_type(Dimension.phase) == "ordinal"


def test_other_categories_are_nominal():
    assert channel_type(Dimension.country) == "nominal"
    assert channel_type(Dimension.sponsor) == "nominal"


def test_numeric_dimensions_are_quantitative():
    assert channel_type(Dimension.enrollment) == "quantitative"
    assert channel_type(Dimension.duration_months) == "quantitative"


def test_start_year_is_temporal():
    assert channel_type(Dimension.start_year) == "temporal"


# --- sort rules -----------------------------------------------------------


def test_phase_axis_uses_explicit_chart_order():
    """Neither alphabetical nor by-count puts "Phase 1/2" between 1 and 2, or
    "Not applicable" at the end."""
    built = build_spec(*case_for(Intent.distribution))
    assert built.spec.encoding.x.sort == PHASE_ORDER


def test_other_category_axes_sort_by_count_descending():
    built = build_spec(*case_for(Intent.geo))
    assert built.spec.encoding.x.sort == "-trial_count"


def test_time_axis_sorts_ascending():
    """A trend read right to left is not a trend."""
    built = build_spec(*case_for(Intent.trend))
    assert built.spec.encoding.x.sort == "year"


def test_geo_is_horizontal():
    """Country names are long and unreadable rotated under a vertical axis."""
    built = build_spec(*case_for(Intent.geo))
    assert built.spec.options["horizontal"] is True


def test_distribution_is_not_horizontal():
    built = build_spec(*case_for(Intent.distribution))
    assert built.spec.options["horizontal"] is False


def test_units_are_set_on_value_axes():
    built = build_spec(*case_for(Intent.distribution))
    assert built.spec.encoding.y.unit == "trials"
    built = build_spec(*case_for(Intent.correlation))
    assert built.spec.encoding.x.unit == "participants"
    assert built.spec.encoding.y.unit == "months"


def test_histogram_passes_the_log_scale_flag_through():
    """Without it the frontend draws even-width bars and misrepresents a
    skewed distribution."""
    records = [record(f"N{i}", enrollment=10) for i in range(50)]
    records.append(record("NBIG", enrollment=50_000))
    result = ops.histogram(records, Dimension.enrollment)
    built = build_spec(plan(intent="histogram", numeric_x="enrollment"), result)
    assert built.spec.options["log_scale"] is True


# --- viz_hint -------------------------------------------------------------


def test_no_hint_uses_the_default_and_says_nothing():
    viz, notes = choose_viz(Intent.distribution)
    assert viz is VizType.bar_chart
    assert notes == []


def test_a_hint_matching_the_default_is_not_worth_a_note():
    viz, notes = choose_viz(Intent.distribution, VizType.bar_chart)
    assert viz is VizType.bar_chart
    assert notes == []


def test_a_compatible_hint_is_honoured_and_recorded():
    viz, notes = choose_viz(Intent.distribution, VizType.table)
    assert viz is VizType.table
    assert notes and "table" in notes[0]


def test_an_incompatible_hint_falls_back_with_a_reason():
    """Silently ignoring a request is worse than refusing it out loud."""
    viz, notes = choose_viz(Intent.distribution, VizType.network_graph)
    assert viz is VizType.bar_chart
    assert notes
    assert "network_graph" in notes[0] and "bar_chart" in notes[0]


@pytest.mark.parametrize("hint", list(VizType), ids=lambda v: v.value)
def test_a_count_never_becomes_a_chart_whatever_is_asked_for(hint):
    """A single number has no rows to plot, so stat has no alternatives."""
    viz, _ = choose_viz(Intent.count, hint)
    if hint is not VizType.stat:
        assert viz is VizType.stat


@pytest.mark.parametrize("hint", list(VizType), ids=lambda v: v.value)
def test_a_network_is_never_drawn_as_something_else(hint):
    """Nodes and edges are not rows, so no other chart can show them."""
    viz, _ = choose_viz(Intent.network, hint)
    if hint is not VizType.network_graph:
        assert viz is VizType.network_graph


def test_an_honoured_hint_actually_changes_the_spec():
    query_plan, result = case_for(Intent.distribution)
    built = build_spec(query_plan, result, viz_hint=VizType.table)
    assert built.spec.type is VizType.table
    assert built.spec.columns  # and it is a valid table, not a mislabelled bar


# --- titles ---------------------------------------------------------------


def test_title_names_the_filters_that_were_applied():
    """A title that overstates what was counted is wrong even when the
    numbers are right."""
    title = build_title(
        plan(
            intent="distribution",
            filters={"drug": "pembrolizumab", "condition": "melanoma", "status": ["RECRUITING"]},
            group_by="phase",
        )
    )
    assert "recruiting" in title.lower()
    assert "pembrolizumab" in title.lower()
    assert "melanoma" in title
    assert "phase" in title.lower()


def test_title_leads_with_the_entity_rather_than_the_word_trials():
    """"Trials in breast cancer by phase" is awkward; a reader expects the
    thing being studied first, capitalized."""
    title = build_title(
        plan(intent="distribution", filters={"condition": "breast cancer"}, group_by="phase")
    )
    assert title == "Breast cancer trials by phase"


def test_title_capitalizes_the_entity_mid_sentence():
    title = build_title(plan(intent="count", filters={"condition": "alzheimer", "status": ["RECRUITING"]}))
    assert title == "Number of recruiting Alzheimer trials"


def test_title_leaves_an_entity_that_is_already_capitalized_alone():
    """str.title() and str.capitalize() would both mangle "BRCA1"."""
    title = build_title(plan(intent="list", filters={"condition": "BRCA1 mutation"}))
    assert title.startswith("BRCA1 mutation trials")


def test_title_with_both_a_drug_and_a_condition_leads_with_the_drug():
    title = build_title(
        plan(intent="trend", filters={"drug": "pembrolizumab", "condition": "breast cancer"})
    )
    assert title == "Pembrolizumab trials in breast cancer per year"


def test_title_includes_a_year_window():
    assert ", 2015-2024" in build_title(
        plan(intent="trend", filters={"start_year": 2015, "end_year": 2024})
    )


def test_title_includes_a_one_sided_window():
    assert "since 2015" in build_title(plan(intent="trend", time={"start": 2015}))


def test_comparison_title_names_the_sides():
    title = build_title(
        plan(
            intent="comparison",
            group_by="phase",
            compare_groups=[
                {"label": "pembrolizumab", "filters": {"drug": "pembrolizumab"}},
                {"label": "nivolumab", "filters": {"drug": "nivolumab"}},
            ],
        )
    )
    assert "pembrolizumab vs nivolumab" in title


def test_cooccurrence_and_bipartite_networks_read_differently():
    same = build_title(plan(intent="network", network={"node_dims": ["drug", "drug"]}))
    different = build_title(plan(intent="network", network={"node_dims": ["sponsor", "drug"]}))
    assert "studied together" in same
    assert "sponsors and drugs" in different.lower()


def test_titles_use_sentence_nouns_not_axis_labels():
    """"Participants of trials" is wrong in a sentence even though
    "Participants" is right on an axis."""
    title = build_title(plan(intent="histogram", numeric_x="enrollment"))
    assert "enrollment" in title.lower()
    assert "participants" not in title.lower()


def test_title_starts_with_a_capital():
    for intent in Intent:
        query_plan, _ = case_for(intent)
        title = build_title(query_plan)
        assert title[0].isupper(), f"{intent.value}: {title!r}"


# --- notes ----------------------------------------------------------------


def test_dropped_trials_are_reported():
    records = [record("NCT1", enrollment=100), record("NCT2", enrollment=None)]
    result = ops.histogram(records, Dimension.enrollment)
    built = build_spec(plan(intent="histogram", numeric_x="enrollment"), result)
    assert any("1 trial was left out" in n for n in built.notes)


def test_dropped_note_is_pluralised():
    """"1 trials were left out" reads like a bug."""
    records = [record("NCT1", enrollment=100)] + [
        record(f"NCT{i}", enrollment=None) for i in range(2, 5)
    ]
    result = ops.histogram(records, Dimension.enrollment)
    built = build_spec(plan(intent="histogram", numeric_x="enrollment"), result)
    assert any("3 trials were left out" in n for n in built.notes)


def test_a_network_explains_why_a_trial_contributed_nothing():
    """A trial missing from a network is missing for a different reason than
    one missing from a bar chart."""
    records = [record("NCT1", drugs=["OnlyOne"])]
    result = ops.network(records, [Dimension.drug, Dimension.drug], min_edge_weight=1)
    built = build_spec(plan(intent="network", network={"node_dims": ["drug", "drug"]}), result)
    assert any("no pair" in n for n in built.notes)


def test_an_empty_graph_says_why_and_what_would_help():
    """A blank canvas with no explanation is the worst possible answer."""
    records = [record("NCT1", sponsor="A", drugs=["X"]), record("NCT2", sponsor="B", drugs=["Y"])]
    result = ops.network(records, [Dimension.sponsor, Dimension.drug], min_edge_weight=2)
    built = build_spec(
        plan(intent="network", network={"node_dims": ["sponsor", "drug"]}), result
    )
    note = " ".join(built.notes)
    assert "empty" in note
    assert "max_studies" in note or "min_edge_weight" in note


def test_a_full_result_adds_no_dropped_note():
    built = build_spec(*case_for(Intent.distribution))
    assert not any("left out" in n for n in built.notes)
