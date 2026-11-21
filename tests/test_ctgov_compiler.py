"""Tests for app/ctgov/compiler.py.

The compiler is pure, so each test states a plan and the exact dict of params
it must produce. Exact comparison rather than "contains" is deliberate: an
extra or missing filter is the difference between a right and a wrong answer,
and a subset assertion would not catch either.
"""

import pytest

from app.ctgov.compiler import (
    ALL_MODULES,
    BASE_MODULES,
    compile_filters,
    compile_groups,
    compile_plan,
    merge_filters,
    module_for,
    modules_for_plan,
)
from app.planner.plan import Dimension, Intent, PlanFilters, QueryPlan

IDENTIFICATION = "protocolSection.identificationModule"
STATUS = "protocolSection.statusModule"
DESIGN = "protocolSection.designModule"
SPONSOR = "protocolSection.sponsorCollaboratorsModule"
CONDITIONS = "protocolSection.conditionsModule"
INTERVENTIONS = "protocolSection.armsInterventionsModule"
LOCATIONS = "protocolSection.contactsLocationsModule"


def plan(**kwargs) -> QueryPlan:
    """A plan with the boilerplate filled in."""
    kwargs.setdefault("interpretation", "A test plan.")
    return QueryPlan(**kwargs)


# --- the five plans -------------------------------------------------------


def test_distribution_by_phase():
    """A condition search grouped by phase: one query param, one module added."""
    params = compile_plan(
        plan(intent="distribution", filters={"condition": "breast cancer"}, group_by="phase")
    )
    assert params == {
        "query.cond": "breast cancer",
        "fields": f"{DESIGN},{IDENTIFICATION},{STATUS}",
        "countTotal": "true",
    }


def test_trend_with_a_year_window_and_a_status_filter():
    """Status has its own param; the year window goes through filter.advanced."""
    params = compile_plan(
        plan(
            intent="trend",
            filters={
                "drug": "pembrolizumab",
                "status": ["RECRUITING"],
                "start_year": 2015,
                "end_year": 2024,
            },
            time={"granularity": "year", "start": 2015},
        )
    )
    assert params == {
        "query.intr": "pembrolizumab",
        "filter.overallStatus": "RECRUITING",
        "filter.advanced": "AREA[StartDate]RANGE[2015-01-01,2024-12-31]",
        "fields": f"{IDENTIFICATION},{STATUS}",
        "countTotal": "true",
    }


def test_geo_by_country():
    params = compile_plan(
        plan(
            intent="geo",
            filters={"condition": "diabetes", "status": ["RECRUITING"]},
            group_by="country",
        )
    )
    assert params == {
        "query.cond": "diabetes",
        "filter.overallStatus": "RECRUITING",
        "fields": f"{LOCATIONS},{IDENTIFICATION},{STATUS}",
        "countTotal": "true",
    }


def test_network_asks_for_both_node_modules():
    params = compile_plan(
        plan(
            intent="network",
            filters={"condition": "non-small cell lung cancer"},
            network={"node_dims": ["sponsor", "drug"]},
        )
    )
    assert params == {
        "query.cond": "non-small cell lung cancer",
        "fields": f"{INTERVENTIONS},{IDENTIFICATION},{SPONSOR},{STATUS}",
        "countTotal": "true",
    }


def test_list_asks_for_every_module():
    """A table shows phase, sponsor and status, which are columns the user
    reads even though they are not dimensions of the question."""
    params = compile_plan(plan(intent="list", filters={"drug": "pembrolizumab"}))
    assert params == {
        "query.intr": "pembrolizumab",
        "fields": ",".join(ALL_MODULES),
        "countTotal": "true",
    }


# --- filters in detail ----------------------------------------------------


def test_all_four_entity_filters_map_to_their_params():
    params = compile_filters(
        PlanFilters(
            drug="pembrolizumab", condition="melanoma", sponsor="Merck", country="France"
        )
    )
    assert params == {
        "query.intr": "pembrolizumab",
        "query.cond": "melanoma",
        "query.spons": "Merck",
        "query.locn": "France",
    }


def test_empty_filters_produce_nothing():
    """No filters set must mean no params, not a param with an empty value -
    an empty query.cond would match nothing rather than everything."""
    assert compile_filters(PlanFilters()) == {}


def test_single_phase_clause():
    assert compile_filters(PlanFilters(phase=["PHASE3"]))["filter.advanced"] == (
        "AREA[Phase]PHASE3"
    )


def test_several_phases_become_an_or_clause():
    assert compile_filters(PlanFilters(phase=["PHASE2", "PHASE3"]))["filter.advanced"] == (
        "AREA[Phase](PHASE2 OR PHASE3)"
    )


def test_several_statuses_are_comma_joined():
    filters = PlanFilters(status=["RECRUITING", "COMPLETED"])
    assert compile_filters(filters)["filter.overallStatus"] == "RECRUITING,COMPLETED"


@pytest.mark.parametrize(
    "start,end,expected",
    [
        (2015, None, "AREA[StartDate]RANGE[2015-01-01,MAX]"),
        (None, 2020, "AREA[StartDate]RANGE[MIN,2020-12-31]"),
        (2015, 2020, "AREA[StartDate]RANGE[2015-01-01,2020-12-31]"),
    ],
)
def test_one_sided_year_windows_use_the_api_bounds(start, end, expected):
    """MIN and MAX are the API's own open-ended bounds, so a half-open window
    needs no special case."""
    filters = PlanFilters(start_year=start, end_year=end)
    assert compile_filters(filters)["filter.advanced"] == expected


def test_phase_and_dates_are_joined_with_and():
    filters = PlanFilters(phase=["PHASE3"], start_year=2015)
    assert compile_filters(filters)["filter.advanced"] == (
        "AREA[Phase]PHASE3 AND AREA[StartDate]RANGE[2015-01-01,MAX]"
    )


# --- the fields projection ------------------------------------------------


def test_every_dimension_maps_to_a_real_module():
    for dim in Dimension:
        module = module_for(dim)
        assert module.startswith("protocolSection.")
        assert module.endswith("Module")


def test_duration_months_needs_the_status_module():
    """It is computed from two dates, so it is not in a module of its own."""
    assert module_for(Dimension.duration_months) == STATUS


def test_base_modules_are_always_requested():
    """Citations need the ID and title, and are ordered by start date, so
    these two are needed whatever the question was."""
    for intent in Intent:
        modules = modules_for_plan(plan(intent=intent.value))
        for base in BASE_MODULES:
            assert base in modules


def test_projection_excludes_modules_the_plan_cannot_use():
    """The point of the projection: a phase question must not drag back every
    trial's full location list."""
    modules = modules_for_plan(plan(intent="distribution", group_by="phase"))
    assert LOCATIONS not in modules
    assert INTERVENTIONS not in modules


def test_filters_do_not_widen_the_projection():
    """Filtering happens on the server, so a filtered field is never needed
    back in the response."""
    with_filter = modules_for_plan(
        plan(intent="count", filters={"country": "France", "sponsor": "Merck"})
    )
    assert with_filter == sorted(BASE_MODULES)


def test_scatter_asks_for_both_axes_modules():
    modules = modules_for_plan(
        plan(intent="correlation", numeric_x="enrollment", numeric_y="duration_months")
    )
    assert DESIGN in modules  # enrollment
    assert STATUS in modules  # duration, from the two dates


# --- comparison groups ----------------------------------------------------


def test_a_plan_without_groups_compiles_to_one_search():
    label, params = compile_groups(plan(intent="count", filters={"condition": "diabetes"}))[0]
    assert label == "all"
    assert params["query.cond"] == "diabetes"


def test_comparison_compiles_to_one_search_per_group():
    """Two sides means two API searches. One search could not tell afterwards
    which trials belonged to which side."""
    groups = compile_groups(
        plan(
            intent="comparison",
            group_by="phase",
            compare_groups=[
                {"label": "pembrolizumab", "filters": {"drug": "pembrolizumab"}},
                {"label": "nivolumab", "filters": {"drug": "nivolumab"}},
            ],
        )
    )
    assert [label for label, _ in groups] == ["pembrolizumab", "nivolumab"]
    assert groups[0][1]["query.intr"] == "pembrolizumab"
    assert groups[1][1]["query.intr"] == "nivolumab"


def test_group_filters_inherit_the_plans_shared_context():
    """"Compare pembrolizumab vs nivolumab in NSCLC" must keep the condition
    on both sides, or the two numbers are not comparable."""
    groups = compile_groups(
        plan(
            intent="comparison",
            filters={"condition": "non-small cell lung cancer", "status": ["RECRUITING"]},
            group_by="phase",
            compare_groups=[
                {"label": "pembrolizumab", "filters": {"drug": "pembrolizumab"}},
                {"label": "nivolumab", "filters": {"drug": "nivolumab"}},
            ],
        )
    )
    for _, params in groups:
        assert params["query.cond"] == "non-small cell lung cancer"
        assert params["filter.overallStatus"] == "RECRUITING"


def test_a_group_can_override_the_shared_context():
    merged = merge_filters(
        PlanFilters(condition="melanoma", drug="pembrolizumab"),
        PlanFilters(drug="nivolumab"),
    )
    assert merged.drug == "nivolumab"  # the group wins
    assert merged.condition == "melanoma"  # and inherits the rest


def test_merge_ignores_fields_the_group_did_not_set():
    """An unset field must not blank out the shared context."""
    merged = merge_filters(PlanFilters(condition="melanoma"), PlanFilters())
    assert merged.condition == "melanoma"
