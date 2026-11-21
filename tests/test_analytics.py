"""Tests for app/analytics/operators.py.

Two kinds of test, on purpose:

- hand-built records with totals worked out by hand, so a wrong number is
  caught rather than merely a crash. Fixture data cannot do this job: nobody
  knows by eye how many Phase 2 breast cancer trials there should be.
- fixture-driven invariants, which hand-built records cannot do: every trial
  accounted for, no orphan edges, no self-loops, contributors present for
  every datum.

The contributors assertions are not incidental. A number with no contributors
cannot be cited, so an operator that forgot to record them would produce an
uncitable chart - which this project treats as a broken chart.
"""

import pytest

from app.analytics import operators as ops
from app.analytics.operators import OTHER_LABEL, OperatorResult
from app.ctgov.normalize import Intervention, TrialRecord, normalize_studies
from app.planner.plan import Dimension
from tests.helpers import all_fixture_studies, fixture_studies


def record(
    nct_id: str,
    phase_label: str = "Phase 2",
    status: str = "RECRUITING",
    study_type: str = "INTERVENTIONAL",
    start_year: int | None = 2020,
    sponsor: str | None = "Merck",
    sponsor_class: str | None = "INDUSTRY",
    countries: list[str] | None = None,
    drugs: list[str] | None = None,
    conditions: list[str] | None = None,
    enrollment: int | None = 100,
    duration_months: float | None = 12.0,
) -> TrialRecord:
    """A record with only the fields a given test cares about."""
    from app.ctgov.normalize import drug_key

    return TrialRecord(
        nct_id=nct_id,
        title=f"Trial {nct_id}",
        phase_label=phase_label,
        status=status,
        study_type=study_type,
        start_year=start_year,
        sponsor=sponsor,
        sponsor_class=sponsor_class,
        countries=countries or [],
        conditions=conditions or [],
        interventions=[
            Intervention(type="DRUG", name=d, key=drug_key(d)) for d in (drugs or [])
        ],
        enrollment=enrollment,
        duration_months=duration_months,
    )


def total(result: OperatorResult, field: str = "trial_count") -> int:
    return sum(row[field] for row in result.rows)


# --- 1. count -------------------------------------------------------------


def test_count():
    result = ops.count([record("NCT1"), record("NCT2"), record("NCT3")])
    assert result.rows == [{"trial_count": 3}]


def test_count_of_nothing_is_zero_not_an_error():
    """A question with no matching trials is a valid answer: zero."""
    result = ops.count([])
    assert result.rows == [{"trial_count": 0}]


def test_even_a_single_number_is_citable():
    result = ops.count([record("NCT1"), record("NCT2")])
    assert result.contributors["total"] == {"NCT1", "NCT2"}


# --- 2. list --------------------------------------------------------------


def test_list_is_newest_first():
    """"Which trials" usually means "what is happening now"."""
    records = [record("NCT1", start_year=2015), record("NCT2", start_year=2024)]
    result = ops.list_trials(records)
    assert [r["nct_id"] for r in result.rows] == ["NCT2", "NCT1"]


def test_list_respects_its_limit_and_reports_what_was_left_out():
    records = [record(f"NCT{i}", start_year=2000 + i) for i in range(10)]
    result = ops.list_trials(records, limit=3)
    assert len(result.rows) == 3
    assert result.extra == {
        "columns": ops.LIST_COLUMNS,
        "shown": 3,
        "available": 10,
    }


def test_each_listed_trial_is_its_own_contributor():
    result = ops.list_trials([record("NCT1")])
    assert result.contributors == {"nct_id=NCT1": {"NCT1"}}


# --- 3. count_by ----------------------------------------------------------


def test_count_by_single_valued_dimension_sums_to_the_trial_total():
    """Phase is single-valued, so the bars must add up to the trial count."""
    records = [
        record("NCT1", phase_label="Phase 2"),
        record("NCT2", phase_label="Phase 2"),
        record("NCT3", phase_label="Phase 3"),
    ]
    result = ops.count_by(records, Dimension.phase)
    assert result.rows == [
        {"phase": "Phase 2", "trial_count": 2},
        {"phase": "Phase 3", "trial_count": 1},
    ]
    assert total(result) == len(records)


def test_count_by_multi_valued_dimension_counts_a_trial_once_per_value():
    """The surprising-but-correct case: one trial in two countries adds one to
    each, so the bars sum to more than the number of trials."""
    records = [
        record("NCT1", countries=["France", "Germany"]),
        record("NCT2", countries=["France"]),
    ]
    result = ops.count_by(records, Dimension.country)
    assert result.rows == [
        {"country": "France", "trial_count": 2},
        {"country": "Germany", "trial_count": 1},
    ]
    assert total(result) == 3 > len(records)
    assert result.extra["multi_valued"] is True


def test_count_by_orders_biggest_first():
    """Readers scan a bar chart from the top, so count has to lead."""
    records = [record(f"NCT{i}", phase_label="Phase 1") for i in range(3)]
    records += [record("NCT9", phase_label="Phase 3")]
    result = ops.count_by(records, Dimension.phase)
    assert [r["phase"] for r in result.rows] == ["Phase 1", "Phase 3"]


def test_count_by_breaks_ties_in_the_dimensions_own_order():
    """Equal counts fall back to chart order, not alphabetical: "Phase 1/2"
    belongs between 1 and 2, and "Not applicable" at the end."""
    records = [
        record("NCT1", phase_label="Not applicable"),
        record("NCT2", phase_label="Phase 2"),
        record("NCT3", phase_label="Phase 1/2"),
        record("NCT4", phase_label="Phase 1"),
    ]
    result = ops.count_by(records, Dimension.phase)
    assert [r["phase"] for r in result.rows] == [
        "Phase 1",
        "Phase 1/2",
        "Phase 2",
        "Not applicable",
    ]


def test_count_by_drops_records_with_no_value_and_says_how_many():
    """Dropped records are never silently discarded: the pipeline turns this
    number into a line in notes.assumptions."""
    records = [record("NCT1", countries=["France"]), record("NCT2", countries=[])]
    result = ops.count_by(records, Dimension.country)
    assert total(result) == 1
    assert result.dropped == 1


def test_count_by_groups_drug_spellings_but_labels_them_verbatim():
    """Several spellings, one bar - and the label is the API's own text, so a
    citation can quote it."""
    records = [
        record("NCT1", drugs=["Pembrolizumab 200 mg"]),
        record("NCT2", drugs=["pembrolizumab (MK-3475)"]),
    ]
    result = ops.count_by(records, Dimension.drug)
    assert len(result.rows) == 1
    assert result.rows[0]["trial_count"] == 2
    assert result.rows[0]["drug"] in {"Pembrolizumab 200 mg", "pembrolizumab (MK-3475)"}


def test_count_by_counts_a_trial_once_per_drug_not_once_per_arm():
    """A trial listing the same drug in three arms is still one trial."""
    records = [record("NCT1", drugs=["Pembrolizumab", "Pembrolizumab 200 mg"])]
    result = ops.count_by(records, Dimension.drug)
    assert result.rows == [{"drug": "Pembrolizumab", "trial_count": 1}]


@pytest.mark.parametrize("dim", list(Dimension))
def test_every_dimension_can_be_read_off_a_record(dim):
    """A dimension the LLM may choose but no operator can read would be a
    runtime failure, so the closed enum has to be fully covered."""
    if dim in (Dimension.enrollment, Dimension.duration_months):
        assert ops.numeric_value(record("NCT1"), dim) is not None
        return
    pairs = ops.dimension_values(
        record("NCT1", countries=["France"], drugs=["Aspirin"], conditions=["Melanoma"]), dim
    )
    assert pairs, f"{dim.value} produced no value"


# --- 4. time_series -------------------------------------------------------


def test_time_series_fills_missing_years_with_zero():
    """The reason this operator exists. Without the zero, the axis skips 2021
    and the line joins 2020 straight to 2022 - which reads as continuity
    where there was a gap."""
    records = [record("NCT1", start_year=2020), record("NCT2", start_year=2022)]
    result = ops.time_series(records)
    assert result.rows == [
        {"year": 2020, "trial_count": 1},
        {"year": 2021, "trial_count": 0},
        {"year": 2022, "trial_count": 1},
    ]


def test_time_series_honours_an_explicit_window():
    records = [record("NCT1", start_year=2010), record("NCT2", start_year=2020)]
    result = ops.time_series(records, start=2019, end=2021)
    assert [r["year"] for r in result.rows] == [2019, 2020, 2021]
    assert result.dropped == 1  # the 2010 trial is outside the window


def test_time_series_drops_trials_with_no_start_date():
    """A trial with no start date cannot be placed on a time axis at all."""
    records = [record("NCT1", start_year=2020), record("NCT2", start_year=None)]
    result = ops.time_series(records)
    assert total(result) == 1
    assert result.dropped == 1


def test_time_series_with_no_dated_trials_returns_nothing_rather_than_raising():
    result = ops.time_series([record("NCT1", start_year=None)])
    assert result.rows == []
    assert result.dropped == 1


def test_time_series_by_series_keeps_the_same_year_axis_for_every_line():
    """Lines with different axes are not comparable, so each series gets a row
    for every year in the window including zeros."""
    records = [
        record("NCT1", start_year=2020, sponsor_class="INDUSTRY"),
        record("NCT2", start_year=2022, sponsor_class="NIH"),
    ]
    result = ops.time_series(records, series_by=Dimension.sponsor_class)
    years_per_series = {}
    for row in result.rows:
        years_per_series.setdefault(row["series"], []).append(row["year"])
    assert years_per_series == {
        "INDUSTRY": [2020, 2021, 2022],
        "NIH": [2020, 2021, 2022],
    }


def test_time_series_zero_years_have_no_contributors():
    """A zero is not citable, because no trial is behind it."""
    records = [record("NCT1", start_year=2020), record("NCT2", start_year=2022)]
    result = ops.time_series(records)
    assert "year=2021" not in result.contributors
    assert result.contributors["year=2020"] == {"NCT1"}


# --- 5. compare -----------------------------------------------------------


def test_compare_produces_long_format_rows():
    """One row per (category, group), which is what a grouped bar chart reads
    and what does not change shape when a third group is added."""
    groups = {
        "A": [record("NCT1", phase_label="Phase 2")],
        "B": [record("NCT2", phase_label="Phase 2")],
    }
    result = ops.compare(groups, Dimension.phase)
    assert result.rows == [
        {"phase": "Phase 2", "group": "A", "trial_count": 1},
        {"phase": "Phase 2", "group": "B", "trial_count": 1},
    ]


def test_compare_includes_zero_rows_so_a_missing_category_is_visible():
    """If one side has no Phase 3 trials at all, that is the finding - it must
    show as an absent bar, not a missing row."""
    groups = {
        "A": [record("NCT1", phase_label="Phase 3")],
        "B": [record("NCT2", phase_label="Phase 2")],
    }
    result = ops.compare(groups, Dimension.phase)
    assert {"phase": "Phase 3", "group": "B", "trial_count": 0} in result.rows
    assert {"phase": "Phase 2", "group": "A", "trial_count": 0} in result.rows


def test_compare_contributors_are_per_group():
    groups = {"A": [record("NCT1")], "B": [record("NCT2")]}
    result = ops.compare(groups, Dimension.phase)
    assert result.contributors["phase=Phase 2|group=A"] == {"NCT1"}
    assert result.contributors["phase=Phase 2|group=B"] == {"NCT2"}


# --- 6. top_k -------------------------------------------------------------


def test_top_k_folds_the_rest_into_other():
    records = [record("NCT1", countries=["A"]), record("NCT2", countries=["A"])]
    records += [record("NCT3", countries=["B"]), record("NCT4", countries=["C"])]
    result = ops.top_k(records, Dimension.country, k=1)
    assert result.rows == [
        {"country": "A", "trial_count": 2},
        {"country": OTHER_LABEL, "trial_count": 2},
    ]


def test_other_is_still_citable():
    """A fold-up bar that cannot be traced back would be a black hole."""
    records = [record("NCT1", countries=["A"]), record("NCT2", countries=["B"])]
    result = ops.top_k(records, Dimension.country, k=1)
    assert result.contributors["country=Other"] == {"NCT2"}


def test_top_k_adds_no_other_bar_when_there_is_nothing_to_fold():
    """An "Other" bar of zero is noise, so it appears only when k is actually
    smaller than the number of categories."""
    records = [record("NCT1", countries=["A"]), record("NCT2", countries=["B"])]
    result = ops.top_k(records, Dimension.country, k=5)
    assert [r["country"] for r in result.rows] == ["A", "B"]
    assert OTHER_LABEL not in [r["country"] for r in result.rows]


def test_top_k_exactly_k_categories_is_not_folded():
    """The boundary case: k == distinct count must not produce an empty Other."""
    records = [record("NCT1", countries=["A"]), record("NCT2", countries=["B"])]
    result = ops.top_k(records, Dimension.country, k=2)
    assert len(result.rows) == 2


def test_other_counts_distinct_trials_not_a_sum_of_counts():
    """A trial in two folded countries must count once in Other, not twice."""
    records = [record("NCT1", countries=["Keep", "FoldA", "FoldB"])]
    result = ops.top_k(records, Dimension.country, k=1)
    other = next(r for r in result.rows if r["country"] == OTHER_LABEL)
    assert other["trial_count"] == 1


# --- 7. histogram ---------------------------------------------------------


def test_histogram_bins_cover_every_trial():
    records = [record(f"NCT{i}", enrollment=i * 10) for i in range(1, 21)]
    result = ops.histogram(records, Dimension.enrollment)
    assert total(result, "count") + result.dropped == len(records)


def test_histogram_drops_trials_with_no_value():
    records = [record("NCT1", enrollment=100), record("NCT2", enrollment=None)]
    result = ops.histogram(records, Dimension.enrollment)
    assert total(result, "count") == 1
    assert result.dropped == 1


def test_histogram_switches_to_log_bins_when_the_data_is_skewed():
    """Real enrollment is heavily right-skewed: many small trials and a few
    enormous ones. On a linear axis that is one tall bar and empty space."""
    records = [record(f"NCT{i}", enrollment=10) for i in range(50)]
    records.append(record("NCT_BIG", enrollment=50_000))
    result = ops.histogram(records, Dimension.enrollment)
    assert result.extra["log_scale"] is True
    assert result.extra["skew"] > 2


def test_log_bins_are_not_blocked_by_a_zero_enrollment_trial():
    """The bug this test exists for: requiring a positive minimum disabled log
    bins on exactly the data that needs them, because a withdrawn trial
    legitimately enrolls 0."""
    records = [record(f"NCT{i}", enrollment=10) for i in range(50)]
    records.append(record("NCT_ZERO", enrollment=0))
    records.append(record("NCT_BIG", enrollment=50_000))
    result = ops.histogram(records, Dimension.enrollment)
    assert result.extra["log_scale"] is True
    assert total(result, "count") == len(records)  # including the zero


def test_symmetric_data_stays_on_linear_bins():
    records = [record(f"NCT{i}", enrollment=100 + i) for i in range(40)]
    result = ops.histogram(records, Dimension.enrollment)
    assert result.extra["log_scale"] is False


def test_histogram_of_identical_values_produces_one_bin():
    """Zero spread would otherwise be a division by zero."""
    records = [record(f"NCT{i}", enrollment=50) for i in range(10)]
    result = ops.histogram(records, Dimension.enrollment)
    assert total(result, "count") == 10


def test_histogram_bin_count_is_capped():
    records = [record(f"NCT{i}", enrollment=i) for i in range(1, 500)]
    result = ops.histogram(records, Dimension.enrollment)
    assert result.extra["bins"] <= ops.MAX_BINS


def test_largest_value_is_counted_rather_than_falling_off_the_end():
    """The last bin includes its upper edge."""
    records = [record("NCT1", enrollment=0), record("NCT2", enrollment=100)]
    result = ops.histogram(records, Dimension.enrollment, bins=4)
    assert total(result, "count") == 2


# --- 8. scatter -----------------------------------------------------------


def test_scatter_is_one_row_per_trial():
    records = [record("NCT1", enrollment=50, duration_months=12.0)]
    result = ops.scatter(records, Dimension.enrollment, Dimension.duration_months)
    assert result.rows == [{"nct_id": "NCT1", "enrollment": 50.0, "duration_months": 12.0}]


def test_scatter_drops_a_point_missing_either_coordinate():
    """A trial with no enrollment figure is not a trial with no participants,
    so it must not be plotted at zero."""
    records = [
        record("NCT1", enrollment=50, duration_months=12.0),
        record("NCT2", enrollment=None, duration_months=12.0),
        record("NCT3", enrollment=50, duration_months=None),
    ]
    result = ops.scatter(records, Dimension.enrollment, Dimension.duration_months)
    assert len(result.rows) == 1
    assert result.dropped == 2


def test_scatter_can_colour_by_a_categorical_dimension():
    records = [record("NCT1", sponsor_class="INDUSTRY")]
    result = ops.scatter(
        records, Dimension.enrollment, Dimension.duration_months, color=Dimension.sponsor_class
    )
    assert result.rows[0]["color"] == "INDUSTRY"


# --- 9. network -----------------------------------------------------------


def test_bipartite_network_links_the_two_dimensions():
    records = [
        record("NCT1", sponsor="Merck", drugs=["Pembrolizumab"]),
        record("NCT2", sponsor="Merck", drugs=["Pembrolizumab"]),
    ]
    result = ops.network(records, [Dimension.sponsor, Dimension.drug], min_edge_weight=2)
    assert {n["id"] for n in result.extra["nodes"]} == {
        "sponsor:Merck",
        "drug:pembrolizumab",
    }
    assert result.extra["edges"] == [
        {"source": "sponsor:Merck", "target": "drug:pembrolizumab", "weight": 2}
    ]
    assert result.extra["bipartite"] is True


def test_node_ids_are_dimension_qualified():
    """A sponsor and a drug of the same name must be two nodes, not one."""
    records = [record(f"NCT{i}", sponsor="Ozempic", drugs=["Ozempic"]) for i in range(2)]
    result = ops.network(records, [Dimension.sponsor, Dimension.drug], min_edge_weight=2)
    assert {n["id"] for n in result.extra["nodes"]} == {
        "sponsor:Ozempic",
        "drug:ozempic",
    }


def test_cooccurrence_network_needs_two_drugs_in_one_trial():
    """A single-drug trial has no co-occurrence to contribute."""
    records = [record("NCT1", drugs=["Aspirin"])]
    result = ops.network(records, [Dimension.drug, Dimension.drug], min_edge_weight=1)
    assert result.extra["edges"] == []
    assert result.dropped == 1


def test_cooccurrence_network_never_links_a_drug_to_itself():
    records = [record(f"NCT{i}", drugs=["Aspirin", "Aspirin 100 mg"]) for i in range(3)]
    result = ops.network(records, [Dimension.drug, Dimension.drug], min_edge_weight=1)
    # Both names share one key, so there is only one distinct drug: no edge.
    assert all(e["source"] != e["target"] for e in result.extra["edges"])


def test_cooccurrence_network_pairs_distinct_drugs():
    records = [record(f"NCT{i}", drugs=["Aspirin", "Ibuprofen"]) for i in range(2)]
    result = ops.network(records, [Dimension.drug, Dimension.drug], min_edge_weight=2)
    assert len(result.extra["edges"]) == 1
    assert result.extra["edges"][0]["weight"] == 2


def test_min_edge_weight_removes_incidental_pairings():
    """A graph that draws every one-off pairing shows nothing."""
    records = [record("NCT1", sponsor="A", drugs=["X"]), record("NCT2", sponsor="B", drugs=["Y"])]
    result = ops.network(records, [Dimension.sponsor, Dimension.drug], min_edge_weight=2)
    assert result.extra["edges"] == []
    assert result.extra["nodes"] == []


def test_an_empty_graph_reports_what_the_threshold_removed():
    """So the pipeline can explain an empty result instead of rendering a
    blank chart. On a thin sample every pair can occur exactly once."""
    records = [record("NCT1", sponsor="A", drugs=["X"]), record("NCT2", sponsor="B", drugs=["Y"])]
    result = ops.network(records, [Dimension.sponsor, Dimension.drug], min_edge_weight=2)
    assert result.extra["edges_before_threshold"] == 2
    assert result.extra["nodes_before_threshold"] == 4


def test_max_nodes_keeps_the_heaviest():
    records = []
    for i in range(10):
        # Sponsor "Big" appears in every trial; the others once each.
        records.append(record(f"NCT{i}", sponsor="Big", drugs=["X"]))
        records.append(record(f"NCTB{i}", sponsor=f"Small{i}", drugs=["X"]))
    result = ops.network(
        records, [Dimension.sponsor, Dimension.drug], min_edge_weight=1, max_nodes=2
    )
    assert len(result.extra["nodes"]) == 2
    assert "sponsor:Big" in {n["id"] for n in result.extra["nodes"]}


def test_every_edge_endpoint_survives_node_pruning():
    """Nodes are kept before edges precisely so this holds. The response
    schema rejects an orphan edge, so a violation here would be a 500."""
    records = []
    for i in range(20):
        records.append(record(f"NCT{i}", sponsor=f"S{i % 5}", drugs=[f"D{i % 7}"]))
        records.append(record(f"NCTx{i}", sponsor=f"S{i % 5}", drugs=[f"D{i % 7}"]))
    result = ops.network(
        records, [Dimension.sponsor, Dimension.drug], min_edge_weight=2, max_nodes=4
    )
    ids = {n["id"] for n in result.extra["nodes"]}
    assert ids
    for edge in result.extra["edges"]:
        assert edge["source"] in ids and edge["target"] in ids


def test_node_weight_is_trials_not_edges():
    records = [
        record("NCT1", sponsor="Merck", drugs=["A", "B"]),
        record("NCT2", sponsor="Merck", drugs=["A", "B"]),
    ]
    result = ops.network(records, [Dimension.sponsor, Dimension.drug], min_edge_weight=2)
    merck = next(n for n in result.extra["nodes"] if n["id"] == "sponsor:Merck")
    assert merck["weight"] == 2  # two trials, though it has two edges


# --- fixture-wide invariants ----------------------------------------------


def _dedupe(records: list[TrialRecord]) -> list[TrialRecord]:
    by_id = {r.nct_id: r for r in records}
    return list(by_id.values())

# The four fixture searches overlap - a pembrolizumab trial is often also an
# NSCLC trial - so the union of 400 studies holds fewer distinct trials. A real
# search never returns the same trial twice, so dedupe to match reality.
ALL_RECORDS = _dedupe(normalize_studies(all_fixture_studies()))


@pytest.mark.parametrize(
    "dim",
    [
        Dimension.phase,
        Dimension.status,
        Dimension.sponsor_class,
        Dimension.country,
        Dimension.drug,
        Dimension.condition,
        Dimension.intervention_type,
        Dimension.study_type,
    ],
)
def test_count_by_accounts_for_every_record_on_real_data(dim):
    """Either a record contributes to a bar or it is counted as dropped.
    Nothing may simply vanish."""
    result = ops.count_by(ALL_RECORDS, dim)
    contributing = set().union(*result.contributors.values()) if result.contributors else set()
    assert len(contributing) + result.dropped == len(ALL_RECORDS)


@pytest.mark.parametrize(
    "dim", [Dimension.phase, Dimension.status, Dimension.country, Dimension.sponsor]
)
def test_every_datum_has_contributors_on_real_data(dim):
    """An uncitable datum is a broken datum."""
    result = ops.count_by(ALL_RECORDS, dim)
    for row in result.rows:
        key = ops.datum_key(dim, row[dim.value])
        assert result.contributors[key], f"{key} has no contributors"
        assert len(result.contributors[key]) == row["trial_count"]


def test_time_series_on_real_data_has_no_gaps():
    result = ops.time_series(ALL_RECORDS)
    years = [row["year"] for row in result.rows]
    assert years == list(range(years[0], years[-1] + 1))


def test_network_on_real_data_is_renderable():
    """The invariants the response schema will enforce anyway, checked here on
    real data: no orphan edges, no self-loops, threshold respected."""
    records = normalize_studies(fixture_studies("nsclc"))
    result = ops.network(records, [Dimension.drug, Dimension.drug], min_edge_weight=2)
    ids = {n["id"] for n in result.extra["nodes"]}
    assert result.extra["edges"], "expected a non-empty co-occurrence graph"
    for edge in result.extra["edges"]:
        assert edge["source"] in ids and edge["target"] in ids
        assert edge["source"] != edge["target"]
        assert edge["weight"] >= 2


def test_histogram_on_real_data_accounts_for_every_trial():
    result = ops.histogram(ALL_RECORDS, Dimension.enrollment)
    assert total(result, "count") + result.dropped == len(ALL_RECORDS)


def test_the_fixture_searches_overlap():
    """Documents why ALL_RECORDS is deduplicated.

    Four searches returned 400 studies but fewer distinct trials, because the
    same trial matches more than one search. Worth knowing: it is why these
    invariants count distinct NCT IDs rather than rows.
    """
    raw = normalize_studies(all_fixture_studies())
    assert len(raw) == 400
    assert len(ALL_RECORDS) < len(raw)
