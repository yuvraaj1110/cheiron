"""Tests for app/ctgov/normalize.py.

The broad test is the important one: every study in all four fixtures must
normalize without raising. 400 real records include studies with no phases, no
interventions, no locations, no enrollment and no start date, so that single
assertion covers more awkward shapes than any hand-written case could.

The narrow tests pin the rules that a reader of the chart depends on: which
phase bucket a trial lands in, how a partial date is read, that a trial is
counted once per country, and that dose text does not split one drug into
several.
"""

from datetime import date

import pytest
from tests.helpers import FIXTURE_NAMES, all_fixture_studies, fixture_studies

from app.ctgov.normalize import (
    NO_PHASE_LABEL,
    PHASE_ORDER,
    drug_key,
    months_between,
    normalize_studies,
    normalize_study,
    parse_partial_date,
    phase_label,
    phase_sort_key,
)

# --- the broad test -------------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_fixture_study_normalizes(name):
    """No study in any fixture may raise, and none may be silently dropped."""
    studies = fixture_studies(name)
    records = normalize_studies(studies)
    assert len(records) == len(studies)
    for record in records:
        assert record.nct_id.startswith("NCT")
        assert record.phase_label in PHASE_ORDER
        assert record.url == f"https://clinicaltrials.gov/study/{record.nct_id}"


def test_all_eight_phase_buckets_occur_in_the_fixtures():
    """Justifies having eight buckets rather than six: the two combination
    buckets and Early Phase 1 are all present in real data."""
    labels = {r.phase_label for r in normalize_studies(all_fixture_studies())}
    assert labels == set(PHASE_ORDER)


def test_records_with_missing_modules_still_normalize():
    """Studies missing whole modules are common, not an edge case."""
    records = normalize_studies(all_fixture_studies())
    assert any(not r.interventions for r in records), "expected some with no interventions"
    assert any(not r.countries for r in records), "expected some with no locations"
    assert any(r.enrollment is None for r in records), "expected some with no enrollment"
    assert any(r.start_year is None for r in records), "expected some with no start date"


def test_a_bare_study_normalizes_to_empty_values():
    """The extreme case: nothing but an ID."""
    record = normalize_study(
        {"protocolSection": {"identificationModule": {"nctId": "NCT00000000"}}}
    )
    assert record.nct_id == "NCT00000000"
    assert record.phase_label == NO_PHASE_LABEL
    assert record.start_date is None
    assert record.duration_months is None
    assert record.countries == []
    assert record.interventions == []


def test_study_without_an_nct_id_is_dropped():
    """A record with no ID could not be cited or linked, so it cannot appear
    in an answer."""
    assert normalize_studies([{"protocolSection": {}}]) == []


# --- phase bucketing ------------------------------------------------------


@pytest.mark.parametrize(
    "phases,expected",
    [
        (None, "Not applicable"),  # key absent: every observational study
        ([], "Not applicable"),
        (["NA"], "Not applicable"),  # explicit NA means the same to a reader
        (["EARLY_PHASE1"], "Early Phase 1"),
        (["PHASE1"], "Phase 1"),
        (["PHASE1", "PHASE2"], "Phase 1/2"),
        (["PHASE2"], "Phase 2"),
        (["PHASE2", "PHASE3"], "Phase 2/3"),
        (["PHASE3"], "Phase 3"),
        (["PHASE4"], "Phase 4"),
        (["PHASE2", "PHASE1"], "Phase 1/2"),  # order in the API list is not fixed
    ],
)
def test_phase_label(phases, expected):
    assert phase_label(phases) == expected


def test_unknown_phase_combination_stays_visible():
    """An unseen combination is built from its parts rather than hidden in an
    'Other' bucket, so an odd trial is still readable on the chart."""
    assert phase_label(["PHASE1", "PHASE3"]) == "Phase 1/3"


def test_phase_sort_order_is_chart_order_not_alphabetical():
    """'Phase 1/2' belongs between 1 and 2, and 'Not applicable' at the end -
    neither of which alphabetical sorting gives you."""
    shuffled = ["Phase 4", "Not applicable", "Phase 1/2", "Phase 1", "Early Phase 1"]
    assert sorted(shuffled, key=phase_sort_key) == [
        "Early Phase 1",
        "Phase 1",
        "Phase 1/2",
        "Phase 4",
        "Not applicable",
    ]


def test_every_phase_bucket_sorts_before_an_unknown_one():
    assert phase_sort_key("Phase 1/3") > phase_sort_key("Not applicable")


# --- dates ----------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2019-09-15", date(2019, 9, 15)),
        ("2019-09", date(2019, 9, 1)),  # partial: up to 39% of a result
        ("2019", date(2019, 1, 1)),
        (None, None),
        ("", None),
        ("not a date", None),
        ("2019-13", None),  # month 13 is not a date
    ],
)
def test_parse_partial_date(value, expected):
    assert parse_partial_date(value) == expected


def test_partial_dates_are_common_in_the_fixtures():
    """Confirms the rule is on a main path, not an edge case."""
    studies = fixture_studies("nsclc")
    partial = sum(
        1
        for s in studies
        if len(s["protocolSection"]["statusModule"].get("startDateStruct", {}).get("date", "")) == 7
    )
    assert partial > 10


@pytest.mark.parametrize(
    "start,end,expected",
    [
        ("2019-01-01", "2020-01-01", 12.0),
        ("2019-09-15", "2022-06-08", 32.8),
        ("2019-01-01", None, None),
        (None, "2020-01-01", None),
        ("2020-01-01", "2019-01-01", None),  # end before start: not negative, None
    ],
)
def test_months_between(start, end, expected):
    assert months_between(parse_partial_date(start), parse_partial_date(end)) == expected


def test_start_year_comes_from_the_start_date():
    record = normalize_study(
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT1"},
                "statusModule": {"startDateStruct": {"date": "2019-09"}},
            }
        }
    )
    assert record.start_date == date(2019, 9, 1)
    assert record.start_year == 2019


# --- countries ------------------------------------------------------------


def test_a_trial_counts_once_per_distinct_country():
    """Three sites in France and one in Germany is two countries, not four."""
    record = normalize_study(
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT1"},
                "contactsLocationsModule": {
                    "locations": [
                        {"facility": "A", "city": "Lyon", "country": "France"},
                        {"facility": "B", "city": "Paris", "country": "France"},
                        {"facility": "C", "city": "Nice", "country": "France"},
                        {"facility": "D", "city": "Berlin", "country": "Germany"},
                    ]
                },
            }
        }
    )
    assert record.countries == ["France", "Germany"]  # sorted and distinct


def test_locations_without_a_country_are_skipped():
    record = normalize_study(
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT1"},
                "contactsLocationsModule": {
                    "locations": [{"facility": "A"}, {"country": "Japan"}]
                },
            }
        }
    )
    assert record.countries == ["Japan"]


def test_country_names_are_kept_verbatim():
    """Full names, not ISO codes. A lookup table would be the one place the
    system could invent a value, and the citation excerpt must match the
    record exactly."""
    records = normalize_studies(all_fixture_studies())
    countries = {c for r in records for c in r.countries}
    assert "United States" in countries
    assert not any(len(c) == 2 for c in countries), "no ISO alpha-2 codes"


# --- drug names -----------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Pembrolizumab", "pembrolizumab"),
        ("pembrolizumab", "pembrolizumab"),
        ("Pembrolizumab 200 mg", "pembrolizumab"),
        ("PEMBROLIZUMAB 200MG Q3W", "pembrolizumab"),
        ("pembrolizumab (MK-3475)", "pembrolizumab"),
        ("Pembrolizumab IV infusion", "pembrolizumab"),
        ("Pembrolizumab, oral tablet", "pembrolizumab"),
        ("Nivolumab 3 mg/kg", "nivolumab"),
    ],
)
def test_drug_key_strips_dose_and_route(name, expected):
    """All of these are one drug. Without this they would be eight bars."""
    assert drug_key(name) == expected


def test_drug_key_keeps_something_when_the_name_is_only_dose_text():
    """Otherwise every such name would collapse into one empty-string bucket."""
    assert drug_key("200 mg") == "200 mg"


def test_display_label_is_untouched():
    """The key is for grouping; the label the user sees stays verbatim."""
    record = normalize_study(
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT1"},
                "armsInterventionsModule": {
                    "interventions": [{"type": "DRUG", "name": "Pembrolizumab 200 mg"}]
                },
            }
        }
    )
    assert record.interventions[0].name == "Pembrolizumab 200 mg"
    assert record.interventions[0].key == "pembrolizumab"


def test_dose_variants_in_the_fixtures_collapse_to_one_key():
    """The payoff, on real data: several spellings, one group."""
    records = normalize_studies(fixture_studies("pembrolizumab"))
    names = {i.name for r in records for i in r.interventions if i.key == "pembrolizumab"}
    assert len(names) > 1, "expected several spellings of the same drug"
    assert all("pembrolizumab" in n.lower() for n in names)


def test_combination_arm_names_are_a_known_limitation():
    """Documents what this does NOT do.

    An arm called "Durvalumab or Pembrolizumab" is one intervention record in
    the API, so it becomes one key and does not count towards either drug on
    its own. Splitting on "or" and "+" would be guesswork about trial design,
    so the limitation is recorded instead of papered over.
    """
    assert drug_key("Durvalumab or Pembrolizumab") == "durvalumab or pembrolizumab"


def test_interventions_without_a_name_are_skipped():
    record = normalize_study(
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT1"},
                "armsInterventionsModule": {
                    "interventions": [{"type": "DRUG"}, {"type": "DRUG", "name": "Aspirin"}]
                },
            }
        }
    )
    assert [i.name for i in record.interventions] == ["Aspirin"]


# --- sources, which the citations depend on -------------------------------


def test_sources_record_the_path_and_the_raw_value():
    """This is what lets a citation quote the record instead of describing it."""
    study = fixture_studies("pembrolizumab")[0]
    record = normalize_study(study)

    assert record.sources["nct_id"].path == "protocolSection.identificationModule.nctId"
    assert record.sources["nct_id"].raw_value == record.nct_id
    assert record.sources["status"].raw_value == record.status


def test_absent_fields_are_left_out_of_sources():
    """So the citation attacher can tell 'no value' from 'value was null'."""
    record = normalize_study(
        {"protocolSection": {"identificationModule": {"nctId": "NCT1"}}}
    )
    assert "phase" not in record.sources
    assert "enrollment" not in record.sources
    assert record.sources["nct_id"].raw_value == "NCT1"


def test_raw_phase_value_is_kept_for_citation():
    """The bucket label is ours; the excerpt must be the API's own string."""
    record = normalize_study(
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT1"},
                "designModule": {"phases": ["PHASE1", "PHASE2"]},
            }
        }
    )
    assert record.phase_label == "Phase 1/2"  # ours, for the chart
    assert record.sources["phase"].raw_value == ["PHASE1", "PHASE2"]  # theirs, for the citation
