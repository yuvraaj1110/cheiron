"""Tests for app/citations/attach.py - the project's hallucination guard.

The central test is `test_every_excerpt_is_verbatim`. It walks every datum of
every chart type built from real fixture data, and asserts that each citation's
excerpt and title appear **literally** in the raw API response for the trial
being cited. If any code on this path ever started summarising, rewording or
reformatting, this test fails.

Two subtleties that took a bug each to learn:

- The comparison dump must use `ensure_ascii=False`. A real trial title
  contains "≥", which json.dumps escapes to \\u2265 by default; the citation
  was correct and the naive check was wrong. A guard that is fooled by its own
  encoding is worse than no guard, because it fails on honest data and would
  invite someone to loosen it.
- `value` is deliberately NOT checked for verbatimness. It is our own bucket
  label ("Phase 1/2"), which by design does not appear in the record. Keeping
  our labels and their text in separate fields is what lets this test be
  strict about the fields that matter.
"""

import json

import pytest

from app.analytics import operators as ops
from app.api.schemas import VizType
from app.citations.attach import as_excerpt, attach, quote
from app.ctgov.normalize import normalize_studies
from app.planner.plan import Dimension, QueryPlan
from app.viz.builder import build_spec
from tests.helpers import fixture_studies


def plan(**kwargs) -> QueryPlan:
    kwargs.setdefault("interpretation", "A test plan.")
    return QueryPlan(**kwargs)


def value_at_path(raw_study: dict, field: str) -> list[str]:
    """Every value the raw record holds at a cited `field` path.

    Returns a list because the paths we emit come in two shapes: a plain
    dotted path to one value, and a path with "[]" marking a list we pulled a
    single entry out of ("...locations[].country"). In the second case every
    entry is returned and the excerpt has to match one of them.

    Checking the path rather than searching the whole record as text is the
    stronger guard, and it sidesteps a trap. Two real fixture trials broke a
    naive substring search: one title contains "≥", which json.dumps escapes
    to \\u2265, and another contains literal double quotes, which the record
    stores as \\". Both citations were correct and the check was wrong. A
    guard that fails on honest data invites someone to loosen it, which is how
    a real guard gets lost. Comparing values at a path has no such ambiguity:
    it proves the excerpt IS what the record says at the place it claims.
    """
    before, _, after = field.partition("[]")
    node = raw_study
    for key in before.split("."):
        if not isinstance(node, dict) or key not in node:
            return []
        node = node[key]

    if not after:  # a plain path to a single value
        return [as_excerpt(node)]

    # A list path. `after` is either empty ("conditions[]") or a field of each
    # entry (".country", ".name", ".type").
    entry_field = after.lstrip(".")
    if not isinstance(node, list):
        return []
    if not entry_field:
        return [as_excerpt(v) for v in node]
    return [as_excerpt(entry[entry_field]) for entry in node if entry_field in entry]


def is_verbatim(citation: dict, raw_study: dict) -> bool:
    """Is the excerpt exactly what the record holds at the path cited?"""
    return citation["excerpt"] in value_at_path(raw_study, citation["field"])


def cited_spec(fixture: str, query_plan: QueryPlan, result_fn, k: int = 3):
    """Build and cite one chart from a fixture, returning everything needed to
    check it against the raw data."""
    studies = fixture_studies(fixture)
    records = normalize_studies(studies)
    by_id = {r.nct_id: r for r in records}
    raw_by_id = {s["protocolSection"]["identificationModule"]["nctId"]: s for s in studies}

    result = result_fn(records)
    spec = attach(build_spec(query_plan, result).spec, result, by_id, query_plan, k=k)
    return spec, result, raw_by_id


# Every chart type, each from real data. Together these cover both the
# row-based specs and the network, and both single- and multi-valued
# dimensions - the multi-valued ones being where a lazy implementation would
# quote the wrong thing.
CASES = {
    "bar_chart (phase)": (
        "pembrolizumab",
        plan(intent="distribution", filters={"drug": "pembrolizumab"}, group_by="phase"),
        lambda r: ops.count_by(r, Dimension.phase),
    ),
    "bar_chart (country, multi-valued)": (
        "diabetes_recruiting",
        plan(intent="geo", filters={"condition": "diabetes"}, group_by="country", top_k=10),
        lambda r: ops.top_k(r, Dimension.country, 10),
    ),
    "bar_chart (drug, grouped keys)": (
        "nsclc",
        plan(intent="distribution", filters={"condition": "NSCLC"}, group_by="drug", top_k=10),
        lambda r: ops.top_k(r, Dimension.drug, 10),
    ),
    "time_series": (
        "pembrolizumab",
        plan(intent="trend", filters={"drug": "pembrolizumab"}, time={"start": 2015}),
        lambda r: ops.time_series(r, start=2015),
    ),
    "stat": (
        "breast_cancer",
        plan(intent="count", filters={"condition": "breast cancer"}),
        lambda r: ops.count(r),
    ),
    "table": (
        "pembrolizumab",
        plan(intent="list", filters={"drug": "pembrolizumab"}),
        lambda r: ops.list_trials(r, limit=10),
    ),
    "histogram": (
        "breast_cancer",
        plan(intent="histogram", filters={"condition": "breast cancer"}, numeric_x="enrollment"),
        lambda r: ops.histogram(r, Dimension.enrollment),
    ),
    "scatter_plot": (
        "nsclc",
        plan(intent="correlation", numeric_x="enrollment", numeric_y="duration_months"),
        lambda r: ops.scatter(r, Dimension.enrollment, Dimension.duration_months),
    ),
    "network_graph": (
        "nsclc",
        plan(
            intent="network",
            filters={"condition": "NSCLC"},
            network={"node_dims": ["drug", "drug"]},
        ),
        lambda r: ops.network(r, [Dimension.drug, Dimension.drug], min_edge_weight=2),
    ),
    "grouped_bar_chart": (
        "pembrolizumab",
        plan(
            intent="comparison",
            group_by="phase",
            compare_groups=[
                {"label": "early", "filters": {"drug": "a"}},
                {"label": "late", "filters": {"drug": "b"}},
            ],
        ),
        lambda r: ops.compare({"early": r[:50], "late": r[50:]}, Dimension.phase),
    ),
}


def all_citations(spec) -> list[dict]:
    """Every citation on a spec, whatever shape it is."""
    if spec.type is VizType.network_graph:
        citations = []
        for node in spec.data.nodes:
            citations += [c.model_dump() for c in node.citations]
        for edge in spec.data.edges:
            citations += [c.model_dump() for c in edge.citations]
        return citations
    return [c for row in spec.data for c in row["citations"]]


# --- THE guard ------------------------------------------------------------


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_every_excerpt_is_verbatim(case):
    """The hallucination guard. No citation text may be written by our code.

    This is the test that has to pass. Every other test in the project checks
    that a number is right; this one checks that the evidence for it is real.
    """
    fixture, query_plan, result_fn = CASES[case]
    spec, _, raw_by_id = cited_spec(fixture, query_plan, result_fn)

    citations = all_citations(spec)
    assert citations, f"{case} produced no citations at all"

    for citation in citations:
        raw = raw_by_id[citation["nct_id"]]
        assert is_verbatim(citation, raw), (
            f"{case}: excerpt {citation['excerpt']!r} is not the value at "
            f"{citation['field']} in {citation['nct_id']}; that path holds "
            f"{value_at_path(raw, citation['field'])!r}"
        )
        assert citation["title"] == raw["protocolSection"]["identificationModule"]["briefTitle"], (
            f"{case}: title for {citation['nct_id']} is not the record's briefTitle"
        )


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_every_cited_field_path_is_real(case):
    """A citation's `field` must name the API path its excerpt came from, so a
    reader can go and check the whole value for themselves."""
    fixture, query_plan, result_fn = CASES[case]
    spec, _, _ = cited_spec(fixture, query_plan, result_fn)

    for citation in all_citations(spec):
        assert citation["field"].startswith("protocolSection.")


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_every_citation_links_to_the_trial_it_cites(case):
    fixture, query_plan, result_fn = CASES[case]
    spec, _, _ = cited_spec(fixture, query_plan, result_fn)

    for citation in all_citations(spec):
        assert citation["url"] == f"https://clinicaltrials.gov/study/{citation['nct_id']}"


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_cited_trials_really_are_contributors(case):
    """A citation must come from the set of trials behind that datum. A
    citation that quotes a real trial which is not actually in the bar would
    be verbatim and still a lie."""
    fixture, query_plan, result_fn = CASES[case]
    spec, result, _ = cited_spec(fixture, query_plan, result_fn)

    every_contributor = set().union(*result.contributors.values())
    for citation in all_citations(spec):
        assert citation["nct_id"] in every_contributor


# --- counts and limits ----------------------------------------------------


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_citation_count_is_the_full_total_not_the_sample(case):
    """citation_count says how many trials are behind the datum; `citations`
    is only the first few. Reporting the sample size as the total would
    understate the evidence."""
    fixture, query_plan, result_fn = CASES[case]
    spec, result, _ = cited_spec(fixture, query_plan, result_fn, k=2)

    if spec.type is VizType.network_graph:
        for node in spec.data.nodes:
            expected = len(result.contributors[ops.node_key(node.id)])
            assert node.citation_count == expected
            assert len(node.citations) <= 2
        return

    for row in spec.data:
        assert row["citation_count"] >= len(row["citations"])
        assert len(row["citations"]) <= 2


def test_citation_count_matches_the_bar_height():
    """For a single-valued dimension the trials behind a bar are exactly the
    bar's height, so the two numbers must agree."""
    spec, _, _ = cited_spec(
        "pembrolizumab",
        plan(intent="distribution", group_by="phase"),
        lambda r: ops.count_by(r, Dimension.phase),
    )
    for row in spec.data:
        assert row["citation_count"] == row["trial_count"]


def test_k_limits_how_many_examples_are_given():
    spec, _, _ = cited_spec(
        "pembrolizumab",
        plan(intent="distribution", group_by="phase"),
        lambda r: ops.count_by(r, Dimension.phase),
        k=1,
    )
    assert all(len(row["citations"]) <= 1 for row in spec.data)


def test_citations_are_newest_first():
    """Deterministic order, and a 2024 trial is better evidence than a 1998
    one when you are checking a claim."""
    studies = fixture_studies("pembrolizumab")
    records = normalize_studies(studies)
    by_id = {r.nct_id: r for r in records}
    result = ops.count_by(records, Dimension.phase)
    query_plan = plan(intent="distribution", group_by="phase")
    spec = attach(build_spec(query_plan, result).spec, result, by_id, query_plan, k=3)

    for row in spec.data:
        years = [by_id[c["nct_id"]].start_year or 0 for c in row["citations"]]
        assert years == sorted(years, reverse=True)


# --- multi-valued dimensions, where it is easy to quote the wrong thing ----


def test_a_country_citation_quotes_that_country_not_the_whole_site_list():
    """A trial runs at many sites. A citation for France must quote "France",
    not every location in the record."""
    spec, _, _ = cited_spec(
        "diabetes_recruiting",
        plan(intent="geo", group_by="country", top_k=5),
        lambda r: ops.top_k(r, Dimension.country, 5),
    )
    for row in spec.data:
        if row["country"] == ops.OTHER_LABEL:
            continue
        for citation in row["citations"]:
            assert citation["excerpt"] == row["country"]
            assert citation["field"].endswith("[].country")


def test_a_drug_citation_quotes_the_api_spelling_not_our_group_key():
    """Several spellings share a group key, but the excerpt must be the name as
    the API wrote it."""
    spec, _, raw_by_id = cited_spec(
        "nsclc",
        plan(intent="distribution", group_by="drug", top_k=10),
        lambda r: ops.top_k(r, Dimension.drug, 10),
    )
    found = False
    for row in spec.data:
        if row["drug"] == ops.OTHER_LABEL:
            continue  # the folded bar is cited at trial level, by design
        for citation in row["citations"]:
            assert is_verbatim(citation, raw_by_id[citation["nct_id"]])
            assert citation["field"].endswith("[].name")
            found = True
    assert found


def test_our_bucket_label_is_not_claimed_to_be_verbatim():
    """"Phase 1/2" is our label and appears nowhere in the record. It lives in
    `value`, which is exactly why `excerpt` can be held to a strict standard."""
    spec, _, raw_by_id = cited_spec(
        "pembrolizumab",
        plan(intent="distribution", group_by="phase"),
        lambda r: ops.count_by(r, Dimension.phase),
    )
    combination_rows = [r for r in spec.data if "/" in r["phase"] and r["citations"]]
    assert combination_rows, "expected a combination phase bucket in this fixture"

    for row in combination_rows:
        for citation in row["citations"]:
            raw = raw_by_id[citation["nct_id"]]
            # `value` is our bucket label; `excerpt` is the API's raw codes.
            # They must not be the same text, which is the whole point of
            # keeping them in separate fields.
            assert citation["value"] == row["phase"]
            assert citation["excerpt"] != citation["value"]
            assert "PHASE" in citation["excerpt"]  # the raw codes, not our label
            assert is_verbatim(citation, raw)


# --- the network, where citations go on two kinds of thing -----------------


def test_network_nodes_and_edges_are_both_cited():
    spec, _, _ = cited_spec(
        "nsclc",
        plan(intent="network", network={"node_dims": ["drug", "drug"]}),
        lambda r: ops.network(r, [Dimension.drug, Dimension.drug], min_edge_weight=2),
    )
    assert spec.data.nodes and spec.data.edges
    assert all(n.citations for n in spec.data.nodes)
    assert all(e.citations for e in spec.data.edges)


def test_an_edge_is_cited_by_trials_containing_both_endpoints():
    """The edge's claim is that both things appear in the same trial, so every
    cited trial must actually contain both."""
    studies = fixture_studies("nsclc")
    records = normalize_studies(studies)
    by_id = {r.nct_id: r for r in records}
    result = ops.network(records, [Dimension.drug, Dimension.drug], min_edge_weight=2)
    query_plan = plan(intent="network", network={"node_dims": ["drug", "drug"]})
    spec = attach(build_spec(query_plan, result).spec, result, by_id, query_plan)

    for edge in spec.data.edges:
        source_key = edge.source.split(":", 1)[1]
        target_key = edge.target.split(":", 1)[1]
        for citation in edge.citations:
            keys = {i.key for i in by_id[citation.nct_id].interventions}
            assert {source_key, target_key} <= keys


# --- the rendering helper -------------------------------------------------


def test_a_string_value_is_quoted_unchanged():
    assert as_excerpt("RECRUITING") == "RECRUITING"


def test_a_list_is_serialised_rather_than_flattened_into_prose():
    """json.dumps output is the fragment that appears in the serialised study,
    so a two-phase trial stays checkable instead of becoming "Phase 1 and 2"."""
    assert as_excerpt(["PHASE1", "PHASE2"]) == '["PHASE1", "PHASE2"]'


def test_a_number_is_rendered_as_it_appears_in_the_json():
    assert as_excerpt(54) == "54"


def test_a_record_with_nothing_to_quote_is_skipped_not_described():
    """Rather than inventing a citation for a field the record lacks."""
    from app.ctgov.normalize import normalize_study

    bare = normalize_study({"protocolSection": {"identificationModule": {"nctId": "NCT1"}}})
    assert quote(bare, Dimension.phase, "Phase 2") is None
    assert quote(bare, Dimension.country, "France") is None
    # But it can still cite itself, because the NCT ID is present.
    assert quote(bare, None, "NCT1") == (
        "protocolSection.identificationModule.nctId",
        "NCT1",
    )


def test_trial_level_data_cites_the_trial_itself():
    """A table row's claim is "this trial is in the result", and the NCT ID is
    what proves it - not some unrelated field like status."""
    spec, _, _ = cited_spec(
        "pembrolizumab",
        plan(intent="list", filters={"drug": "pembrolizumab"}),
        lambda r: ops.list_trials(r, limit=5),
    )
    for row in spec.data:
        citation = row["citations"][0]
        assert citation["nct_id"] == row["nct_id"]
        assert citation["excerpt"] == row["nct_id"]
        assert citation["field"].endswith("nctId")


def test_the_other_bar_is_cited_at_trial_level():
    """"Other" is a bar we created by folding categories together, so it is not
    a value of the dimension and cannot be quoted as one. Before this was
    handled, the citation quoted the trial's entire site list - a 60 KB
    "excerpt" that supported nothing.
    """
    spec, _, raw_by_id = cited_spec(
        "diabetes_recruiting",
        plan(intent="geo", group_by="country", top_k=5),
        lambda r: ops.top_k(r, Dimension.country, 5),
    )
    other = next(row for row in spec.data if row["country"] == ops.OTHER_LABEL)
    assert other["citations"], "the Other bar must stay traceable"
    for citation in other["citations"]:
        assert citation["field"].endswith("nctId")
        assert citation["excerpt"] == citation["nct_id"]
        assert is_verbatim(citation, raw_by_id[citation["nct_id"]])


def test_no_excerpt_is_absurdly_long():
    """A citation is evidence a human reads. The bug that motivated this
    quoted every location in a trial, which is not a quote, it is a dump."""
    for case, (fixture, query_plan, result_fn) in CASES.items():
        spec, _, _ = cited_spec(fixture, query_plan, result_fn)
        for citation in all_citations(spec):
            assert len(citation["excerpt"]) < 300, f"{case}: {citation['excerpt'][:80]}..."
