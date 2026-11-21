"""Attach citations to every datum in a spec.

This is the file that makes the project's central claim checkable. Every bar,
point, node and edge gets a count of the trials behind it and a few named
examples, each one a link to the study page and a quote from the API record.

The rule here is absolute and it is why `excerpt` exists: **no text in a
citation is written by this code**. An excerpt is a value copied out of the raw
API record, unchanged - not summarised, not reworded, not reformatted. A test
asserts that every excerpt on every datum appears verbatim in the raw study
JSON, and that test is the project's hallucination guard. If this file ever
started composing prose, that test would fail.

`value` is ours (the bucket label, e.g. "Phase 1/2"); `excerpt` and `title`
are theirs. Keeping the two apart is what lets the guard be strict.
"""

import json
from typing import Any

from app.analytics import operators as ops
from app.analytics.operators import OperatorResult
from app.api.schemas import Citation, VizSpec, VizType
from app.ctgov.normalize import TrialRecord
from app.planner.plan import Dimension, QueryPlan

DEFAULT_CITATIONS_PER_DATUM = 3

# Which stored source a dimension's excerpt is quoted from. The paths
# themselves live in normalize.SOURCE_PATHS, so there is one definition of
# where each value came from.
DIMENSION_SOURCE: dict[Dimension, str] = {
    Dimension.phase: "phase",
    Dimension.status: "status",
    Dimension.study_type: "study_type",
    Dimension.sponsor: "sponsor",
    Dimension.sponsor_class: "sponsor_class",
    Dimension.start_year: "start_date",
    Dimension.country: "country",
    Dimension.condition: "condition",
    Dimension.drug: "intervention",
    Dimension.intervention_type: "intervention",
    Dimension.enrollment: "enrollment",
    Dimension.duration_months: "completion_date",
}


def as_excerpt(raw: Any) -> str:
    """Render a raw API value as a string without altering it.

    A string is used as-is. Anything else is serialised with json.dumps, whose
    output for a list is exactly the fragment that appears in the serialised
    study - so `["PHASE1", "PHASE2"]` stays checkable rather than being
    flattened into prose like "Phase 1 and 2".
    """
    if isinstance(raw, str):
        return raw
    return json.dumps(raw)


def quote(record: TrialRecord, dim: Dimension | None, value: str) -> tuple[str, str] | None:
    """The (api_path, excerpt) proving this record belongs to this datum.

    `dim` is None when the datum is a trial in its own right - a table row, a
    scatter point, or one of the trials behind a total. The claim then is
    simply "this trial is in the result", and the thing that proves it is the
    record's own NCT ID, quoted from the identification module.

    For a multi-valued dimension the narrow matching value is quoted rather
    than the whole list: a citation for France quotes "France", not every
    site in the trial. Returns None when the record holds nothing to quote,
    which is how a record that cannot be cited is skipped rather than
    described.
    """
    if dim is None:
        source = record.sources.get("nct_id")
        return (source.path, as_excerpt(source.raw_value)) if source else None

    source_name = DIMENSION_SOURCE.get(dim)
    source = record.sources.get(source_name) if source_name else None
    if source is None:
        return None

    # Multi-valued dimensions: quote the one value that matched, and refuse if
    # none did. Falling through to the single-valued path below would quote the
    # entire list - every site in the trial - which does not support the claim.
    if dim is Dimension.country:
        return (f"{source.path}[].country", value) if value in record.countries else None
    if dim is Dimension.condition:
        return (f"{source.path}[]", value) if value in record.conditions else None
    if dim is Dimension.drug:
        for intervention in record.interventions:
            if intervention.name == value or intervention.key == value:
                return f"{source.path}[].name", intervention.name
        return None
    if dim is Dimension.intervention_type:
        for intervention in record.interventions:
            if intervention.type == value:
                return f"{source.path}[].type", intervention.type
        return None

    # Single-valued: the stored raw value is already the narrow one.
    return source.path, as_excerpt(source.raw_value)


def cite(
    nct_ids: set[str],
    records_by_id: dict[str, TrialRecord],
    dim: Dimension | None,
    value: str,
    k: int,
) -> list[Citation]:
    """Up to k citations for one datum, most recently started first.

    Newest first because a 2024 trial is more useful evidence than a 1998 one
    when you are checking a claim, and because a stable order makes the
    examples reproducible rather than whatever the set happened to yield.
    """
    records = [records_by_id[i] for i in nct_ids if i in records_by_id]
    records.sort(key=lambda r: (r.start_year or 0, r.nct_id), reverse=True)

    citations = []
    for record in records:
        if len(citations) == k:
            break
        quoted = quote(record, dim, value)
        if quoted is None:
            continue
        path, excerpt = quoted
        citations.append(
            Citation(
                nct_id=record.nct_id,
                title=record.title,
                field=path,
                value=value if dim is not None else record.nct_id,
                excerpt=excerpt,
                url=record.url,
            )
        )
    return citations


# --------------------------------------------------------------------------
# Per-chart wiring
# --------------------------------------------------------------------------
# Each chart shape stores its data differently, so each needs to know which
# datum key and which dimension a row corresponds to. The keys are built with
# the same functions the operators used, so they cannot drift apart.


def _attach_rows(
    spec: VizSpec,
    result: OperatorResult,
    records_by_id: dict[str, TrialRecord],
    plan: QueryPlan,
    k: int,
) -> None:
    """Add citations to each row of a row-based chart, in place."""
    dim = _datum_dimension(plan, spec.type)

    for row in spec.data:
        key = _row_key(row, spec.type, plan, dim)
        contributors = result.contributors.get(key, set())
        value = _row_value(row, spec.type, plan, dim)

        # "Other" is a bar we created by folding categories together, so it is
        # not a value of the dimension and cannot be quoted as one. Its trials
        # are cited at trial level instead, which keeps the bar traceable.
        row_dim = None if value == ops.OTHER_LABEL else dim

        row["citation_count"] = len(contributors)
        citations = cite(contributors, records_by_id, row_dim, value, k)
        row["citations"] = [c.model_dump() for c in citations]


def _datum_dimension(plan: QueryPlan, viz_type: VizType) -> Dimension | None:
    """Which dimension a datum of this chart is about, and so which field its
    excerpt is quoted from. None means the datum is a trial in its own right.
    """
    if viz_type in (VizType.stat, VizType.table, VizType.scatter_plot):
        return None  # the datum IS a trial, so cite the trial itself
    if viz_type is VizType.time_series:
        return Dimension.start_year
    if viz_type is VizType.histogram:
        return plan.numeric_x or Dimension.enrollment
    return plan.group_by or Dimension.phase


def _row_value(row: dict, viz_type: VizType, plan: QueryPlan, dim: Dimension | None) -> str:
    """The datum's own label, used as Citation.value."""
    if viz_type is VizType.stat:
        return str(row.get("trial_count"))
    if viz_type in (VizType.table, VizType.scatter_plot):
        return row["nct_id"]
    if viz_type is VizType.time_series:
        return str(row["year"])
    if viz_type is VizType.histogram:
        return f"{row['bin_start']}-{row['bin_end']}"
    return str(row.get(dim.value))


def _row_key(row: dict, viz_type: VizType, plan: QueryPlan, dim: Dimension | None) -> str:
    if viz_type is VizType.stat:
        return ops.TOTAL_KEY
    if viz_type in (VizType.table, VizType.scatter_plot):
        return ops.trial_key(row["nct_id"])
    if viz_type is VizType.time_series:
        series = row.get("series")
        return ops.year_key(row["year"], series)
    if viz_type is VizType.histogram:
        return ops.bin_key(row["bin_start"], row["bin_end"])
    if viz_type is VizType.grouped_bar_chart:
        return ops.group_key(dim, row[dim.value], row["group"])
    return ops.datum_key(dim, row[dim.value])


def _attach_network(
    spec: VizSpec,
    result: OperatorResult,
    records_by_id: dict[str, TrialRecord],
    plan: QueryPlan,
    k: int,
) -> None:
    """Citations go on both nodes and edges.

    A node is cited on its own dimension (a sponsor node quotes the sponsor
    field); an edge is cited on the dimension of its target, since that is the
    narrower claim - "this sponsor ran a trial using this drug".
    """
    left, right = plan.network.node_dims
    dim_of_type = {left.value: left, right.value: right}
    nodes_by_id = {node.id: node for node in spec.data.nodes}

    for node in spec.data.nodes:
        contributors = result.contributors.get(ops.node_key(node.id), set())
        node.citation_count = len(contributors)
        node.citations = cite(contributors, records_by_id, dim_of_type[node.type], node.label, k)

    for edge in spec.data.edges:
        contributors = result.contributors.get(ops.edge_key(edge.source, edge.target), set())
        target = nodes_by_id[edge.target]
        edge.citation_count = len(contributors)
        edge.citations = cite(
            contributors, records_by_id, dim_of_type[target.type], target.label, k
        )


def attach(
    spec: VizSpec,
    result: OperatorResult,
    records_by_id: dict[str, TrialRecord],
    plan: QueryPlan,
    k: int = DEFAULT_CITATIONS_PER_DATUM,
) -> VizSpec:
    """Add citation_count and citations to every datum. Returns the same spec.

    Mutates rather than rebuilding because the spec has already been validated
    and only these two keys are being added; rebuilding would mean re-deriving
    the whole thing for no gain.
    """
    if spec.type is VizType.network_graph:
        _attach_network(spec, result, records_by_id, plan, k)
    else:
        _attach_rows(spec, result, records_by_id, plan, k)
    return spec
