"""The nine operators: every number the API ever returns is computed here.

All of them are pure functions over a list of TrialRecord. No I/O, no LLM, no
globals. That is the whole point of the project's design: the model chooses
which operator runs, and this file decides what the answer is.

Every operator returns an OperatorResult with two parts that matter:

- `rows`, the data the chart draws, and
- `contributors`, a map from each datum to the set of NCT IDs behind it.

`contributors` is what makes citations possible. A bar is not just "49" - it
is 49 named trials, and the citation attacher turns any of them into a link
you can click and check. A number with no contributors could not be cited, so
an operator that forgot to record them would fail its own test.

`dropped` counts records that had no value for the dimension asked about. It
is never silently discarded: the pipeline turns it into a line in
notes.assumptions, so a chart of 380 trials never claims to describe 400.
"""

import math
from collections import defaultdict

from pydantic import BaseModel, Field

from app.ctgov.normalize import TrialRecord, phase_sort_key
from app.planner.plan import DIMENSIONS, Dimension, is_multi_valued

# How many trials a table shows, and how many nodes a network keeps, unless
# asked otherwise.
DEFAULT_LIST_LIMIT = 50
DEFAULT_MAX_NODES = 60


class OperatorResult(BaseModel):
    """What every operator returns."""

    rows: list[dict] = Field(default_factory=list)

    # datum key -> the NCT IDs behind it, e.g. "phase=Phase 3" -> {"NCT01", ...}
    contributors: dict[str, set[str]] = Field(default_factory=dict)

    # records that had no value for the dimension in question
    dropped: int = 0

    # anything chart-specific: network nodes and edges, histogram bin width
    extra: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Reading a dimension off a record
# --------------------------------------------------------------------------


def dimension_values(record: TrialRecord, dim: Dimension) -> list[tuple[str, str]]:
    """Every (group_key, display_label) pair this record contributes.

    Returns a list because of multi-valued dimensions: a trial in twelve
    countries contributes twelve pairs, and an empty list means the record has
    no value for this dimension and should be counted as dropped.

    Key and label differ only for `drug`, where several spellings of one drug
    share a key but the label stays exactly as the API wrote it - so a
    citation can quote it verbatim.
    """
    if dim is Dimension.phase:
        return [(record.phase_label, record.phase_label)]
    if dim is Dimension.status:
        return [(record.status, record.status)] if record.status else []
    if dim is Dimension.study_type:
        return [(record.study_type, record.study_type)] if record.study_type else []
    if dim is Dimension.sponsor:
        return [(record.sponsor, record.sponsor)] if record.sponsor else []
    if dim is Dimension.sponsor_class:
        return [(record.sponsor_class, record.sponsor_class)] if record.sponsor_class else []
    if dim is Dimension.start_year:
        return [(str(record.start_year), str(record.start_year))] if record.start_year else []
    if dim is Dimension.country:
        return [(c, c) for c in record.countries]
    if dim is Dimension.condition:
        return [(c, c) for c in record.conditions]
    if dim is Dimension.intervention_type:
        return sorted({(i.type, i.type) for i in record.interventions})
    if dim is Dimension.drug:
        # One pair per distinct key, labelled by the first spelling seen.
        seen: dict[str, str] = {}
        for intervention in record.interventions:
            seen.setdefault(intervention.key, intervention.name)
        return sorted(seen.items())
    if dim in (Dimension.enrollment, Dimension.duration_months):
        value = numeric_value(record, dim)
        return [(str(value), str(value))] if value is not None else []
    raise ValueError(f"no accessor for dimension {dim}")  # unreachable: enum is closed


def numeric_value(record: TrialRecord, dim: Dimension) -> float | None:
    """The numeric dimensions, for histograms and scatter axes."""
    if dim is Dimension.enrollment:
        return None if record.enrollment is None else float(record.enrollment)
    if dim is Dimension.duration_months:
        return record.duration_months
    raise ValueError(f"{dim.value} is not numeric")


# --- datum keys -----------------------------------------------------------
# Each chart shape has its own key format. They live here, as functions,
# because the operators write these keys and the citation attacher reads them:
# two hand-rolled copies of the same f-string would eventually disagree and
# the citations would silently come back empty.

TOTAL_KEY = "total"


def datum_key(dim: Dimension, value: str) -> str:
    """One category of one dimension, e.g. "phase=Phase 3".

    Dimension-qualified so two charts in one response cannot collide.
    """
    return f"{dim.value}={value}"


def trial_key(nct_id: str) -> str:
    """One row that is itself a trial, as in a table or a scatter point."""
    return f"nct_id={nct_id}"


def year_key(year: int, series: str | None = None) -> str:
    """One point on a time axis, optionally on one of several lines."""
    return f"year={year}" if series is None else f"year={year}|series={series}"


def group_key(dim: Dimension, value: str, group: str) -> str:
    """One bar of one group in a comparison."""
    return f"{dim.value}={value}|group={group}"


def bin_key(start: float, end: float) -> str:
    """One bar of a histogram, identified by its edges."""
    return f"bin={start}-{end}"


def node_key(node_id: str) -> str:
    return f"node={node_id}"


def edge_key(source: str, target: str) -> str:
    return f"edge={source}|{target}"


def sort_key_for(dim: Dimension):
    """How to order the categories of this dimension on an axis.

    Phase has a meaningful order that is not alphabetical; years are numbers;
    everything else is alphabetical as a tiebreaker after count.
    """
    if dim is Dimension.phase:
        return phase_sort_key
    if dim is Dimension.start_year:
        return lambda value: int(value)
    return lambda value: value


# --------------------------------------------------------------------------
# 1. count
# --------------------------------------------------------------------------


def count(records: list[TrialRecord]) -> OperatorResult:
    """How many trials. One row, one number.

    Even this records contributors, so a bare "412" can still be traced to the
    trials behind it.
    """
    return OperatorResult(
        rows=[{"trial_count": len(records)}],
        contributors={TOTAL_KEY: {r.nct_id for r in records}},
    )


# --------------------------------------------------------------------------
# 2. list
# --------------------------------------------------------------------------

LIST_COLUMNS = ["nct_id", "title", "phase_label", "status", "sponsor", "start_year"]


def list_trials(records: list[TrialRecord], limit: int = DEFAULT_LIST_LIMIT) -> OperatorResult:
    """The trials themselves, newest first.

    Newest first because "which trials are there" almost always means "what is
    happening now"; and a stable order matters once the list is truncated.
    """
    ordered = sorted(records, key=lambda r: (r.start_year or 0), reverse=True)
    shown = ordered[:limit]

    rows = [
        {
            "nct_id": r.nct_id,
            "title": r.title,
            "phase_label": r.phase_label,
            "status": r.status,
            "sponsor": r.sponsor,
            "start_year": r.start_year,
        }
        for r in shown
    ]
    return OperatorResult(
        rows=rows,
        # Each row is one trial, so it is its own sole contributor.
        contributors={trial_key(r.nct_id): {r.nct_id} for r in shown},
        extra={"columns": LIST_COLUMNS, "shown": len(shown), "available": len(records)},
    )


# --------------------------------------------------------------------------
# 3. count_by
# --------------------------------------------------------------------------


def count_by(records: list[TrialRecord], dim: Dimension) -> OperatorResult:
    """Trials per category.

    For a multi-valued dimension a trial is counted once per distinct value,
    so a trial running in France and Germany adds one to each. The counts then
    sum to more than the number of trials, which is correct but surprising, so
    the pipeline says so in notes.assumptions.
    """
    counts: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, str] = {}
    dropped = 0

    for record in records:
        pairs = dimension_values(record, dim)
        if not pairs:
            dropped += 1
            continue
        for key, label in pairs:
            counts[key].add(record.nct_id)
            labels.setdefault(key, label)

    # Biggest first, then by the dimension's own order for ties. Readers scan
    # a bar chart from the top, so count has to lead.
    order = sorted(counts, key=lambda k: (-len(counts[k]), sort_key_for(dim)(k)))

    rows = [{dim.value: labels[key], "trial_count": len(counts[key])} for key in order]
    contributors = {datum_key(dim, labels[key]): counts[key] for key in order}

    return OperatorResult(
        rows=rows,
        contributors=contributors,
        dropped=dropped,
        extra={"multi_valued": is_multi_valued(dim), "distinct": len(counts)},
    )


# --------------------------------------------------------------------------
# 4. time_series
# --------------------------------------------------------------------------


def time_series(
    records: list[TrialRecord],
    start: int | None = None,
    end: int | None = None,
    series_by: Dimension | None = None,
) -> OperatorResult:
    """Trials per start year, with the gaps filled in.

    Filling missing years with 0 is the point of this operator. Without it a
    year in which nothing started simply vanishes from the axis, and the line
    joins 2017 straight to 2019 - which reads as continuity where there was a
    gap. Trials with no start date cannot be placed on a time axis at all, so
    they are dropped and counted.
    """
    dated = [r for r in records if r.start_year is not None]
    dropped = len(records) - len(dated)

    if not dated:
        return OperatorResult(dropped=dropped)

    # The window: what was asked for, else what the data covers.
    first = start if start is not None else min(r.start_year for r in dated)
    last = end if end is not None else max(r.start_year for r in dated)
    in_window = [r for r in dated if first <= r.start_year <= last]
    dropped += len(dated) - len(in_window)
    years = list(range(first, last + 1))

    if series_by is None:
        buckets: dict[int, set[str]] = defaultdict(set)
        for record in in_window:
            buckets[record.start_year].add(record.nct_id)

        rows = [{"year": y, "trial_count": len(buckets[y])} for y in years]
        contributors = {year_key(y): buckets[y] for y in years if buckets[y]}
        return OperatorResult(rows=rows, contributors=contributors, dropped=dropped)

    # One line per series value, each with the same filled year axis, so the
    # lines stay comparable.
    buckets_by_series: dict[tuple[int, str], set[str]] = defaultdict(set)
    series_labels: dict[str, str] = {}
    for record in in_window:
        for key, label in dimension_values(record, series_by):
            buckets_by_series[(record.start_year, key)].add(record.nct_id)
            series_labels.setdefault(key, label)

    rows = []
    contributors = {}
    for key in sorted(series_labels, key=sort_key_for(series_by)):
        label = series_labels[key]
        for year in years:
            ids = buckets_by_series[(year, key)]
            rows.append({"year": year, "series": label, "trial_count": len(ids)})
            if ids:
                contributors[year_key(year, label)] = ids

    return OperatorResult(rows=rows, contributors=contributors, dropped=dropped)


# --------------------------------------------------------------------------
# 5. compare
# --------------------------------------------------------------------------


def compare(groups: dict[str, list[TrialRecord]], dim: Dimension) -> OperatorResult:
    """Two or more groups counted over the same categories.

    Long format - one row per (category, group) - rather than a column per
    group, because that is what a grouped bar chart reads and it does not
    change shape when a third group is added.

    Every group gets a row for every category, including zeros, so a category
    one side lacks entirely is visibly absent rather than silently missing.
    """
    counts: dict[tuple[str, str], set[str]] = defaultdict(set)
    labels: dict[str, str] = {}
    dropped = 0

    for group_label, records in groups.items():
        for record in records:
            pairs = dimension_values(record, dim)
            if not pairs:
                dropped += 1
                continue
            for key, label in pairs:
                counts[(key, group_label)].add(record.nct_id)
                labels.setdefault(key, label)

    categories = sorted(labels, key=lambda k: (-_total(counts, k), sort_key_for(dim)(k)))

    rows = []
    contributors = {}
    for key in categories:
        for group_label in groups:
            ids = counts[(key, group_label)]
            rows.append(
                {dim.value: labels[key], "group": group_label, "trial_count": len(ids)}
            )
            if ids:
                contributors[group_key(dim, labels[key], group_label)] = ids

    return OperatorResult(
        rows=rows, contributors=contributors, dropped=dropped, extra={"groups": list(groups)}
    )


def _total(counts: dict[tuple[str, str], set[str]], key: str) -> int:
    """Trials in one category across all groups, for ordering the axis."""
    return sum(len(ids) for (k, _), ids in counts.items() if k == key)


# --------------------------------------------------------------------------
# 6. top_k
# --------------------------------------------------------------------------

OTHER_LABEL = "Other"


def top_k(records: list[TrialRecord], dim: Dimension, k: int) -> OperatorResult:
    """The k biggest categories, with the rest folded into "Other".

    "Other" appears only when it stands for something: if there are exactly k
    categories or fewer, folding would produce an "Other" bar of zero, which
    is noise. When it does appear it keeps its contributors, so it is still a
    citable bar rather than a black hole.
    """
    counted = count_by(records, dim)
    rows = counted.rows

    if len(rows) <= k:
        return counted

    kept, folded = rows[:k], rows[k:]
    folded_ids: set[str] = set()
    for row in folded:
        folded_ids |= counted.contributors.get(datum_key(dim, row[dim.value]), set())

    contributors = {
        datum_key(dim, row[dim.value]): counted.contributors[datum_key(dim, row[dim.value])]
        for row in kept
    }
    contributors[datum_key(dim, OTHER_LABEL)] = folded_ids

    return OperatorResult(
        rows=kept + [{dim.value: OTHER_LABEL, "trial_count": len(folded_ids)}],
        contributors=contributors,
        dropped=counted.dropped,
        extra={**counted.extra, "folded_categories": len(folded)},
    )


# --------------------------------------------------------------------------
# 7. histogram
# --------------------------------------------------------------------------

MAX_BINS = 50
SKEW_THRESHOLD = 2.0

# log10(0) is undefined, so log bins start here and a zero gets its own first bin.
LOG_FLOOR = 1.0


def _freedman_diaconis_bins(values: list[float]) -> int:
    """Bin count from the Freedman-Diaconis rule, clamped to something sane.

    FD uses the interquartile range rather than the standard deviation, so a
    handful of enormous trials cannot force everything else into one bin. If
    the IQR is zero - most values identical - it gives no width at all, so we
    fall back to the square-root rule.
    """
    n = len(values)
    ordered = sorted(values)
    q1 = ordered[n // 4]
    q3 = ordered[(3 * n) // 4]
    iqr = q3 - q1
    spread = ordered[-1] - ordered[0]

    if spread == 0:
        return 1
    if iqr == 0:
        return max(1, min(MAX_BINS, int(math.sqrt(n))))

    width = 2 * iqr / (n ** (1 / 3))
    return max(1, min(MAX_BINS, math.ceil(spread / width)))


def _skew(values: list[float]) -> float:
    """Pearson's moment coefficient of skewness. 0 is symmetric."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    if variance == 0:
        return 0.0
    sd = math.sqrt(variance)
    return sum(((v - mean) / sd) ** 3 for v in values) / n


def histogram(
    records: list[TrialRecord], numeric_dim: Dimension, bins: int | str = "auto"
) -> OperatorResult:
    """Distribution of a numeric dimension.

    Enrollment is heavily right-skewed in real data: thousands of trials of
    fewer than 100 participants and a few of 50,000. On a linear axis that is
    one tall bar and a lot of empty space, so when the skew passes 2 the bins
    are spaced logarithmically instead and the shape becomes readable.
    """
    values: list[tuple[float, str]] = []
    for record in records:
        value = numeric_value(record, numeric_dim)
        if value is not None:
            values.append((value, record.nct_id))
    dropped = len(records) - len(values)

    if not values:
        return OperatorResult(dropped=dropped)

    numbers = [v for v, _ in values]
    low, high = min(numbers), max(numbers)
    skew = _skew(numbers)

    # Log bins whenever the distribution is lopsided enough to need them. The
    # test for this is `high > 0`, not `low > 0`: enrollment legitimately
    # includes zeros (withdrawn trials), and requiring a positive minimum
    # disabled log bins on exactly the data that is most skewed.
    log_scale = bins == "auto" and skew > SKEW_THRESHOLD and high > 0

    if log_scale:
        # Bin count from the log-transformed values, since that is the scale
        # the bins are on. On raw skewed values FD asks for far too many.
        logs = [math.log10(max(v, LOG_FLOOR)) for v in numbers]
        bin_count = _freedman_diaconis_bins(logs)
        edges = _log_edges(low, high, bin_count)
    else:
        bin_count = _freedman_diaconis_bins(numbers) if bins == "auto" else int(bins)
        edges = _linear_edges(low, high, bin_count)

    buckets: list[set[str]] = [set() for _ in range(len(edges) - 1)]
    for value, nct_id in values:
        buckets[_bin_index(value, edges)].add(nct_id)

    rows = []
    contributors = {}
    for i, ids in enumerate(buckets):
        start, end = edges[i], edges[i + 1]
        rows.append({"bin_start": start, "bin_end": end, "count": len(ids)})
        if ids:
            contributors[bin_key(start, end)] = ids

    return OperatorResult(
        rows=rows,
        contributors=contributors,
        dropped=dropped,
        extra={"log_scale": log_scale, "skew": round(skew, 2), "bins": len(rows)},
    )


def _linear_edges(low: float, high: float, bin_count: int) -> list[float]:
    if high == low:  # every value identical: one bin wide enough to hold it
        return [low, low + 1]
    step = (high - low) / bin_count
    return [round(low + i * step, 2) for i in range(bin_count + 1)]


def _log_edges(low: float, high: float, bin_count: int) -> list[float]:
    """Logarithmically spaced edges, with a home for zero.

    log10(0) is undefined, so the spacing starts at LOG_FLOOR. When the data
    really does contain values below that - a withdrawn trial enrolling 0 - an
    extra first edge is prepended so those trials land in bin one instead of
    being silently excluded.
    """
    floor = max(low, LOG_FLOOR)
    log_low, log_high = math.log10(floor), math.log10(high)
    if log_high <= log_low:
        return [low, high + 1]

    step = (log_high - log_low) / bin_count
    edges = [round(10 ** (log_low + i * step), 2) for i in range(bin_count + 1)]
    if low < floor:
        edges.insert(0, low)
    return edges


def _bin_index(value: float, edges: list[float]) -> int:
    """Which bin a value falls in. The last bin includes its upper edge, so
    the largest trial is counted rather than falling off the end."""
    for i in range(len(edges) - 1):
        if value < edges[i + 1]:
            return i
    return len(edges) - 2


# --------------------------------------------------------------------------
# 8. scatter
# --------------------------------------------------------------------------


def scatter(
    records: list[TrialRecord],
    x: Dimension,
    y: Dimension,
    color: Dimension | None = None,
) -> OperatorResult:
    """One point per trial.

    A point needs both coordinates, so a trial missing either is dropped
    rather than plotted at zero - a trial with no enrollment figure is not a
    trial with no participants.
    """
    rows = []
    contributors = {}
    dropped = 0

    for record in records:
        x_value = numeric_value(record, x)
        y_value = numeric_value(record, y)
        if x_value is None or y_value is None:
            dropped += 1
            continue

        row = {"nct_id": record.nct_id, x.value: x_value, y.value: y_value}
        if color is not None:
            pairs = dimension_values(record, color)
            row["color"] = pairs[0][1] if pairs else None
        rows.append(row)
        contributors[trial_key(record.nct_id)] = {record.nct_id}

    return OperatorResult(rows=rows, contributors=contributors, dropped=dropped)


# --------------------------------------------------------------------------
# 9. network
# --------------------------------------------------------------------------


def network(
    records: list[TrialRecord],
    node_dims: list[Dimension],
    min_edge_weight: int = 2,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> OperatorResult:
    """Who appears alongside whom.

    Two shapes, from the same code:

    - different dimensions, e.g. sponsor and drug: a bipartite graph, an edge
      for each sponsor that studied each drug.
    - the same dimension twice, e.g. drug and drug: co-occurrence, an edge
      between two drugs used in the same trial. Only trials with at least two
      distinct drugs can contribute, and a drug is never linked to itself.

    Node weight is the number of trials it appears in; edge weight is the
    number of trials both endpoints appear in. Edges below min_edge_weight are
    dropped first - a graph where every incidental pairing is drawn shows
    nothing - and then only the heaviest max_nodes nodes are kept, with the
    edges among them. Keeping nodes before edges is what guarantees every
    surviving edge still has both endpoints present.
    """
    left_dim, right_dim = node_dims[0], node_dims[1]
    bipartite = left_dim is not right_dim

    node_trials: dict[str, set[str]] = defaultdict(set)
    node_labels: dict[str, str] = {}
    node_types: dict[str, str] = {}
    edge_trials: dict[tuple[str, str], set[str]] = defaultdict(set)
    dropped = 0

    for record in records:
        if bipartite:
            pairs = _bipartite_pairs(record, left_dim, right_dim, node_labels, node_types)
        else:
            pairs = _cooccurrence_pairs(record, left_dim, node_labels, node_types)

        if not pairs:
            dropped += 1
            continue

        for source, target in pairs:
            node_trials[source].add(record.nct_id)
            node_trials[target].add(record.nct_id)
            edge_trials[(source, target)].add(record.nct_id)

    # Heavy edges first, then the heaviest nodes among them.
    strong_edges = {
        pair: ids for pair, ids in edge_trials.items() if len(ids) >= min_edge_weight
    }
    nodes_in_strong_edges = {n for pair in strong_edges for n in pair}
    kept = sorted(
        nodes_in_strong_edges, key=lambda n: (-len(node_trials[n]), n)
    )[:max_nodes]
    kept_set = set(kept)

    nodes = [
        {
            "id": node_id,
            "label": node_labels[node_id],
            "type": node_types[node_id],
            "weight": len(node_trials[node_id]),
        }
        for node_id in kept
    ]
    edges = [
        {"source": source, "target": target, "weight": len(ids)}
        for (source, target), ids in sorted(strong_edges.items(), key=lambda kv: -len(kv[1]))
        if source in kept_set and target in kept_set
    ]

    contributors = {node_key(n["id"]): node_trials[n["id"]] for n in nodes}
    contributors.update(
        {edge_key(e["source"], e["target"]): strong_edges[(e["source"], e["target"])] for e in edges}
    )

    return OperatorResult(
        rows=[],  # a network is not rows; see extra
        contributors=contributors,
        dropped=dropped,
        extra={
            "nodes": nodes,
            "edges": edges,
            "bipartite": bipartite,
            "min_edge_weight": min_edge_weight,
            # How much the threshold removed. A thin sample can leave every
            # pair occurring once, and the graph comes back empty; the
            # pipeline reads these to say so in the notes rather than
            # rendering a blank chart.
            "edges_before_threshold": len(edge_trials),
            "nodes_before_threshold": len(node_trials),
        },
    )


def _node_id(dim: Dimension, key: str) -> str:
    """Dimension-qualified, so a sponsor and a drug of the same name are two
    nodes rather than one."""
    return f"{dim.value}:{key}"


def _bipartite_pairs(
    record: TrialRecord,
    left_dim: Dimension,
    right_dim: Dimension,
    labels: dict[str, str],
    types: dict[str, str],
) -> list[tuple[str, str]]:
    left = dimension_values(record, left_dim)
    right = dimension_values(record, right_dim)
    if not left or not right:
        return []

    pairs = []
    for left_key, left_label in left:
        left_id = _node_id(left_dim, left_key)
        labels.setdefault(left_id, left_label)
        types.setdefault(left_id, left_dim.value)
        for right_key, right_label in right:
            right_id = _node_id(right_dim, right_key)
            labels.setdefault(right_id, right_label)
            types.setdefault(right_id, right_dim.value)
            pairs.append((left_id, right_id))
    return pairs


def _cooccurrence_pairs(
    record: TrialRecord,
    dim: Dimension,
    labels: dict[str, str],
    types: dict[str, str],
) -> list[tuple[str, str]]:
    """Every unordered pair of distinct values within one trial.

    A trial with one drug produces nothing: there is no co-occurrence to show,
    which is why `min_edge_weight` on a drug<->drug network only ever counts
    trials with two or more distinct drugs.
    """
    values = dimension_values(record, dim)
    if len(values) < 2:
        return []

    ids = []
    for key, label in values:
        node_id = _node_id(dim, key)
        labels.setdefault(node_id, label)
        types.setdefault(node_id, dim.value)
        ids.append(node_id)

    # Sorted pairs so A-B and B-A are the same edge, and i<j excludes self-loops.
    ids = sorted(set(ids))
    return [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
