"""The public contract: what POST /v1/visualize accepts and what it returns.

Read top to bottom. Four sections:

1. Vocabularies  - the fixed string sets (phase, status, viz type, channel type)
2. Request       - what the caller sends
3. VizSpec       - the chart description we return, one class per viz type
4. Response      - VizSpec + meta + notes

Nothing here computes anything. These are shapes plus their validation rules,
so an invalid request or a malformed chart fails here and never reaches the
pipeline.
"""

from enum import Enum
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------
# 1. Vocabularies
# --------------------------------------------------------------------------
# Phase and Status use the API's own spellings, so a request value can be
# passed straight through to the API with no translation step - one less place
# a value could be mangled.


class Phase(str, Enum):
    """designModule.phases values. All six seen in the fixtures."""

    EARLY_PHASE1 = "EARLY_PHASE1"
    PHASE1 = "PHASE1"
    PHASE2 = "PHASE2"
    PHASE3 = "PHASE3"
    PHASE4 = "PHASE4"
    NA = "NA"  # trial has no phase (observational); see CLAUDE.md data rules


class Status(str, Enum):
    """statusModule.overallStatus values.

    The full documented set, not just the eight the fixtures happen to contain,
    so a caller can filter on any real status.
    """

    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    RECRUITING = "RECRUITING"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    COMPLETED = "COMPLETED"
    WITHDRAWN = "WITHDRAWN"
    AVAILABLE = "AVAILABLE"
    NO_LONGER_AVAILABLE = "NO_LONGER_AVAILABLE"
    TEMPORARILY_NOT_AVAILABLE = "TEMPORARILY_NOT_AVAILABLE"
    APPROVED_FOR_MARKETING = "APPROVED_FOR_MARKETING"
    WITHHELD = "WITHHELD"
    UNKNOWN = "UNKNOWN"


class VizType(str, Enum):
    """The eight charts we can return. One class per value in section 3."""

    stat = "stat"
    table = "table"
    bar_chart = "bar_chart"
    grouped_bar_chart = "grouped_bar_chart"
    time_series = "time_series"
    histogram = "histogram"
    scatter_plot = "scatter_plot"
    network_graph = "network_graph"


class ChannelType(str, Enum):
    """How a frontend should treat a field: the Vega-Lite vocabulary."""

    nominal = "nominal"  # unordered categories, e.g. country
    ordinal = "ordinal"  # ordered categories, e.g. phase
    quantitative = "quantitative"  # numbers, e.g. trial_count
    temporal = "temporal"  # dates and years


# --------------------------------------------------------------------------
# 2. Request
# --------------------------------------------------------------------------


def check_year_order(start: int | None, end: int | None, start_name: str, end_name: str) -> None:
    """Raise if a year window runs backwards.

    Defined once and reused by the request here and by the plan's filters and
    time window in app/planner/plan.py, so a backwards window is impossible
    whether it came from the caller or from the LLM.
    """
    if start is not None and end is not None and start > end:
        raise ValueError(f"{start_name} must be <= {end_name}")


class VisualizationRequest(BaseModel):
    """One question, plus optional structured filters.

    `query` is the only required field. The structured fields are an escape
    hatch: whatever the planner reads out of the question, an explicit field
    here wins, and the override is recorded in notes.overrides.

    extra="forbid" so a typo like `drug` instead of `drug_name` is a loud 422
    rather than a filter that silently does nothing.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=3, max_length=500, description="The question, in plain English.")

    # Entity filters. Free text because the API matches them as search terms.
    drug_name: str | None = Field(default=None, max_length=200)
    condition: str | None = Field(default=None, max_length=200)
    sponsor: str | None = Field(default=None, max_length=200)
    country: str | None = Field(default=None, max_length=100)

    # Enum filters, so an unknown phase or status is rejected before any API call.
    phase: list[Phase] | None = None
    status: list[Status] | None = None

    # Year window on the trial start date.
    start_year: int | None = Field(default=None, ge=1990, le=2035)
    end_year: int | None = Field(default=None, ge=1990, le=2035)

    # For comparison questions: the other things to compare the query against.
    compare_with: list[str] | None = Field(default=None, max_length=4)

    # A chart the caller would prefer. Honored only if it fits the result shape.
    viz_hint: VizType | None = None

    # None = the server default (settings.max_studies, 10,000).
    max_studies: int | None = Field(default=None, ge=1, le=10000)

    @model_validator(mode="after")
    def years_in_order(self) -> "VisualizationRequest":
        check_year_order(self.start_year, self.end_year, "start_year", "end_year")
        return self


# --------------------------------------------------------------------------
# 3. VizSpec
# --------------------------------------------------------------------------


class Citation(BaseModel):
    """Proof that one datum came from one trial.

    `excerpt` is copied verbatim out of the API record - never written by the
    LLM, never reworded. A test asserts every excerpt appears literally in the
    raw study JSON, which is the project's hallucination guard.
    """

    nct_id: str
    title: str  # the trial's briefTitle, also verbatim
    field: str  # the API path the value came from
    value: str  # the datum this citation supports, e.g. "Phase 1/2"
    excerpt: str  # verbatim text from the API record at `field`
    url: str  # https://clinicaltrials.gov/study/<nct_id>


class Channel(BaseModel):
    """One encoding channel: which field to read and how to draw it."""

    field: str  # a key in the data rows, e.g. "phase_label"
    type: ChannelType
    title: str  # axis or legend label
    sort: str | list[str] | None = None  # "-trial_count", or an explicit order
    unit: str | None = None  # "trials", "participants", "months"
    format: str | None = None  # number/date format hint


class Encoding(BaseModel):
    """Which channels this chart uses. Every channel is optional here; each
    viz type below declares the ones it actually requires.

    There is no `nodes` or `edges` channel. A channel names a field to read
    out of the data, and for a network the nodes and edges are the data, so
    such a channel would have nothing to point at. The network graph uses
    `color` and `size` instead, which do name real node fields.
    """

    x: Channel | None = None
    y: Channel | None = None
    series: Channel | None = None
    color: Channel | None = None
    size: Channel | None = None
    label: Channel | None = None


class NetworkNode(BaseModel):
    id: str
    label: str
    type: str  # which dimension it came from, e.g. "sponsor" or "drug"
    weight: int  # trials this node appears in
    citation_count: int = 0
    citations: list[Citation] = Field(default_factory=list)


class NetworkEdge(BaseModel):
    source: str  # a NetworkNode.id
    target: str
    weight: int  # trials both endpoints appear in
    citation_count: int = 0
    citations: list[Citation] = Field(default_factory=list)


class NetworkData(BaseModel):
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]

    @model_validator(mode="after")
    def edges_connect_known_nodes(self) -> "NetworkData":
        """Every edge endpoint must be a node in this graph.

        Cheap to check and it makes an unrenderable graph impossible: a
        frontend can draw the edges without first checking the ids resolve.
        """
        ids = {node.id for node in self.nodes}
        for edge in self.edges:
            unknown = {edge.source, edge.target} - ids
            if unknown:
                raise ValueError(f"edge references unknown node(s): {sorted(unknown)}")
        return self


class _SpecBase(BaseModel):
    """Shared shape for every viz type.

    Subclasses differ in exactly two ways: their `type` literal, and which
    encoding channels they list in `required_channels`. One validator below
    enforces that list for all of them, so adding a ninth chart is four lines.
    """

    model_config = ConfigDict(extra="forbid")

    required_channels: ClassVar[tuple[str, ...]] = ()

    title: str
    encoding: Encoding
    options: dict = Field(default_factory=dict)  # e.g. {"horizontal": True}

    @model_validator(mode="after")
    def has_required_channels(self) -> "_SpecBase":
        missing = [c for c in self.required_channels if getattr(self.encoding, c) is None]
        if missing:
            raise ValueError(
                f"{type(self).__name__} requires encoding channels: {', '.join(missing)}"
            )
        return self


# Each `data` row is a plain dict so the citation attacher can add
# `citation_count` and `citations` keys to it without a schema change.
# The row keys are whatever the operator produced, named by the channels above.


class StatSpec(_SpecBase):
    """A single number. data is one row; `y` names the number to show."""

    type: Literal[VizType.stat] = VizType.stat
    required_channels: ClassVar[tuple[str, ...]] = ("y",)
    data: list[dict]


class TableSpec(_SpecBase):
    """Rows of trials.

    Columns are their own typed field rather than an entry in `options`,
    because a table has many columns and `encoding` only models a handful of
    channels. Each column is a Channel, so a frontend reads a table's columns
    exactly the way it reads a chart's axes - same shape, same field names.
    """

    type: Literal[VizType.table] = VizType.table
    columns: list[Channel] = Field(min_length=1)
    data: list[dict]


class BarChartSpec(_SpecBase):
    """One bar per category. Also used for geo, with options["horizontal"]."""

    type: Literal[VizType.bar_chart] = VizType.bar_chart
    required_channels: ClassVar[tuple[str, ...]] = ("x", "y")
    data: list[dict]


class GroupedBarChartSpec(_SpecBase):
    """Bars split by group, so `series` is required on top of x and y."""

    type: Literal[VizType.grouped_bar_chart] = VizType.grouped_bar_chart
    required_channels: ClassVar[tuple[str, ...]] = ("x", "y", "series")
    data: list[dict]


class TimeSeriesSpec(_SpecBase):
    type: Literal[VizType.time_series] = VizType.time_series
    required_channels: ClassVar[tuple[str, ...]] = ("x", "y")
    data: list[dict]


class HistogramSpec(_SpecBase):
    """Binned numbers. Rows are {bin_start, bin_end, count}."""

    type: Literal[VizType.histogram] = VizType.histogram
    required_channels: ClassVar[tuple[str, ...]] = ("x", "y")
    data: list[dict]


class ScatterPlotSpec(_SpecBase):
    """One point per trial."""

    type: Literal[VizType.scatter_plot] = VizType.scatter_plot
    required_channels: ClassVar[tuple[str, ...]] = ("x", "y")
    data: list[dict]


class NetworkGraphSpec(_SpecBase):
    """The one type whose data is not a list of rows.

    `color` names the node field that splits the two kinds of node apart
    (`type`, e.g. sponsor vs drug) and `size` names the node field that sets
    how big to draw it (`weight`). Both point at real NetworkNode fields.
    """

    type: Literal[VizType.network_graph] = VizType.network_graph
    required_channels: ClassVar[tuple[str, ...]] = ("color", "size")
    data: NetworkData


# Discriminated on `type`, so Pydantic picks the right class from the JSON and
# the generated schema at GET /v1/schema spells out all eight shapes.
VizSpec = Annotated[
    StatSpec
    | TableSpec
    | BarChartSpec
    | GroupedBarChartSpec
    | TimeSeriesSpec
    | HistogramSpec
    | ScatterPlotSpec
    | NetworkGraphSpec,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------
# 4. Response
# --------------------------------------------------------------------------


class Meta(BaseModel):
    """How the answer was produced - the audit trail."""

    filters_applied: dict  # the filters that actually reached the API
    plan: dict  # the QueryPlan, as filled in
    total_matched: int  # trials matching the search
    studies_analyzed: int  # trials we actually pulled and counted
    truncated: bool  # True if total_matched > studies_analyzed
    chart_recommended: bool  # False for stat and table
    viz_rationale: str  # why this chart
    source: str = "clinicaltrials.gov/api/v2"
    timings_ms: dict[str, float] = Field(default_factory=dict)  # per stage

    @model_validator(mode="after")
    def truncated_matches_the_counts(self) -> "Meta":
        """`truncated` is not an independent claim - it follows from the two
        counts above. Enforced so the flag can never disagree with them and
        quietly understate how much of the result was left out."""
        expected = self.total_matched > self.studies_analyzed
        if self.truncated != expected:
            raise ValueError(
                f"truncated={self.truncated} but total_matched={self.total_matched} "
                f"and studies_analyzed={self.studies_analyzed} imply {expected}"
            )
        return self


class Notes(BaseModel):
    """What the caller should know to read the chart honestly."""

    interpretation: str  # how the question was understood
    assumptions: list[str] = Field(default_factory=list)  # e.g. trials with no start date
    overrides: list[str] = Field(default_factory=list)  # request field beat the planner


class VisualizationResponse(BaseModel):
    visualization: VizSpec
    meta: Meta
    notes: Notes
