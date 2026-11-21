"""The QueryPlan: the only thing the LLM is allowed to produce.

This file is the heart of the no-hallucination design. In B4 the LLM is handed
`QueryPlan.model_json_schema()` as a forced tool call, so this schema is
literally the boundary of what the model can say. It can pick an intent from a
list of nine, name dimensions from a list of twelve, and copy entity strings out
of the user's question. It cannot emit a number, a chart, a row or a citation,
because there is nowhere in this shape to put one.

Everything downstream - compiler, operators, viz builder, citations - reads the
plan and is plain deterministic Python.

Sections:

1. Vocabularies      - Intent, Dimension, Metric, DimensionKind
2. DIMENSIONS catalog - what each dimension means and where it comes from
3. QueryPlan          - the plan itself, plus its small sub-models
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Phase and Status are reused from the request schema on purpose: the request,
# the plan and the API all speak one vocabulary, so there is no place for the
# three to drift apart.
from app.api.schemas import Phase, Status

# --------------------------------------------------------------------------
# 1. Vocabularies
# --------------------------------------------------------------------------


class Intent(str, Enum):
    """What the user is asking for. Each one maps to exactly one operator and
    one viz type (see the intent -> viz table in CLAUDE.md)."""

    count = "count"  # "how many ...?"            -> stat
    list = "list"  # "which trials ...?"          -> table
    distribution = "distribution"  # "across ..." -> bar_chart
    trend = "trend"  # "per year", "since 2015"   -> time_series
    comparison = "comparison"  # "A vs B"         -> grouped_bar_chart
    geo = "geo"  # "which countries ...?"         -> bar_chart (horizontal)
    network = "network"  # "relationship between" -> network_graph
    histogram = "histogram"  # "distribution of enrollment" -> histogram
    correlation = "correlation"  # "X against Y"  -> scatter_plot


class Dimension(str, Enum):
    """Every field a question may group, filter, bin or plot by.

    This list is closed. A dimension that is not here cannot be requested,
    which is what stops the LLM inventing a field the data does not have.
    """

    phase = "phase"
    status = "status"
    start_year = "start_year"
    sponsor = "sponsor"
    sponsor_class = "sponsor_class"
    country = "country"
    intervention_type = "intervention_type"
    drug = "drug"
    condition = "condition"
    study_type = "study_type"
    enrollment = "enrollment"
    duration_months = "duration_months"


class Metric(str, Enum):
    """What is being counted on the value axis.

    Only one metric today. It is an enum rather than a hardcoded assumption so
    that adding, say, a median-enrollment metric later changes this list and
    not the shape of every plan.
    """

    trial_count = "trial_count"


class DimensionKind(str, Enum):
    """How a dimension behaves, which decides what you can do with it."""

    categorical = "categorical"  # group and count: phase, country, sponsor
    numeric = "numeric"  # bin or plot on an axis: enrollment, duration_months
    temporal = "temporal"  # a time axis: start_year


# --------------------------------------------------------------------------
# 2. DIMENSIONS catalog
# --------------------------------------------------------------------------


class DimensionInfo(BaseModel):
    """One row of the catalog: everything the rest of the app needs to know
    about a dimension, in one place."""

    kind: DimensionKind
    source_path: str  # the API path it is read from, or "derived"
    multi_valued: bool  # True if one trial can hold several values
    description: str  # one line; B4 pastes these into the LLM system prompt


# Source paths were verified against the live API - see CLAUDE.md "Field paths".
#
# multi_valued is the subtle column. A trial has one sponsor but can run in
# twelve countries, so a count by country must count that trial once per
# distinct country and will therefore sum to more than the number of trials.
# The operators read this flag rather than hardcoding which dimensions behave
# that way.
#
# `phase` is deliberately NOT multi-valued: the API gives a list like
# ["PHASE1", "PHASE2"], but the data rules fold that into the single bucket
# "Phase 1/2", so each trial lands in exactly one phase bucket.
DIMENSIONS: dict[Dimension, DimensionInfo] = {
    Dimension.phase: DimensionInfo(
        kind=DimensionKind.categorical,
        source_path="protocolSection.designModule.phases",
        multi_valued=False,
        description="Trial phase, bucketed: Phase 1, Phase 1/2, Phase 2, Phase 3, Phase 4, Not applicable.",
    ),
    Dimension.status: DimensionInfo(
        kind=DimensionKind.categorical,
        source_path="protocolSection.statusModule.overallStatus",
        multi_valued=False,
        description="Recruitment status, e.g. RECRUITING, COMPLETED, TERMINATED.",
    ),
    Dimension.start_year: DimensionInfo(
        kind=DimensionKind.temporal,
        source_path="protocolSection.statusModule.startDateStruct.date",
        multi_valued=False,
        description="Calendar year the trial started. Use this for any trend over time.",
    ),
    Dimension.sponsor: DimensionInfo(
        kind=DimensionKind.categorical,
        source_path="protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
        multi_valued=False,
        description="Name of the lead sponsor organisation, e.g. 'Merck Sharp & Dohme LLC'.",
    ),
    Dimension.sponsor_class: DimensionInfo(
        kind=DimensionKind.categorical,
        source_path="protocolSection.sponsorCollaboratorsModule.leadSponsor.class",
        multi_valued=False,
        description="Kind of sponsor: INDUSTRY, NIH, OTHER_GOV, NETWORK, FED, OTHER.",
    ),
    Dimension.country: DimensionInfo(
        kind=DimensionKind.categorical,
        source_path="protocolSection.contactsLocationsModule.locations[].country",
        multi_valued=True,
        description="Country of a trial site, as a full name like 'United States'. A trial may run in many.",
    ),
    Dimension.intervention_type: DimensionInfo(
        kind=DimensionKind.categorical,
        source_path="protocolSection.armsInterventionsModule.interventions[].type",
        multi_valued=True,
        description="Kind of intervention: DRUG, BIOLOGICAL, DEVICE, PROCEDURE, BEHAVIORAL, and others.",
    ),
    Dimension.drug: DimensionInfo(
        kind=DimensionKind.categorical,
        source_path="protocolSection.armsInterventionsModule.interventions[].name",
        multi_valued=True,
        description="Name of a drug or other intervention used in the trial, e.g. 'pembrolizumab'.",
    ),
    Dimension.condition: DimensionInfo(
        kind=DimensionKind.categorical,
        source_path="protocolSection.conditionsModule.conditions[]",
        multi_valued=True,
        description="Disease or condition studied, e.g. 'Breast Cancer'. A trial may list several.",
    ),
    Dimension.study_type: DimensionInfo(
        kind=DimensionKind.categorical,
        source_path="protocolSection.designModule.studyType",
        multi_valued=False,
        description="INTERVENTIONAL, OBSERVATIONAL or EXPANDED_ACCESS.",
    ),
    Dimension.enrollment: DimensionInfo(
        kind=DimensionKind.numeric,
        source_path="protocolSection.designModule.enrollmentInfo.count",
        multi_valued=False,
        description="Number of participants. Numeric, so usable for a histogram or a scatter axis.",
    ),
    Dimension.duration_months: DimensionInfo(
        kind=DimensionKind.numeric,
        source_path="derived",  # completion date minus start date
        multi_valued=False,
        description="Months from start date to completion date. Numeric; computed, not a raw API field.",
    ),
}


def is_numeric(dim: Dimension) -> bool:
    """True for dimensions that can go on a numeric axis or be binned."""
    return DIMENSIONS[dim].kind is DimensionKind.numeric


def is_multi_valued(dim: Dimension) -> bool:
    """True if one trial can contribute several values to this dimension."""
    return DIMENSIONS[dim].multi_valued


# --------------------------------------------------------------------------
# 3. QueryPlan
# --------------------------------------------------------------------------


class PlanFilters(BaseModel):
    """Which trials to fetch.

    These fields mirror the structured fields on VisualizationRequest one for
    one. That is what makes the override rule a plain field-by-field
    comparison in B4's validator: if the caller set `drug_name`, it replaces
    `drug` here and the swap is recorded in notes.overrides.

    The string fields are free text because the API matches them as search
    terms. They hold entity names copied from the user's question - the LLM is
    told to copy, not to normalise or expand them.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    drug: str | None = Field(default=None, max_length=200)
    condition: str | None = Field(default=None, max_length=200)
    sponsor: str | None = Field(default=None, max_length=200)
    country: str | None = Field(default=None, max_length=100)
    phase: list[Phase] | None = None
    status: list[Status] | None = None
    start_year: int | None = Field(default=None, ge=1990, le=2035)
    end_year: int | None = Field(default=None, ge=1990, le=2035)


class CompareGroup(BaseModel):
    """One side of a comparison, e.g. label "pembrolizumab" with its filters.

    Each group is fetched separately, so "A vs B" is two API searches rather
    than one search that is then split up.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(max_length=100)
    filters: PlanFilters


class NetworkSpec(BaseModel):
    """Settings for a network question."""

    model_config = ConfigDict(extra="forbid")

    # Exactly two: bipartite when they differ (sponsor <-> drug), or
    # co-occurrence within a trial when they are the same (drug <-> drug).
    node_dims: list[Dimension] = Field(min_length=2, max_length=2)
    min_edge_weight: int = Field(default=2, ge=1)


class TimeSpec(BaseModel):
    """Settings for a trend question. Year is the only granularity, because
    start dates are often only accurate to the month."""

    model_config = ConfigDict(extra="forbid")

    granularity: str = Field(default="year", pattern="^year$")
    start: int | None = Field(default=None, ge=1990, le=2035)
    end: int | None = Field(default=None, ge=1990, le=2035)


class QueryPlan(BaseModel):
    """The LLM's complete output.

    extra="forbid" matters here more than anywhere else: it means a model that
    tries to be helpful by adding a `data` or `answer` key gets a validation
    error instead of having that content quietly carried along.

    Most fields are optional because each intent uses only a few of them. B4's
    validator checks that the right ones are present for the chosen intent -
    network needs `network`, trend needs `time`, and so on. This class only
    enforces what is true for every plan.
    """

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    filters: PlanFilters = Field(default_factory=PlanFilters)
    metric: Metric = Metric.trial_count

    group_by: Dimension | None = None  # the category axis
    series_by: Dimension | None = None  # splits a trend into several lines
    compare_groups: list[CompareGroup] | None = Field(default=None, max_length=5)
    network: NetworkSpec | None = None
    time: TimeSpec | None = None
    numeric_x: Dimension | None = None  # histogram bins, or scatter x
    numeric_y: Dimension | None = None  # scatter y
    top_k: int | None = Field(default=None, ge=1, le=100)

    # Written by the LLM in its own words, and shown to the user as notes.
    # Prose only - these never carry numbers that the response relies on.
    interpretation: str = Field(max_length=500)
    assumptions: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def numeric_axes_are_numeric(self) -> "QueryPlan":
        """A numeric axis must name a numeric dimension.

        Caught here rather than in the validator because it is wrong for every
        intent, not just some: you cannot bin `country` or plot `phase` on a
        continuous axis whatever the question was.
        """
        for name in ("numeric_x", "numeric_y"):
            dim = getattr(self, name)
            if dim is not None and not is_numeric(dim):
                raise ValueError(f"{name}={dim.value} is not a numeric dimension")
        return self
