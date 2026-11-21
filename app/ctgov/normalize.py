"""Turn one raw API study into a TrialRecord.

This is where every messy thing about the real data is dealt with, once, so
that nothing downstream has to know about it. The operators in B3 see tidy
records with a single phase label, a real date and a deduplicated country
list, and can stay short and pure.

Two rules shape the whole file:

- Nothing is ever assumed to be present. Every module and field in a
  ClinicalTrials.gov record is optional in practice, and the fixtures contain
  studies missing phases, interventions, locations, enrollment and even the
  start date. A missing value becomes None and is reported, never guessed.

- Every value keeps a pointer back to where it came from. `TrialRecord.sources`
  records the API path and the raw value for each field, which is what lets the
  citation attacher in B3 quote the record verbatim instead of writing prose.
"""

import re
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Phase bucketing
# --------------------------------------------------------------------------
# The API gives a list, e.g. ["PHASE1", "PHASE2"]. We fold it into exactly one
# bucket so each trial is counted once and phase counts sum to the trial total.
# All eight buckets occur in the fixtures.
PHASE_LABELS: dict[tuple[str, ...], str] = {
    ("EARLY_PHASE1",): "Early Phase 1",
    ("PHASE1",): "Phase 1",
    ("PHASE1", "PHASE2"): "Phase 1/2",
    ("PHASE2",): "Phase 2",
    ("PHASE2", "PHASE3"): "Phase 2/3",
    ("PHASE3",): "Phase 3",
    ("PHASE4",): "Phase 4",
    ("NA",): "Not applicable",
}

# Chart order. Not alphabetical: "Phase 1/2" belongs between 1 and 2, and
# "Not applicable" belongs at the end rather than under N.
PHASE_ORDER: list[str] = list(PHASE_LABELS.values())

# A study with no phases key at all - every observational study - means the
# same thing to a reader as an explicit "NA".
NO_PHASE_LABEL = "Not applicable"


def phase_label(phases: list[str] | None) -> str:
    """Fold a raw phase list into one bucket label.

    Unknown combinations are built from their parts rather than hidden in an
    "Other" bucket, so a real PHASE1|PHASE3 trial reads as "Phase 1/3" and
    stays visible.
    """
    if not phases:
        return NO_PHASE_LABEL

    key = tuple(sorted(phases))
    if key in PHASE_LABELS:
        return PHASE_LABELS[key]

    numbers = [p.replace("EARLY_PHASE", "").replace("PHASE", "") for p in sorted(phases)]
    return "Phase " + "/".join(n for n in numbers if n)


def phase_sort_key(label: str) -> tuple[int, str]:
    """Sort known buckets in chart order; anything unexpected goes last."""
    if label in PHASE_ORDER:
        return (PHASE_ORDER.index(label), "")
    return (len(PHASE_ORDER), label)


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
# Start dates are often only accurate to the month: up to 39% of a result in
# the fixtures is "YYYY-MM", and a few are "YYYY".
DAYS_PER_MONTH = 30.4375  # 365.25 / 12


def parse_partial_date(value: str | None) -> date | None:
    """Parse "YYYY", "YYYY-MM" or "YYYY-MM-DD" into a date.

    Missing parts default to the first, so "2019-09" becomes 2019-09-01. That
    is a documented assumption rather than a hidden one: it is accurate enough
    for a yearly trend, which is the only granularity we offer.
    """
    if not value:
        return None
    parts = value.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return date(year, month, day)
    except (ValueError, IndexError):
        return None


def months_between(start: date | None, end: date | None) -> float | None:
    """Trial duration in months, or None if either end is missing."""
    if start is None or end is None or end < start:
        return None
    return round((end - start).days / DAYS_PER_MONTH, 1)


# --------------------------------------------------------------------------
# Drug names
# --------------------------------------------------------------------------
# Intervention names in the wild: "Pembrolizumab", "Pembrolizumab 200 mg",
# "pembrolizumab (MK-3475)", "Pembrolizumab IV infusion". All one drug. The
# key is what we group by; the original string stays as the display label.

_PARENTHESES = re.compile(r"\([^)]*\)")
# A dose, with its optional per-weight or per-period tail: "200 mg",
# "3 mg/kg", "50 mg/m2", "10 mg/day". Without the tail, "3 mg/kg" would strip
# to "kg" and leave that stuck to the drug name.
_DOSE = re.compile(
    r"\b\d+(\.\d+)?\s*(mg|mcg|ug|g|ml|l|iu|units?|%)"
    r"(\s*/\s*(kg|m2|m\^2|day|dose|week|hr|hour))?\b",
    re.I,
)
_ROUTE_WORDS = re.compile(
    r"\b(iv|im|sc|po|oral(ly)?|intravenous(ly)?|intramuscular|subcutaneous(ly)?|"
    r"tablets?|capsules?|injections?|infusions?|solution|suspension|placebo|"
    r"q\d+w|qd|bid|tid|daily|weekly)\b",
    re.I,
)
_PUNCTUATION = re.compile(r"[^\w\s-]")
_WHITESPACE = re.compile(r"\s+")


def drug_key(name: str) -> str:
    """Group key for an intervention name: lowercase, no dose or route text.

    Falls back to the cleaned original if stripping would leave nothing, so a
    name made only of dose text still groups with itself rather than with
    every other such name under "".
    """
    cleaned = _PARENTHESES.sub(" ", name.lower())
    cleaned = _DOSE.sub(" ", cleaned)
    cleaned = _ROUTE_WORDS.sub(" ", cleaned)
    cleaned = _PUNCTUATION.sub(" ", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip(" -")
    return cleaned or _WHITESPACE.sub(" ", name.lower()).strip()


# --------------------------------------------------------------------------
# Reading nested values safely
# --------------------------------------------------------------------------


def dig(raw: dict, path: str) -> Any:
    """Read a dotted API path, returning None if anything along it is absent.

    The paths are written exactly as they appear in CLAUDE.md's field list, so
    a citation can quote the same string it was read from.
    """
    value: Any = raw
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
        if value is None:
            return None
    return value


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


class Intervention(BaseModel):
    type: str  # DRUG, BIOLOGICAL, DEVICE, ...
    name: str  # display label, exactly as the API wrote it
    key: str  # grouping key from drug_key()


class FieldSource(BaseModel):
    """Where one field came from, so a citation can quote it."""

    path: str  # the API path, e.g. "protocolSection.designModule.phases"
    raw_value: Any  # the value at that path, untouched


class TrialRecord(BaseModel):
    """One trial, tidied. Everything the operators need and nothing else."""

    nct_id: str
    title: str
    status: str | None = None
    study_type: str | None = None

    phases: list[str] = Field(default_factory=list)  # raw, e.g. ["PHASE1","PHASE2"]
    phase_label: str = NO_PHASE_LABEL  # one of the eight buckets

    start_date: date | None = None
    start_year: int | None = None
    completion_date: date | None = None
    duration_months: float | None = None

    enrollment: int | None = None
    sponsor: str | None = None
    sponsor_class: str | None = None
    conditions: list[str] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)  # sorted, distinct

    # field name -> where it came from. Used only by the citation attacher.
    sources: dict[str, FieldSource] = Field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"


# Path for every field we cite. One table, so the record and its citations can
# never disagree about where a value came from.
SOURCE_PATHS: dict[str, str] = {
    "nct_id": "protocolSection.identificationModule.nctId",
    "title": "protocolSection.identificationModule.briefTitle",
    "status": "protocolSection.statusModule.overallStatus",
    "study_type": "protocolSection.designModule.studyType",
    "phase": "protocolSection.designModule.phases",
    "start_date": "protocolSection.statusModule.startDateStruct.date",
    "completion_date": "protocolSection.statusModule.completionDateStruct.date",
    "enrollment": "protocolSection.designModule.enrollmentInfo.count",
    "sponsor": "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
    "sponsor_class": "protocolSection.sponsorCollaboratorsModule.leadSponsor.class",
    "condition": "protocolSection.conditionsModule.conditions",
    "intervention": "protocolSection.armsInterventionsModule.interventions",
    "country": "protocolSection.contactsLocationsModule.locations",
}


def normalize_study(raw: dict) -> TrialRecord:
    """Pure function: one raw API study in, one TrialRecord out.

    Reads every field through `dig`, so a study missing whole modules
    normalizes without error instead of raising.
    """
    phases = dig(raw, SOURCE_PATHS["phase"]) or []
    start_date = parse_partial_date(dig(raw, SOURCE_PATHS["start_date"]))
    completion_date = parse_partial_date(dig(raw, SOURCE_PATHS["completion_date"]))

    raw_interventions = dig(raw, SOURCE_PATHS["intervention"]) or []
    interventions = [
        Intervention(
            type=item.get("type", "UNKNOWN"),
            name=item["name"],
            key=drug_key(item["name"]),
        )
        for item in raw_interventions
        if item.get("name")
    ]

    # A trial runs at many sites in the same country; it should count once per
    # country, so collapse the site list to distinct country names.
    locations = dig(raw, SOURCE_PATHS["country"]) or []
    countries = sorted({loc["country"] for loc in locations if loc.get("country")})

    record = TrialRecord(
        nct_id=dig(raw, SOURCE_PATHS["nct_id"]) or "",
        title=dig(raw, SOURCE_PATHS["title"]) or "",
        status=dig(raw, SOURCE_PATHS["status"]),
        study_type=dig(raw, SOURCE_PATHS["study_type"]),
        phases=phases,
        phase_label=phase_label(phases),
        start_date=start_date,
        start_year=start_date.year if start_date else None,
        completion_date=completion_date,
        duration_months=months_between(start_date, completion_date),
        enrollment=dig(raw, SOURCE_PATHS["enrollment"]),
        sponsor=dig(raw, SOURCE_PATHS["sponsor"]),
        sponsor_class=dig(raw, SOURCE_PATHS["sponsor_class"]),
        conditions=dig(raw, SOURCE_PATHS["condition"]) or [],
        interventions=interventions,
        countries=countries,
    )

    # Keep the raw value behind every field, at the path it was read from.
    # Fields that were absent are left out rather than stored as None, so the
    # citation attacher can tell "no value" from "value happened to be null".
    record.sources = {
        name: FieldSource(path=path, raw_value=dig(raw, path))
        for name, path in SOURCE_PATHS.items()
        if dig(raw, path) is not None
    }
    return record


def normalize_studies(raw_studies: list[dict]) -> list[TrialRecord]:
    """Normalize a page of studies, skipping any without an NCT ID.

    A record with no ID could not be cited or linked, so it has no place in an
    answer. In practice the API always supplies one.
    """
    records = [normalize_study(raw) for raw in raw_studies]
    return [record for record in records if record.nct_id]
