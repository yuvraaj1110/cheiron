"""Score both planners on a fixed set of questions.

Run: python scripts/eval_planners.py
Writes examples/planner_eval.json (full detail) and examples/planner_eval.md
(the summary table). Plans only - no ClinicalTrials.gov calls, so the rule
planner half runs anywhere, offline, in under a second. The LLM half needs
OPENAI_API_KEY; without it that half is reported as "not run" rather than
silently skipped.

What is scored, and why only that
---------------------------------
A plan has fields its intent never reads: the pipeline's trend branch ignores
`group_by`, its count branch reads nothing but filters. Scoring those would
punish or reward a planner for fields that cannot change the answer, so each
question is scored only on the fields its intent consumes (CONSUMED below,
which mirrors `compute()` in app/pipeline.py) plus the filters it asserts.

Each question lists the checks it expects. A check is one comparison, e.g.
intent == "trend" or node_dims == ["drug", "sponsor"]. Three numbers come out:

- intent accuracy: the intent alone, since a wrong intent means a wrong chart
- check accuracy:  every check, intent included
- exact plans:     questions where every check passed

Expectations were written from the contract in CLAUDE.md, before either
planner was run, so this measures the planners rather than describing them.
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.api.schemas import VisualizationRequest
from app.planner.llm_planner import make_client
from app.pipeline import DEFAULT_GEO_TOP_K
from app.planner.plan import Intent, QueryPlan
from app.planner.rule_planner import rule_plan
from app.planner.validator import plan_with_llm, validate_and_merge

OUT = Path("examples")

# Which plan fields each intent's operator actually reads. Mirrors compute()
# in app/pipeline.py; a test asserts every name here is a real check.
CONSUMED: dict[Intent, tuple[str, ...]] = {
    Intent.count: (),
    Intent.list: (),
    Intent.distribution: ("group_by", "top_k"),
    Intent.trend: ("time_start", "time_end", "series_by"),
    Intent.comparison: ("compare_labels", "group_by"),
    Intent.geo: ("group_by", "top_k"),
    Intent.network: ("node_dims", "min_edge_weight"),
    Intent.histogram: ("numeric_x",),
    Intent.correlation: ("numeric_x", "numeric_y"),
}

# The defaults compute() substitutes for an empty field. Read alongside
# CONSUMED: a planner that leaves group_by empty on a distribution question
# gets phase, which is what it would have got by naming phase.
GROUP_BY_DEFAULT: dict[Intent, str] = {
    Intent.distribution: "phase",
    Intent.comparison: "phase",
    Intent.geo: "country",
}
TOP_K_DEFAULT: dict[Intent, int] = {Intent.geo: DEFAULT_GEO_TOP_K}

FILTER_CHECKS = ("drug", "condition", "sponsor", "country", "phase", "status", "start_year", "end_year")


# --------------------------------------------------------------------------
# Reading one value out of a plan
# --------------------------------------------------------------------------
def enum_value(value: Any) -> Any:
    """Enums (and lists of them) as plain strings, so JSON and == both work."""
    if isinstance(value, list):
        return [enum_value(item) for item in value]
    return getattr(value, "value", value)


def read(plan: QueryPlan, check: str) -> Any:
    """The value a check compares against. One branch per check name."""
    if check == "intent":
        return plan.intent.value
    if check == "group_by":
        # compute() reads `plan.group_by or <default>`, so an empty group_by
        # is the default, not a miss.
        return enum_value(plan.group_by) or GROUP_BY_DEFAULT.get(plan.intent)
    if check in ("series_by", "numeric_x", "numeric_y"):
        return enum_value(getattr(plan, check))
    if check == "top_k":
        return plan.top_k if plan.top_k is not None else TOP_K_DEFAULT.get(plan.intent)
    if check == "node_dims":
        return sorted(enum_value(plan.network.node_dims)) if plan.network else None
    if check == "min_edge_weight":
        return plan.network.min_edge_weight if plan.network else None
    if check == "time_start":
        return plan.time.start if plan.time else None
    if check == "time_end":
        return plan.time.end if plan.time else None
    if check == "compare_labels":
        groups = plan.compare_groups or []
        return sorted(group.label.casefold() for group in groups)
    if check in FILTER_CHECKS:
        value = enum_value(getattr(plan.filters, check))
        if isinstance(value, str):
            return value.casefold()
        if isinstance(value, list):
            return sorted(value)
        return value
    raise KeyError(f"unknown check: {check}")


def matches(expected: Any, actual: Any) -> bool:
    """Equality, with free-text entity names compared case-insensitively."""
    if isinstance(expected, str):
        return isinstance(actual, str) and expected.casefold() == actual.casefold()
    if isinstance(expected, list):
        return sorted(enum_value(expected)) == actual
    return expected == actual


# --------------------------------------------------------------------------
# The question set
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Question:
    id: str
    body: dict
    expected: dict  # check name -> expected value; "intent" is always present
    note: str = ""

    def checks(self) -> dict:
        """The expectations that this intent's operator can act on.

        Keeps a planner from being marked wrong on a field the chart never
        reads. A question may only expect a field its own intent consumes,
        which a test enforces.
        """
        intent = Intent(self.expected["intent"])
        keep = {"intent", *CONSUMED[intent], *FILTER_CHECKS}
        return {name: value for name, value in self.expected.items() if name in keep}


QUESTIONS: list[Question] = [
    Question(
        "q01_count_plain",
        {"query": "How many recruiting Alzheimer's disease trials are there?"},
        {"intent": "count", "condition": "alzheimer's disease", "status": ["RECRUITING"]},
    ),
    Question(
        "q02_count_phrased_as_total",
        {"query": "What is the total number of completed trials sponsored by Pfizer?"},
        {"intent": "count", "sponsor": "pfizer", "status": ["COMPLETED"]},
        note="'total number of' instead of 'how many'.",
    ),
    Question(
        "q03_list",
        {"query": "List phase 3 trials for pembrolizumab"},
        {"intent": "list", "drug": "pembrolizumab", "phase": ["PHASE3"]},
    ),
    Question(
        "q04_distribution_phase",
        {"query": "How are breast cancer trials distributed across phases?"},
        {"intent": "distribution", "group_by": "phase", "condition": "breast cancer"},
    ),
    Question(
        "q05_distribution_sponsor",
        {"query": "Which sponsors run the most obesity trials?"},
        {"intent": "distribution", "group_by": "sponsor", "condition": "obesity"},
        note="Needs sponsor, not the default phase grouping.",
    ),
    Question(
        "q06_distribution_sponsor_class",
        {"query": "What share of melanoma trials are run by industry versus government?"},
        {"intent": "distribution", "group_by": "sponsor_class", "condition": "melanoma"},
        note="'industry versus government' is the sponsor_class dimension, not a comparison of two drugs.",
    ),
    Question(
        "q07_trend_with_window",
        {"query": "How many trials for pembrolizumab started each year since 2015?"},
        {"intent": "trend", "time_start": 2015, "drug": "pembrolizumab"},
    ),
    Question(
        "q08_trend_no_window",
        {"query": "Has interest in gene therapy trials grown over time?"},
        {"intent": "trend", "time_start": None, "condition": "gene therapy"},
        note="A trend with no window given; time.start must stay empty rather than be invented.",
    ),
    Question(
        "q09_comparison_explicit",
        {"query": "Compare phases for trials involving pembrolizumab vs nivolumab"},
        {"intent": "comparison", "group_by": "phase", "compare_labels": ["nivolumab", "pembrolizumab"]},
    ),
    Question(
        "q10_comparison_structured",
        {"query": "How do these two drugs compare by trial phase?", "drug_name": "semaglutide", "compare_with": ["tirzepatide"]},
        {"intent": "comparison", "group_by": "phase", "compare_labels": ["semaglutide", "tirzepatide"]},
        note="The entities arrive as structured fields, not in the question text.",
    ),
    Question(
        "q11_geo",
        {"query": "Which countries have the most recruiting trials for diabetes?"},
        {"intent": "geo", "group_by": "country", "condition": "diabetes", "status": ["RECRUITING"]},
    ),
    Question(
        "q12_network_bipartite",
        {"query": "Show a network of sponsors and drugs for non-small cell lung cancer trials"},
        {"intent": "network", "node_dims": ["drug", "sponsor"], "condition": "non-small cell lung cancer"},
    ),
    Question(
        "q13_network_cooccurrence",
        {"query": "Which drugs frequently co-occur with pembrolizumab in combination studies?", "drug_name": "pembrolizumab"},
        {"intent": "network", "node_dims": ["drug", "drug"], "drug": "pembrolizumab"},
        note="Same dimension twice: a co-occurrence graph, not a bipartite one.",
    ),
    Question(
        "q14_histogram",
        {"query": "What does the distribution of enrollment look like for Alzheimer's trials?"},
        {"intent": "histogram", "numeric_x": "enrollment", "condition": "alzheimer's"},
        note="'distribution of enrollment' is a histogram over a numeric field, not a bar chart.",
    ),
    Question(
        "q15_correlation",
        {"query": "Is enrollment related to study duration in obesity trials?"},
        {"intent": "correlation", "numeric_x": "enrollment", "numeric_y": "duration_months", "condition": "obesity"},
    ),
]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
@dataclass
class Score:
    planner: str
    questions: list[dict] = field(default_factory=list)
    error: str | None = None

    def totals(self) -> dict:
        if self.error is not None:
            return {"status": "not run", "reason": self.error}
        passed = sum(q["checks_passed"] for q in self.questions)
        total = sum(q["checks_total"] for q in self.questions)
        intents = sum(1 for q in self.questions if q["intent_correct"])
        exact = sum(1 for q in self.questions if q["exact"])
        n = len(self.questions)
        return {
            "status": "run",
            "questions": n,
            "intent_correct": intents,
            "intent_accuracy": round(intents / n, 3) if n else 0.0,
            "checks_passed": passed,
            "checks_total": total,
            "check_accuracy": round(passed / total, 3) if total else 0.0,
            "exact_plans": exact,
            "exact_plan_rate": round(exact / n, 3) if n else 0.0,
            "plan_errors": sum(1 for q in self.questions if q.get("plan_error")),
        }


def score_question(question: Question, plan: QueryPlan | None, error: str | None) -> dict:
    """One question's result. A planner that could not produce a plan at all
    fails every check rather than being left out of the denominator."""
    checks = question.checks()
    if plan is None:
        return {
            "id": question.id,
            "query": question.body["query"],
            "plan_error": error,
            "intent_correct": False,
            "checks_passed": 0,
            "checks_total": len(checks),
            "exact": False,
            "failures": [{"check": name, "expected": value, "actual": None} for name, value in checks.items()],
        }

    failures = []
    for name, expected in checks.items():
        actual = read(plan, name)
        if not matches(expected, actual):
            failures.append({"check": name, "expected": enum_value(expected), "actual": actual})

    return {
        "id": question.id,
        "query": question.body["query"],
        "intent_correct": read(plan, "intent") == checks["intent"],
        "checks_passed": len(checks) - len(failures),
        "checks_total": len(checks),
        "exact": not failures,
        "failures": failures,
        "interpretation": plan.interpretation,
    }


def run_rule_planner() -> Score:
    score = Score("rule")
    for question in QUESTIONS:
        request = VisualizationRequest(**question.body)
        try:
            plan, _, _ = validate_and_merge(rule_plan(request), request, source="rule planner")
            score.questions.append(score_question(question, plan, None))
        except Exception as error:  # a plan the validator rejects outright
            score.questions.append(score_question(question, None, f"{type(error).__name__}: {error}"))
    return score


async def run_llm_planner() -> Score:
    client = make_client()
    if client is None:
        return Score("llm", error="OPENAI_API_KEY not set")
    score = Score("llm")
    for question in QUESTIONS:
        request = VisualizationRequest(**question.body)
        try:
            plan, _, _ = await plan_with_llm(request, client)
            score.questions.append(score_question(question, plan, None))
        except Exception as error:
            score.questions.append(score_question(question, None, f"{type(error).__name__}: {error}"))
    return score


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def summary_table(scores: list[Score]) -> list[str]:
    rows = [
        "| Planner | Questions | Intent correct | Checks passed | Exact plans |",
        "| --- | --- | --- | --- | --- |",
    ]
    for score in scores:
        t = score.totals()
        if t["status"] != "run":
            rows.append(f"| `{score.planner}` | — | not run ({t['reason']}) | — | — |")
            continue
        rows.append(
            f"| `{score.planner}` | {t['questions']} | "
            f"{t['intent_correct']}/{t['questions']} ({t['intent_accuracy']:.0%}) | "
            f"{t['checks_passed']}/{t['checks_total']} ({t['check_accuracy']:.0%}) | "
            f"{t['exact_plans']}/{t['questions']} ({t['exact_plan_rate']:.0%}) |"
        )
    return rows


def per_question_table(scores: list[Score]) -> list[str]:
    ran = [s for s in scores if s.error is None]
    header = "| # | Question | " + " | ".join(f"`{s.planner}`" for s in ran) + " |"
    rows = [header, "| --- | --- | " + " | ".join("---" for _ in ran) + " |"]
    for index, question in enumerate(QUESTIONS, start=1):
        cells = []
        for score in ran:
            result = next(q for q in score.questions if q["id"] == question.id)
            mark = "pass" if result["exact"] else f"{result['checks_passed']}/{result['checks_total']}"
            if result["failures"]:
                mark += " — " + ", ".join(f["check"] for f in result["failures"])
            cells.append(mark)
        rows.append(f"| {index:02d} | {question.body['query']} | " + " | ".join(cells) + " |")
    return rows


def report(scores: list[Score]) -> str:
    lines = [
        "# Planner evaluation",
        "",
        f"`python scripts/eval_planners.py` over {len(QUESTIONS)} fixed questions, covering all nine",
        "intents. Plans only: no ClinicalTrials.gov calls, so this is a measure of the",
        "planners and nothing downstream of them.",
        "",
        "Each question is scored only on the plan fields its intent's operator reads",
        "(plus the filters it asserts), because a field the chart never reads cannot",
        "change the answer. Expectations come from the contract in `CLAUDE.md`.",
        "",
        "## Totals",
        "",
        *summary_table(scores),
        "",
        "- **Intent correct** — the right intent, and so the right chart type.",
        "- **Checks passed** — every expectation, intent and filters included.",
        "- **Exact plans** — questions where nothing at all was missed.",
        "",
        "## Per question",
        "",
        *per_question_table(scores),
        "",
        "A cell shows `pass`, or the score with the checks that failed. Full detail,",
        "including each planner's own interpretation line, is in",
        "[`planner_eval.json`](planner_eval.json).",
        "",
    ]
    return "\n".join(lines)


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    scores = [run_rule_planner(), await run_llm_planner()]

    payload = {
        "questions": [
            {"id": q.id, "request": q.body, "expected": q.checks(), "note": q.note} for q in QUESTIONS
        ],
        "results": [
            {"planner": s.planner, "totals": s.totals(), "questions": s.questions} for s in scores
        ],
    }
    (OUT / "planner_eval.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (OUT / "planner_eval.md").write_text(report(scores))

    for score in scores:
        print(score.planner, score.totals())


if __name__ == "__main__":
    asyncio.run(main())
