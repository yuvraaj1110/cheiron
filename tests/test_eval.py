"""The planner eval's question set has to be well formed, or its score is
meaningless. These run the rule planner only, so they stay offline."""

import pytest

from app.api.schemas import VisualizationRequest
from app.planner.plan import Intent
from app.planner.rule_planner import rule_plan
from app.planner.validator import validate_and_merge
from scripts.eval_planners import (
    CONSUMED,
    FILTER_CHECKS,
    QUESTIONS,
    GROUP_BY_DEFAULT,
    read,
    run_rule_planner,
    score_question,
)


def test_fifteen_questions_with_unique_ids() -> None:
    assert len(QUESTIONS) == 15
    assert len({q.id for q in QUESTIONS}) == 15


def test_every_intent_is_covered() -> None:
    covered = {q.expected["intent"] for q in QUESTIONS}
    assert covered == {intent.value for intent in Intent}


def test_consumed_covers_every_intent() -> None:
    assert set(CONSUMED) == set(Intent)


def test_group_by_defaults_name_real_dimensions() -> None:
    """Each default has to be what compute() actually substitutes."""
    assert GROUP_BY_DEFAULT[Intent.distribution] == "phase"
    assert GROUP_BY_DEFAULT[Intent.comparison] == "phase"
    assert GROUP_BY_DEFAULT[Intent.geo] == "country"


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q.id)
def test_question_only_expects_fields_its_intent_reads(question) -> None:
    """A question may not expect a field the chart never looks at - scoring
    one would measure nothing."""
    intent = Intent(question.expected["intent"])
    allowed = {"intent", *CONSUMED[intent], *FILTER_CHECKS}
    assert set(question.expected) <= allowed
    assert set(question.checks()) == set(question.expected)


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q.id)
def test_every_check_is_readable_off_a_plan(question) -> None:
    """Every check name has a branch in read(), so no expectation is silently
    unscoreable, and the request itself is valid."""
    request = VisualizationRequest(**question.body)
    plan, _, _ = validate_and_merge(rule_plan(request), request, source="rule planner")
    for name in question.checks():
        read(plan, name)  # raises KeyError on an unknown check


def test_read_rejects_an_unknown_check() -> None:
    request = VisualizationRequest(query="How many trials are there?")
    plan, _, _ = validate_and_merge(rule_plan(request), request, source="rule planner")
    with pytest.raises(KeyError):
        read(plan, "not_a_check")


def test_rule_planner_scores_and_plans_every_question() -> None:
    """The rule planner must at least produce a usable plan for all fifteen.
    The accuracy floor is deliberately low: this guards against a regression
    that breaks planning, not against the score drifting a little."""
    score = run_rule_planner()
    totals = score.totals()
    assert totals["questions"] == 15
    assert totals["plan_errors"] == 0
    assert totals["intent_accuracy"] >= 0.7


def test_a_planner_that_fails_still_counts_against_the_total() -> None:
    """A missing plan fails every check rather than shrinking the denominator,
    so a planner cannot score well by refusing the hard questions."""
    question = QUESTIONS[0]
    result = score_question(question, None, "boom")
    assert result["checks_total"] == len(question.checks())
    assert result["checks_passed"] == 0
    assert result["exact"] is False
    assert result["plan_error"] == "boom"
