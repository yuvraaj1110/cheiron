"""LLM planner and the make_plan orchestration, against a fake OpenAI client."""

from types import SimpleNamespace

import httpx
import openai

from app.api.schemas import VisualizationRequest
from app.planner.llm_planner import (
    LLM_SCHEMA,
    build_system_prompt,
    few_shot_examples,
    llm_schema,
)
from app.planner.plan import Dimension, Intent, QueryPlan
from app.planner.validator import make_plan

STRIPPED = {"default", "minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems", "pattern"}

VALID = '{"intent":"count","filters":{"condition":"asthma"},"interpretation":"Count of asthma trials."}'
INVALID = '{"intent":"count","interpretation":"x","extra":1}'
NEEDS_NETWORK = '{"intent":"network","interpretation":"missing node dims"}'


def reply(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    """Pops canned responses (or raises them) and records every call."""

    def __init__(self, responses: list) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, responses: list) -> None:
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


REQUEST = VisualizationRequest(query="How many asthma trials?")


async def test_valid_plan_is_used_after_one_call():
    client = FakeClient([reply(VALID)])
    plan, overrides, notes = await make_plan(REQUEST, client)
    assert plan.filters.condition == "asthma"
    assert len(client.completions.calls) == 1
    assert notes == [] and overrides == []


async def test_call_uses_strict_schema_and_temperature_zero():
    client = FakeClient([reply(VALID)])
    await make_plan(REQUEST, client)
    call = client.completions.calls[0]
    assert call["temperature"] == 0
    assert call["response_format"]["json_schema"]["strict"] is True


async def test_invalid_then_repaired_makes_two_calls():
    client = FakeClient([reply(INVALID), reply(VALID)])
    plan, _, notes = await make_plan(REQUEST, client)
    assert plan.intent == Intent.count
    assert len(client.completions.calls) == 2
    second = client.completions.calls[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": INVALID}
    assert "That plan was invalid" in second[-1]["content"]
    assert "extra" in second[-1]["content"]
    assert notes == []


async def test_incompatible_plan_is_repaired_too():
    client = FakeClient([reply(NEEDS_NETWORK), reply(VALID)])
    plan, _, _ = await make_plan(REQUEST, client)
    assert plan.intent == Intent.count
    assert "node_dims" in client.completions.calls[1]["messages"][-1]["content"]


async def test_invalid_twice_falls_back_to_rule_planner():
    client = FakeClient([reply(INVALID), reply(INVALID)])
    plan, _, notes = await make_plan(REQUEST, client)
    assert plan.intent == Intent.count  # "how many" -> count
    assert len(client.completions.calls) == 2
    assert any("LLM planner failed" in n and "used the rule planner" in n for n in notes)


async def test_openai_error_falls_back_to_rule_planner():
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com"))
    client = FakeClient([timeout])
    plan, _, notes = await make_plan(REQUEST, client)
    assert plan.intent == Intent.count
    assert any("LLM planner failed" in n for n in notes)


async def test_refusal_falls_back_to_rule_planner():
    refusal = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, refusal="no"))])
    client = FakeClient([refusal])
    _, _, notes = await make_plan(REQUEST, client)
    assert any("LLM planner failed" in n for n in notes)


async def test_no_key_uses_rule_planner(monkeypatch):
    monkeypatch.setattr("app.planner.llm_planner.settings.openai_api_key", "")
    plan, _, notes = await make_plan(REQUEST)
    assert plan.intent == Intent.count
    assert notes == ["No OpenAI key configured; used the rule planner."]


def all_keys(node) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {k for v in node.values() for k in all_keys(v)}
    if isinstance(node, list):
        return {k for v in node for k in all_keys(v)}
    return set()


def all_objects(node) -> list[dict]:
    if isinstance(node, dict):
        found = [node] if node.get("type") == "object" else []
        return found + [o for v in node.values() for o in all_objects(v)]
    if isinstance(node, list):
        return [o for v in node for o in all_objects(v)]
    return []


def test_schema_has_no_unsupported_keywords():
    assert not (all_keys(llm_schema()) & STRIPPED)
    assert LLM_SCHEMA == llm_schema()


def test_every_schema_object_forbids_extra_properties():
    objects = all_objects(LLM_SCHEMA)
    assert objects
    assert all(obj["additionalProperties"] is False for obj in objects)


def test_system_prompt_mentions_every_dimension_and_intent():
    prompt = build_system_prompt()
    for dim in Dimension:
        assert dim.value in prompt
    for intent in Intent:
        assert intent.value in prompt


def test_few_shot_examples_are_valid_and_cover_every_intent():
    examples = few_shot_examples()
    assert len(examples) >= 8
    for _, plan in examples:
        assert QueryPlan.model_validate_json(plan.model_dump_json(exclude_none=True)) == plan
    assert {plan.intent for _, plan in examples} == set(Intent)
