"""Representative fake-client coverage for structured model calls sharing one retry budget."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "agent",
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent_core.actors import VLMAgent
from agent_core.context_policy import ContextPolicy
from agent_core.llm import DEFAULT_API_MAX_ATTEMPTS, LLMConfig, configure_api_retries
from agent_core.runtime import EmbodiedAgent
from nav.locate_task import RESOLVE_SCHEMA, endpoint_json
from orchestrator.orchestrator_llm import decompose_task


class _Completions:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))]
        )


class _Client:
    def __init__(self, outcomes):
        self.completions = _Completions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.fixture(autouse=True)
def _retry_policy(monkeypatch):
    configure_api_retries(2)
    monkeypatch.setattr("agent_core.llm.time.sleep", lambda _delay: None)
    yield
    configure_api_retries(DEFAULT_API_MAX_ATTEMPTS)


def test_semantic_reasoning_retries_malformed_content():
    learner = SimpleNamespace(
        client=_Client(["not a mapping", "{'mode': 'navigation'}"]),
    )
    from agent_core.contracts import JSON_BLOCK_PATTERN
    from agent_core.llm import BaseAgent

    learner.extractable_json_structured_output = JSON_BLOCK_PATTERN
    learner._api_call_with_retry = BaseAgent._api_call_with_retry.__get__(learner)
    learner.config = LLMConfig(model_id="test", api_key="test")
    agent = object.__new__(EmbodiedAgent)
    agent.associative_learner = learner

    assert agent._call_associative("system", None, "state") == "{'mode': 'navigation'}"
    assert learner.client.completions.calls == 2


def test_actor_reasoning_retries_malformed_content_without_polluting_history():
    actor = object.__new__(VLMAgent)
    actor.config = LLMConfig(model_id="test", api_key="test")
    actor.client = _Client(["garbage", '{"actions": ["STOP"], "times": [1]}'])
    actor.context_policy = ContextPolicy()
    actor.history = []

    reply = actor.send_message([{"type": "text", "text": "act"}])

    assert "STOP" in reply
    assert actor.client.completions.calls == 2
    assert [row["role"] for row in actor.history] == ["user", "assistant"]


def test_decomposer_retries_malformed_content_before_using_contract_parser():
    client = _Client([
        "not json",
        '[{"type": "goto", "text": "Go to checkout", "location": "checkout"}]',
    ])

    result = decompose_task(client, "Go to checkout")

    assert result[0]["type"] == "goto"
    assert client.completions.calls == 2


def test_resolver_retries_malformed_schema_response(monkeypatch):
    bodies = iter([
        {"choices": [{"message": {"content": '{"target_name": "chips"}'}}]},
        {"choices": [{"message": {"content": (
            '{"target_name":"chips","target_appearance":"bag","candidates":[1],'
            '"tier":"name","reasoning":"index match"}'
        )}}]},
    ])

    calls = []

    def create(*_args, **_kwargs):
        calls.append(1)
        body = next(bodies)
        content = body["choices"][0]["message"]["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            model_dump=lambda **_kwargs: body,
        )

    monkeypatch.setattr("agent_core.llm.ChatEndpoint.create", create)
    result, _body = endpoint_json(
        "system", "prompt", RESOLVE_SCHEMA,
        base_url="http://127.0.0.1:8000", api_key="test",
    )

    assert result["candidates"] == [1]
    assert len(calls) == 2
