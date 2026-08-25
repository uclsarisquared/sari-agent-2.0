from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_core import llm
from agent_core.actors import VLMAgent
from agent_core.context_policy import ContextPolicy


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class _Response:
    def __init__(self, content, *, metadata=None, finish_reason="stop", usage=None):
        message = {"role": "assistant", "content": content, **(metadata or {})}
        self.choices = [SimpleNamespace(
            message=SimpleNamespace(**message), finish_reason=finish_reason
        )]
        self.usage = SimpleNamespace(**(usage or {"completion_tokens": 7}))


class _Completions:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _client(*outcomes):
    completions = _Completions(outcomes)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions), completions=completions
    )


@pytest.mark.parametrize(
    ("level", "workload", "expected"),
    [
        ("LOW", "guard", 1024), ("LOW", "localization", 1536),
        ("LOW", "reasoning", 4096), ("LOW", "annotation", 4096),
        ("MEDIUM", "guard", 2048), ("MEDIUM", "localization", 3072),
        ("MEDIUM", "reasoning", 6144), ("MEDIUM", "annotation", 6144),
        ("HIGH", "guard", 4096), ("HIGH", "localization", 4096),
        ("HIGH", "reasoning", 8192), ("HIGH", "annotation", 8192),
    ],
)
def test_vertex_budget_matrix(monkeypatch, level, workload, expected):
    monkeypatch.delenv(llm._VERTEX_TOKEN_OVERRIDE[workload], raising=False)
    assert llm.effective_max_tokens("vertex", 256, workload, level) == expected


def test_budget_minimal_overrides_and_vllm(monkeypatch):
    monkeypatch.delenv("VERTEX_MAX_TOKENS_GUARD", raising=False)
    assert llm.effective_max_tokens("vertex", 257, "guard", "MINIMAL") == 257
    assert llm.effective_max_tokens("vllm", 257, "guard", "HIGH") == 257
    monkeypatch.setenv("VERTEX_MAX_TOKENS_GUARD", "333")
    assert llm.effective_max_tokens("vertex", 257, "guard", "MINIMAL") == 333


@pytest.mark.parametrize("value", ["0", "-1", "1.5", " 3 ", "abc", ""])
def test_budget_override_rejects_non_exact_positive_integer(monkeypatch, value):
    monkeypatch.setenv("VERTEX_MAX_TOKENS_GUARD", value)
    with pytest.raises(llm.EndpointConfigurationError, match="exact positive integer"):
        llm.effective_max_tokens("vertex", 256, "guard", "LOW")


def test_provider_payloads_differ_without_mutating_messages():
    messages = [{"role": "user", "content": [{"type": "text", "text": "answer"}]}]
    vertex = _client(_Response('{"answer":"yes"}'))
    llm.structured_chat_completion(
        client=vertex, provider="vertex", thinking_level="LOW", default_extra_body={},
        messages=messages, schema=SCHEMA, schema_name="answer", model="gemini",
        max_tokens=100, call_name="vertex_payload",
    )
    vertex_call = vertex.completions.calls[0]
    assert vertex_call["max_tokens"] == 4096
    assert "response_format" in vertex_call
    assert "schema" not in vertex_call["messages"][0]["content"][0]["text"]

    qwen = _client(_Response("{'answer': 'yes'}"))
    llm.structured_chat_completion(
        client=qwen, provider="vllm", thinking_level=None, default_extra_body={},
        messages=messages, schema=SCHEMA, schema_name="answer", model="qwen",
        max_tokens=100, call_name="qwen_payload",
    )
    qwen_call = qwen.completions.calls[0]
    assert qwen_call["max_tokens"] == 100
    assert "response_format" in qwen_call
    assert '"required":["answer"]' in qwen_call["messages"][0]["content"][-1]["text"]
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "answer"}]}]


class _StatusError(RuntimeError):
    def __init__(self, status_code, text):
        super().__init__(text)
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code)
        self.body = {"error": text}


@pytest.mark.parametrize("native", ["not json", '{"wrong":"shape"}'])
def test_vertex_malformed_native_uses_one_prompt_fallback(native):
    client = _client(_Response(native), _Response("```json\n{\"answer\":\"ok\"}\n```"))
    result = llm.structured_chat_completion(
        client=client, provider="vertex", thinking_level="MINIMAL", default_extra_body={},
        messages=[{"role": "user", "content": "answer"}], schema=SCHEMA,
        schema_name="answer", model="gemini", call_name="fallback",
    )
    assert result.value == {"answer": "ok"}
    assert result.enforcement == "prompt_fallback"
    assert "response_format" in client.completions.calls[0]
    assert "response_format" not in client.completions.calls[1]


def test_vertex_schema_400_falls_back_but_unrelated_errors_do_not():
    client = _client(
        _StatusError(400, "response_format json_schema unsupported"),
        _Response('{"answer":"ok"}'),
    )
    result = llm.structured_chat_completion(
        client=client, provider="vertex", thinking_level="MINIMAL", default_extra_body={},
        messages=[{"role": "user", "content": "answer"}], schema=SCHEMA,
        schema_name="answer", model="gemini",
    )
    assert result.enforcement == "prompt_fallback"

    denied = _client(_StatusError(403, "permission denied"))
    with pytest.raises(_StatusError):
        llm.structured_chat_completion(
            client=denied, provider="vertex", thinking_level="MINIMAL", default_extra_body={},
            messages=[{"role": "user", "content": "answer"}], schema=SCHEMA,
            schema_name="answer", model="gemini",
        )
    assert len(denied.completions.calls) == 1


@pytest.mark.parametrize(
    "error",
    [
        _StatusError(400, "safety policy blocked the request"),
        _StatusError(401, "authentication failed"),
        _StatusError(403, "permission denied"),
    ],
)
def test_vertex_never_falls_back_for_non_schema_client_failures(error):
    client = _client(error)
    with pytest.raises(_StatusError):
        llm.structured_chat_completion(
            client=client, provider="vertex", thinking_level="LOW", default_extra_body={},
            messages=[{"role": "user", "content": "answer"}], schema=SCHEMA,
            schema_name="answer", model="gemini",
        )
    assert len(client.completions.calls) == 1
    assert "response_format" in client.completions.calls[0]


def test_vertex_transient_and_quota_retries_do_not_enter_fallback(monkeypatch):
    previous = llm.api_max_attempts()
    try:
        llm.configure_api_retries(2)
        monkeypatch.setattr(llm.time, "sleep", lambda _delay: None)
        for error in (TimeoutError("timed out"), _StatusError(429, "quota exhausted")):
            client = _client(error, error)
            with pytest.raises(type(error)):
                llm.structured_chat_completion(
                    client=client, provider="vertex", thinking_level="LOW",
                    default_extra_body={}, messages=[{"role": "user", "content": "answer"}],
                    schema=SCHEMA, schema_name="answer", model="gemini",
                )
            assert len(client.completions.calls) == 2
            assert all("response_format" in call for call in client.completions.calls)
    finally:
        llm.configure_api_retries(previous)


def test_vertex_malformed_fallback_candidate_retries_without_repair(monkeypatch):
    previous = llm.api_max_attempts()
    llm.configure_api_retries(2)
    monkeypatch.setattr(llm.time, "sleep", lambda _delay: None)
    client = _client(
        _Response("bad native"), _Response("bad fallback"),
        _Response('{"answer":"must not be consumed"}'),
    )
    try:
        result = llm.structured_chat_completion(
            client=client, provider="vertex", thinking_level="LOW", default_extra_body={},
            messages=[{"role": "user", "content": "answer"}], schema=SCHEMA,
            schema_name="answer", model="gemini",
        )
    finally:
        llm.configure_api_retries(previous)
    assert result.value == {"answer": "must not be consumed"}
    assert len(client.completions.calls) == 3


def test_invalid_and_recursive_schemas_fail_before_transport():
    client = _client()
    with pytest.raises(ValueError, match="invalid JSON Schema"):
        llm.structured_chat_completion(
            client=client, provider="vertex", thinking_level="LOW", default_extra_body={},
            messages=[{"role": "user", "content": "answer"}],
            schema={"type": "not-a-type"}, schema_name="bad", model="gemini",
        )
    with pytest.raises(ValueError, match="recursive JSON Schemas"):
        llm.validate_json_schema(
            {"type": "object", "properties": {"child": {"$ref": "#"}}},
            provider="vertex",
        )
    assert client.completions.calls == []


def test_representative_runtime_schemas_validate_real_instances():
    from annotate.annotator_sys_inst import CATEGORY_ENUM, SHELF_ANNOTATION_SCHEMA
    from drivers.vlm_planner import DECISION_SCHEMA
    from nav.locate_task import RESOLVE_SCHEMA, VERIFY_SCHEMA

    samples = [
        (RESOLVE_SCHEMA, {
            "target_name": "chips", "target_appearance": "bag", "candidates": [1],
            "tier": "name", "reasoning": "matched index",
        }),
        (VERIFY_SCHEMA, {
            "target_visible": False, "confidence": "low", "where": None,
            "evidence": "label is not visible", "seen_instead": [],
        }),
        (SHELF_ANNOTATION_SCHEMA, {
            "semantic_summary": "snack shelf", "shelf_type": [CATEGORY_ENUM[0]],
            "sign_text": None,
            "items": [{
                "name": "Sample", "variant": None, "price": None,
                "appearance": "red bag", "category": CATEGORY_ENUM[0],
            }],
        }),
        (DECISION_SCHEMA, {
            "reasoning": "nearest open frontier", "target_frontier_id": 1,
            "waypoint_x": 1.0, "waypoint_z": 2.0, "exploration_complete": False,
        }),
    ]
    for schema, value in samples:
        llm.validate_json_schema(schema, provider="vertex").validate(value)


def test_actor_preserves_thought_signature_across_pruning_and_next_turn():
    first = _Response(
        '{"actions":["STOP"],"times":[1]}',
        metadata={"thought_signature": "signed-thought-1"},
    )
    second = _Response(
        '{"actions":["STOP"],"times":[1]}',
        metadata={"thought_signature": "signed-thought-2"},
    )
    client = _client(first, second)
    actor = object.__new__(VLMAgent)
    actor.config = llm.LLMConfig(
        model_id="gemini", api_key="test", max_tokens=100, provider="vertex",
        extra_body={"google": {"thinking_config": {"thinking_level": "LOW"}}},
    )
    actor.client = client
    actor.context_policy = ContextPolicy(actor_image_history=1)
    actor.history = []
    actor.send_message([
        {"type": "image_url", "image_url": {"url": "old-image"}},
        {"type": "text", "text": "first"},
    ])
    actor.send_message([
        {"type": "image_url", "image_url": {"url": "new-image"}},
        {"type": "text", "text": "second"},
    ])
    assert actor.history[1]["thought_signature"] == "signed-thought-1"
    outbound_second = client.completions.calls[1]["messages"]
    assistant = next(row for row in outbound_second if row.get("thought_signature"))
    assert assistant["thought_signature"] == "signed-thought-1"
    old_user = next(row for row in outbound_second if row.get("role") == "user")
    assert [part["type"] for part in old_user["content"]] == ["text"]


def test_actor_keeps_only_final_malformed_assistant_metadata(monkeypatch):
    previous = llm.api_max_attempts()
    try:
        llm.configure_api_retries(2)
        monkeypatch.setattr(llm.time, "sleep", lambda _delay: None)
        client = _client(
            _Response("bad-one", metadata={"thought_signature": "discard"}),
            _Response("bad-two", metadata={"thought_signature": "keep"}),
        )
        actor = object.__new__(VLMAgent)
        actor.config = llm.LLMConfig(model_id="gemini", api_key="test", provider="vertex")
        actor.client = client
        actor.context_policy = ContextPolicy()
        actor.history = []
        assert actor.send_message([{"type": "text", "text": "act"}]) == "bad-two"
        assert len(actor.history) == 2
        assert actor.history[-1]["thought_signature"] == "keep"
    finally:
        llm.configure_api_retries(previous)
