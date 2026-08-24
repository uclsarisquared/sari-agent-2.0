"""Offline tests for the agent's bounded, quiet LLM retry policy."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent_core.agent import BaseAgent, OpenRouterConfig
from agent_core.llm import (
    DEFAULT_API_MAX_ATTEMPTS,
    MalformedContentError,
    agent_vlm_config,
    call_with_api_retries,
    configure_api_retries,
    endpoint_creds,
)


@pytest.fixture(autouse=True)
def _reset_retry_policy():
    configure_api_retries(DEFAULT_API_MAX_ATTEMPTS)
    yield
    configure_api_retries(DEFAULT_API_MAX_ATTEMPTS)


class _Completions:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        message = SimpleNamespace(content=outcome)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _Client:
    def __init__(self, outcomes):
        self.completions = _Completions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


class _Agent(BaseAgent):
    def __init__(self):
        self.config = OpenRouterConfig(model_id="test", api_key="test")


def test_endpoint_url_owns_port_and_clients_append_only_v1(monkeypatch):
    monkeypatch.setenv("OPENAI_API_URL", "endpoint.example:9123")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    endpoint, key = endpoint_creds()

    assert endpoint == "http://endpoint.example:9123"
    assert key == "test-key"
    assert agent_vlm_config().base_url == "http://endpoint.example:9123/v1"


@pytest.mark.parametrize(
    "endpoint, message",
    [
        ("http://endpoint.example", "must include the endpoint port"),
        ("http://endpoint.example:9123/v1", "code appends /v1"),
    ],
)
def test_endpoint_url_rejects_missing_port_or_api_path(monkeypatch, endpoint, message):
    monkeypatch.setenv("OPENAI_API_URL", endpoint)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(RuntimeError, match=message):
        endpoint_creds()


def test_api_call_recovers_quietly(monkeypatch, caplog):
    sleeps = []
    monkeypatch.setattr("agent_core.llm.time.sleep", sleeps.append)
    client = _Client([TimeoutError("down"), TimeoutError("still down"), "recovered"])

    assert _Agent()._api_call_with_retry(client, []) == "recovered"
    assert client.completions.calls == 3
    assert sleeps == [1, 2]
    assert caplog.records == []


def test_api_call_raises_original_error_after_ten_attempts(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr("agent_core.llm.time.sleep", lambda _delay: None)
    signal_path = tmp_path / "api_retry_exhausted.json"
    monkeypatch.setenv("SARI_API_RETRY_EXHAUSTED_PATH", str(signal_path))
    final_error = TimeoutError("server stayed down")
    failures = [TimeoutError(f"timeout {n}") for n in range(9)] + [final_error]
    client = _Client(failures)

    with pytest.raises(TimeoutError) as raised:
        _Agent()._api_call_with_retry(client, [])

    assert raised.value is final_error
    assert client.completions.calls == 10
    assert caplog.records == []
    signal = json.loads(signal_path.read_text(encoding="utf-8"))
    assert signal["attempts"] == 10
    assert signal["call_name"] == "model_call"
    assert signal["failure_kind"] == "timeout"
    assert signal["error_type"] == "TimeoutError"
    assert signal["error"] == "server stayed down"


def test_non_transient_programming_error_fails_immediately(monkeypatch):
    sleeps = []
    monkeypatch.setattr("agent_core.llm.time.sleep", sleeps.append)
    error = ValueError("bad response handling")
    client = _Client([error])

    with pytest.raises(ValueError) as raised:
        _Agent()._api_call_with_retry(client, [])

    assert raised.value is error
    assert client.completions.calls == 1
    assert sleeps == []


def test_configurable_attempts_and_exact_capped_delay_sequence(monkeypatch):
    configure_api_retries(10)
    sleeps = []
    monkeypatch.setattr("agent_core.llm.time.sleep", sleeps.append)
    outcomes = iter([TimeoutError("down")] * 9 + ["ok"])

    def operation():
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    assert call_with_api_retries(operation, call_name="semantic") == "ok"
    assert sleeps == [1, 2, 4, 8, 15, 30, 30, 30, 30]


def test_malformed_content_retries_inside_same_budget(monkeypatch):
    configure_api_retries(3)
    monkeypatch.setattr("agent_core.llm.time.sleep", lambda _delay: None)
    outcomes = iter(["bad", "still bad", "valid"])

    def validate(content):
        if content != "valid":
            raise MalformedContentError("invalid JSON", content=content)
        return content

    assert call_with_api_retries(
        lambda: next(outcomes), call_name="resolver", validator=validate
    ) == "valid"


def test_recoverable_caller_can_suppress_malformed_exhaustion_signal(monkeypatch, tmp_path):
    configure_api_retries(1)
    signal_path = tmp_path / "api_retry_exhausted.json"
    monkeypatch.setenv("SARI_API_RETRY_EXHAUSTED_PATH", str(signal_path))

    def malformed():
        raise MalformedContentError("invalid bbox JSON")

    with pytest.raises(MalformedContentError):
        call_with_api_retries(
            malformed,
            call_name="perception.bounding_boxes",
            signal_malformed_content_exhaustion=False,
        )

    assert not signal_path.exists()


def test_suppressing_malformed_signal_does_not_suppress_transport_signal(monkeypatch, tmp_path):
    configure_api_retries(1)
    signal_path = tmp_path / "api_retry_exhausted.json"
    monkeypatch.setenv("SARI_API_RETRY_EXHAUSTED_PATH", str(signal_path))

    def timeout():
        raise TimeoutError("endpoint unavailable")

    with pytest.raises(TimeoutError):
        call_with_api_retries(
            timeout,
            call_name="perception.bounding_boxes",
            signal_malformed_content_exhaustion=False,
        )

    signal = json.loads(signal_path.read_text(encoding="utf-8"))
    assert signal["failure_kind"] == "timeout"
