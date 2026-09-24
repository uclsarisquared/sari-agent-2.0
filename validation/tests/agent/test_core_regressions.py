"""Regression tests for agent_core / root-config audit fixes."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from agent_core import llm, token_meter
from agent_core.actors import VLMAgent
from agent_core.context_policy import ContextPolicy
from agent_core.hands import HandController
from agent_core.navigation import GraphNavigator
import agent_core.navigation as navigation
from sari_runconfig import RunConfigError, load_run_config


class _StoreMap:
    by_id = {5: {}, 7: {}}

    def nearest_checkpoint(self, _xz):
        return 5

    def hops(self, a, b):
        return 0 if a == b else 1

    def checkpoint(self, _cp):
        return {"holds": [], "summary": ""}


def test_navigate_prefers_the_candidate_already_underfoot(monkeypatch):
    monkeypatch.setattr(navigation, "_sync_pose", lambda nav: None)
    monkeypatch.setattr(navigation, "_fresh_frame", lambda uri=None: b"png")
    goals = []
    nav = SimpleNamespace(pos=(0, 0, 0), args=SimpleNamespace(uri=None),
                          goto=lambda cp: goals.append(cp) or True)
    navigator = GraphNavigator(SimpleNamespace(set_pose=lambda _pose: None), nav_mode="graph")
    navigator.graph_nav = (_StoreMap(), nav)
    navigator.task, navigator.candidates = "task", [7, 5]
    navigator.navigate("task")
    assert goals == [5]


def _actor(outcome) -> VLMAgent:
    def create(**_kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    actor = object.__new__(VLMAgent)
    actor.config = llm.LLMConfig(model_id="m", api_key="k")
    actor.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    actor.context_policy = ContextPolicy()
    actor.history = []
    return actor


def test_actor_history_keeps_content_for_none_reply(monkeypatch):
    previous = llm.api_max_attempts()
    llm.configure_api_retries(1)
    try:
        message = SimpleNamespace(role="assistant", content=None)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="length")], usage=None
        )
        actor = _actor(response)
        assert actor.send_message([{"type": "text", "text": "act"}]) == ""
        assert actor.history[-1]["content"] == ""
        assert "ASSISTANT:" in actor.get_history_text()
    finally:
        llm.configure_api_retries(previous)


def test_actor_failed_call_does_not_leave_dangling_user_turn():
    actor = _actor(RuntimeError("auth failed"))
    with pytest.raises(RuntimeError):
        actor.send_message([{"type": "text", "text": "act"}])
    assert actor.history == []


def test_periodic_token_dump_is_claimed_by_one_thread(monkeypatch, tmp_path):
    monkeypatch.setattr(token_meter, "_run_dir", str(tmp_path))
    monkeypatch.setattr(token_meter, "_last_dump", 0.0)
    with token_meter._lock:
        claims = [token_meter._claim_dump_locked(), token_meter._claim_dump_locked()]
    assert claims == [True, False]


def test_hand_pose_is_unknown_after_a_failed_move(monkeypatch):
    def set_hand_pose(_pose, hand):
        if hand == "right":
            raise TimeoutError("sim wedged")
        return True, (0, 0, 0), 0.0

    monkeypatch.setitem(
        sys.modules, "manip.manipulation", types.SimpleNamespace(set_hand_pose=set_hand_pose)
    )
    hands = HandController(active=True, pose="rest")
    with pytest.raises(TimeoutError):
        hands.set_pose("grab")
    assert hands.pose is None


def test_endpoint_creds_ignores_malformed_conda_state(monkeypatch, tmp_path):
    state = tmp_path / "state"
    state.write_text("{not json", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("SARI_CONDA_STATE", str(state))
    assert llm.endpoint_creds() == (None, None)


def test_structured_parse_treats_unhashable_literal_as_malformed():
    completion = SimpleNamespace(text="{[1]: 2}", diagnostic=lambda: "diag")
    validator = llm.validate_json_schema({"type": "object"}, provider="vllm")
    with pytest.raises(llm.MalformedContentError):
        llm._parse_structured(completion, validator, tolerant=True, call_name="probe")


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_run_config_rejects_non_finite_limits(tmp_path, value):
    path = tmp_path / "run.toml"
    path.write_text(f"[limits]\nmax_minutes = {value}\n", encoding="utf-8")
    with pytest.raises(RunConfigError, match="finite"):
        load_run_config(path)
