from types import SimpleNamespace
from dataclasses import replace
import json

import pytest

from orchestrator import orchestration
from orchestrator.task_response import new_response_memory
from orchestrator.plan_controller import PlanController
from orchestrator import plan_controller as plan_controller_module


def _config(tmp_path, *, retries=1, refusal_cap_action="continue"):
    return orchestration.OrchestrationConfig(
        task="Pick up the chips",
        arm="graph",
        caps=(10, 1.0),
        out=None,
        run_dir=str(tmp_path),
        resolver_backend="endpoint",
        reset_start=False,
        restart_env=False,
        leg_retries=retries,
        output_dir=None,
        completion_guard="deterministic",
        refusal_cap_action=refusal_cap_action,
        ocr_url=None,
        runs_dir=None,
        context_policy="baseline",
    )


def _state(tmp_path, *, retries=1, refusal_cap_action="continue"):
    agent = SimpleNamespace(vlm_agent=SimpleNamespace(episodic_memory="remembered"))
    return orchestration._RunState(
        config=_config(tmp_path, retries=retries, refusal_cap_action=refusal_cap_action),
        policy=SimpleNamespace(findings_enabled=False),
        run_dir=str(tmp_path),
        response_memory=new_response_memory("Pick up the chips"),
        agent=agent,
        store_map=object(),
    )


def _metrics(*, success, reason, held=None):
    return {
        "success": success,
        "end_reason": reason,
        "llm_calls": 1,
        "t_grip": None,
        "t_checkout": None,
        "timesteps": 1,
        "halts_refused": 0,
        "wall_s": 0.1,
        "final_state": {
            "last_halt_refused": None if success else "wrong item",
            "gripped_names": held or {},
        },
        "new_semantic_entries": "",
    }


def test_leg_retry_uses_failure_context_and_separate_log(monkeypatch, tmp_path):
    state = _state(tmp_path)
    leg = {"type": "pickup", "text": "Pick up the chips"}
    state.legs = [leg]
    calls = []
    results = [
        _metrics(success=False, reason="halt_refused"),
        _metrics(success=True, reason="halt_granted", held={"left": "chips"}),
    ]

    def fake_run_leg(*_args, **kwargs):
        calls.append(kwargs)
        return results.pop(0)

    monkeypatch.setattr(orchestration, "run_leg", fake_run_leg)
    monkeypatch.setattr(orchestration.token_meter, "snapshot", lambda: {})
    monkeypatch.setattr(
        orchestration.token_meter,
        "delta",
        lambda _before: {"tokens_in": 2, "tokens_out": 1, "by_role": {}},
    )
    monkeypatch.setattr(orchestration.token_meter, "dump", lambda *_args: None)

    metrics, attempt = orchestration._run_leg_with_retries(state, leg, 0)

    assert metrics["success"] is True
    assert attempt == 2
    assert calls[0]["context"] == ""
    assert "wrong item" in calls[1]["context"]
    assert calls[0]["log_path"].endswith("leg00.jsonl")
    assert calls[1]["log_path"].endswith("leg00_retry1.jsonl")
    assert [row["attempt"] for row in state.leg_rows] == [1, 2]
    assert state.carried_names == {"left": "chips"}
    assert state.llm_calls == 2


def test_execute_plan_aborts_remaining_legs(monkeypatch, tmp_path):
    state = _state(tmp_path, retries=0)
    state.legs = [
        {"type": "pickup", "text": "Pick up the chips"},
        {"type": "checkout", "text": "Check out"},
    ]
    attempted = []

    def fail_first(_state, _leg, leg_index):
        attempted.append(leg_index)
        return _metrics(success=False, reason="time_cap"), 1

    monkeypatch.setattr(orchestration, "_run_leg_with_retries", fail_first)

    orchestration._execute_plan(state)

    assert attempted == [0]
    assert state.success is False


def _two_legs():
    return [
        {"type": "pickup", "text": "Pick up the chips"},
        {"type": "checkout", "text": "Check out"},
    ]


def _plan_with_first_leg_forced(monkeypatch, state):
    """Fail leg 1 on the refusal cap, pass leg 2; return the leg indices actually run."""
    attempted = []

    def run(_state, _leg, leg_index):
        attempted.append(leg_index)
        if leg_index == 0:
            return _metrics(success=False, reason="halt_forced"), 1
        return _metrics(success=True, reason="halt_granted"), 1

    monkeypatch.setattr(orchestration, "_run_leg_with_retries", run)
    orchestration._execute_plan(state)
    return attempted


def test_execute_plan_continues_past_refusal_cap(monkeypatch, tmp_path):
    state = _state(tmp_path, retries=0)
    state.legs = _two_legs()

    attempted = _plan_with_first_leg_forced(monkeypatch, state)

    assert attempted == [0, 1]
    # The leg stays failed and is named as unverified; only the abort is waived.
    assert state.success is False
    assert state.unverified_legs == [1]


def test_execute_plan_halt_policy_aborts_on_refusal_cap(monkeypatch, tmp_path):
    state = _state(tmp_path, retries=0, refusal_cap_action="halt")
    state.legs = _two_legs()

    attempted = _plan_with_first_leg_forced(monkeypatch, state)

    assert attempted == [0]
    assert state.success is False
    assert state.unverified_legs == []


def test_exhausted_leg_revision_can_replace_failed_strategy_and_complete_goal(
    monkeypatch, tmp_path
):
    state = _state(tmp_path, retries=0)
    state.config = replace(state.config, adaptive_leg_replanning=True)
    initial = [{"type": "pickup", "text": "Pick up chips", "target": "chips"}]

    def resolve(_sm, _call, legs):
        for leg in legs:
            leg["feasible"] = True
        return legs, 1

    monkeypatch.setattr(plan_controller_module, "plan_legs", resolve)
    controller = PlanController(
        initial, object(), object(),
        lambda *_args: json.dumps({"revised_suffix": [{
            "type": "pickup", "text": "Pick up chips from the display", "target": "chips",
            "goal_id": "goal-001",
        }]}),
    )
    state.plan_controller = controller
    state.legs = controller.pending
    state.response_memory["experimental"] = {
        "adaptive_leg_replanning": True, "initial_plan": controller.initial_legs,
        "current_plan": controller.pending, "completed_goal_ids": [], "revision_events": [],
    }
    attempted = []

    def run(_state, leg, _index, **_kwargs):
        attempted.append(leg["text"])
        return (
            _metrics(success=len(attempted) == 2,
                     reason="halt_granted" if len(attempted) == 2 else "time_cap"),
            1,
        )

    monkeypatch.setattr(orchestration, "_run_leg_with_retries", run)
    orchestration._execute_revisable_plan(state)

    assert attempted == ["Pick up chips", "Pick up chips from the display"]
    assert state.success is True
    assert controller.completed_goal_ids == {"goal-001"}
    assert controller.accepted_revisions == 1


def test_orchestrate_preserves_run_error_and_always_closes(monkeypatch, tmp_path):
    state = _state(tmp_path)
    closed = []
    state.agent.close = lambda: closed.append(True)

    monkeypatch.setattr(orchestration, "_new_run_state", lambda _config: state)

    def initialize(current):
        current.ready_to_finalize = True

    monkeypatch.setattr(orchestration, "_initialize_runtime", initialize)
    monkeypatch.setattr(
        orchestration,
        "_plan_task",
        lambda _state: (_ for _ in ()).throw(RuntimeError("planning failed")),
    )
    monkeypatch.setattr(
        orchestration,
        "_finalize_run",
        lambda _state: (_ for _ in ()).throw(ValueError("finalization failed")),
    )
    monkeypatch.setattr(orchestration.chime, "beep", lambda: None)

    with pytest.raises(RuntimeError, match="planning failed"):
        orchestration.orchestrate(orchestration.OrchestrationConfig(
            task="Pick up the chips",
            run_dir=str(tmp_path),
        ))

    assert closed == [True]
    assert isinstance(state.error, RuntimeError)
    assert state.success is False
