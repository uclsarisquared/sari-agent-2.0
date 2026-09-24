from types import SimpleNamespace

from orchestrator import leg_completion
from orchestrator.leg_runtime import GripTracker, InspectionEvidence


def _metrics():
    return {
        "llm_calls": 0,
        "success": False,
        "end_reason": None,
        "completion_evidence": None,
        "reported_answer": None,
        "halts_refused": 0,
        "halt_forced": False,
        "corrective_release": None,
    }


def test_disabled_guard_clears_stale_implicit_completion():
    metrics = _metrics()
    controller = leg_completion.CompletionController(
        SimpleNamespace(), {"type": "goto"}, "none", metrics, lambda _row: None, 1
    )
    state = {"goal_check": "stale"}

    guards = controller.prepare(
        state, "frame", 1, InspectionEvidence(), last_actor_text=""
    )
    observation = controller.observe_after_action(state, "", guards, 1)

    assert state["goal_check"] is None
    assert observation.met is False
    assert observation.terminal is False


def test_granted_stop_updates_the_public_result_contract(monkeypatch):
    monkeypatch.setattr(
        leg_completion, "reported_completion_answer", lambda _response: "verified answer"
    )
    monkeypatch.setattr(
        leg_completion, "completion_predicate", lambda *_args, **_kwargs: (True, "verified")
    )
    metrics = _metrics()
    controller = leg_completion.CompletionController(
        SimpleNamespace(), {"type": "inspect"}, "deterministic", metrics,
        lambda _row: None, 2,
    )
    tracker = GripTracker(names={"left": None, "right": None}, start_grips=set())

    status = controller.handle_stop(
        {"halt": True}, {}, "", leg_completion.StepGuards(), 3, tracker
    )

    assert status == "granted"
    assert metrics["success"] is True
    assert metrics["end_reason"] == "halt_granted"
    assert metrics["completion_evidence"] == "verified"
    assert metrics["reported_answer"] == "verified answer"


def test_refused_stop_keeps_the_claimed_answer_separately(monkeypatch):
    monkeypatch.setattr(
        leg_completion, "reported_completion_answer", lambda _response: "claimed answer"
    )
    monkeypatch.setattr(
        leg_completion, "completion_predicate", lambda *_args, **_kwargs: (False, "rejected")
    )
    metrics = _metrics()
    controller = leg_completion.CompletionController(
        SimpleNamespace(), {"type": "compare"}, "deterministic", metrics,
        lambda _row: None, 2,
    )
    tracker = GripTracker(names={"left": None, "right": None}, start_grips=set())
    state = {}

    status = controller.handle_stop(
        {"halt": True}, state, "", leg_completion.StepGuards(), 3, tracker
    )

    assert status == "refused"
    assert metrics["success"] is False
    assert metrics.get("reported_answer") is None
    assert metrics["refused_reported_answer"] == "claimed answer"
    assert state["last_halt_refused"] == "rejected"



def test_compare_guard_logs_the_step_it_is_evaluated_on(monkeypatch):
    rows = []
    captured = {}

    def fake_make_compare_guard(*_args, on_verdict=None):
        captured["on_verdict"] = on_verdict
        return lambda *_a: None

    monkeypatch.setattr(leg_completion, "make_compare_guard", fake_make_compare_guard)
    vlm = SimpleNamespace(client=None, config=SimpleNamespace(model_id="m"))
    leg = {"type": "compare", "targets": ["a", "b"], "candidate_sets": [[1], [2]]}
    controller = leg_completion.CompletionController(
        SimpleNamespace(vlm_agent=vlm), leg, "vlm", _metrics(), rows.append, 1
    )
    for step, near in ((1, 1), (2, 2), (5, 2)):
        controller.prepare({"nearest_checkpoint": near}, "img", step, InspectionEvidence(), "")

    captured["on_verdict"]("cheaper", {}, {"match": True}, False)

    assert rows[-1]["guard"] == "compare" and rows[-1]["step"] == 5


def test_vlm_backstop_records_completion_evidence():
    metrics = _metrics()
    controller = leg_completion.CompletionController(
        SimpleNamespace(), {"type": "pickup", "target": "chips"}, "vlm", metrics,
        lambda _row: None, 1,
    )

    controller._complete_without_stop(4, "held chips", backend="vlm", guard_verdicts={})

    assert metrics["end_reason"] == "completed_no_stop"
    assert metrics["completion_evidence"] == "held chips"
