import copy
import json

from orchestrator import plan_controller as module


class _Map:
    by_id = {1: object(), 2: object()}


def _legs():
    return [
        {"type": "pickup", "text": "Pick up chips", "target": "chips"},
        {"type": "checkout", "text": "Check out chips"},
    ]


def _controller(monkeypatch, reply):
    def resolve(_sm, _call, legs):
        for leg in legs:
            leg["feasible"] = leg.get("target") != "missing"
        return legs, 1

    monkeypatch.setattr(module, "plan_legs", resolve)
    return module.PlanController(_legs(), _Map(), object(), lambda *_args: json.dumps(reply))


def test_revision_can_insert_replace_and_reorder_without_deleting_goals(monkeypatch):
    suffix = [
        {"type": "goto", "text": "Go to aisle", "location": "aisle", "goal_id": None},
        {"type": "checkout", "text": "Check out chips", "goal_id": "goal-002"},
        {"type": "pickup", "text": "Get another chips pack", "target": "chips", "goal_id": "goal-001"},
    ]
    controller = _controller(monkeypatch, {"revised_suffix": suffix})

    result = controller.request_revision(
        {"reason_code": "missing_prerequisite", "evidence": "aisle unknown", "suggested_change": "locate it"},
        trigger="semantic",
    )

    assert result["accepted"] is True
    assert [leg.get("goal_id") for leg in controller.pending] == [None, "goal-002", "goal-001"]
    assert controller.inserted_legs == 1


def test_completed_goals_are_immutable_and_invalid_revisions_are_atomic(monkeypatch):
    controller = _controller(monkeypatch, {"revised_suffix": []})
    first = controller.pending[0]
    controller.complete(first)
    controller.remove_current(first)
    before = copy.deepcopy(controller.pending)
    controller.replanner_call = lambda *_args: json.dumps({"revised_suffix": [
        {"type": "pickup", "text": "Restore old goal", "target": "chips", "goal_id": "goal-001"},
        {"type": "checkout", "text": "Check out chips", "goal_id": "goal-002"},
    ]})

    result = controller.request_revision(
        {"reason_code": "dependency_change", "evidence": "changed", "suggested_change": "adapt"},
        trigger="semantic",
    )

    assert result["accepted"] is False
    assert controller.pending == before
    assert "exactly once" in result["feedback"]


def test_noop_infeasible_growth_and_request_caps_are_rejected(monkeypatch):
    controller = _controller(monkeypatch, {"revised_suffix": [
        {key: value for key, value in leg.items() if key != "feasible"}
        for leg in module.PlanController(_legs(), _Map(), object(), lambda *_: "").pending
    ]})
    request = {"reason_code": "stale_assumption", "evidence": "stale", "suggested_change": "change"}
    assert controller.request_revision(request, trigger="semantic")["accepted"] is False

    controller.replanner_call = lambda *_: json.dumps({"revised_suffix": [
        {"type": "pickup", "text": "Missing", "target": "missing", "goal_id": "goal-001"},
        {"type": "checkout", "text": "Check out", "goal_id": "goal-002"},
    ]})
    assert "infeasible" in controller.request_revision(request, trigger="semantic")["feedback"]

    for _ in range(5):
        controller.request_revision(request, trigger="semantic")
    assert controller.total_requests == module.MAX_REVISION_REQUESTS
    assert controller.accepted_revisions <= module.MAX_ACCEPTED_REVISIONS
