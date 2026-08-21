from types import SimpleNamespace

from orchestrator import leg_runtime as runtime


def test_build_augmented_task_keeps_current_context_and_future_goals_separate():
    prompt = runtime.build_augmented_task(
        {"type": "pickup", "text": "Pick up crackers"},
        context="Milk was already checked out.",
        future_legs=[{"text": "Go to checkout"}, "Report completion"],
    )

    assert "CURRENT GOAL: Pick up crackers" in prompt
    assert "CONTEXT FROM PREVIOUS SUBTASKS:\nMilk was already checked out." in prompt
    assert "  1. Go to checkout\n  2. Report completion" in prompt
    assert "do NOT pursue these yet" in prompt


def test_inspect_prompt_preserves_the_restricted_action_contract():
    prompt = runtime.build_augmented_task(
        {"type": "inspect", "text": "Read the expiration date"}
    )

    assert "Never grab, release, or check out during this inspect leg" in prompt
    assert "Reporting a definite absence is a valid answer" in prompt


def test_model_facing_state_removes_code_only_and_large_nested_fields():
    state = {
        "visited_checkpoints": {1, 2},
        "last_checkout": {"scanned": True, "steps": {"scan": "large"}},
        "last_inspection": None,
        "translation": (1, 0, 2),
    }

    view = runtime.model_facing_state(state)

    assert "visited_checkpoints" not in view
    assert view["last_checkout"] == {"scanned": True}
    assert view["translation"] == (1, 0, 2)


def test_grip_tracker_preserves_only_live_carried_names():
    state = {
        "leftGrippedState": True,
        "rightGrippedState": False,
    }
    tracker = runtime.GripTracker.from_state(
        state, {"left": "Piattos", "right": "Stale value"}
    )

    assert tracker.names == {"left": "Piattos", "right": None}
    assert tracker.start_grips == {"left"}
    assert state["gripped_name"] is None

    state["leftGrippedState"] = False
    state["rightGrippedState"] = True
    tracker.record_grab({"gripped": True, "hovered": "Ritz", "hand": "right"})
    tracker.reconcile(state)

    assert tracker.names == {"left": None, "right": "Ritz"}
    assert state["released_grip_this_leg"] is True
    assert state["new_grip_this_leg"] is True


def test_reconcile_after_actions_merges_durable_state_and_visit_tracking():
    live_state = {
        "translation": (4, 0, 6),
        "leftGrippedState": True,
        "rightGrippedState": False,
    }
    grip_tracker = runtime.GripTracker(
        names={"left": "Ritz", "right": None}, start_grips=set()
    )
    evidence = runtime.InspectionEvidence()
    metrics = {"t_checkout": None, "t_grip": None}
    visited = {2}
    outcome = SimpleNamespace(
        blocked_reason=False,
        center_message="centered",
        last_reach={"reachable": True},
        grab_failed=False,
        checkout_result={"scanned": True},
    )
    store_map = SimpleNamespace(nearest_checkpoint=lambda _position: 7)

    state, near = runtime.reconcile_after_actions(
        outcome=outcome,
        mode="manipulation",
        last_inspection_result=None,
        store_map=store_map,
        visited=visited,
        grip_tracker=grip_tracker,
        inspection_evidence=evidence,
        metrics=metrics,
        started_at=0,
        read_state=lambda: dict(live_state),
    )

    assert near == 7
    assert visited == {2, 7}
    assert state["visited_checkpoints"] == {2, 7}
    assert state["last_checkout"] == {"scanned": True}
    assert state["gripped_name"] == "Ritz"


def _reconcile_recovery(previous_count, live_count):
    live_state = {
        "translation": (1.5, 0.0, -2.0),
        "rotation": (0.0, 27.0, 0.0),
        "out_of_bounds_recovery_count": live_count,
        "leftGrippedState": True,
        "rightGrippedState": False,
    }
    prior = {
        "out_of_bounds_recovery_count": previous_count,
        "last_checkout": {"scanned": True},
        "last_inspection": {"label_legible": True},
    }
    tracker = runtime.GripTracker(
        names={"left": "Ritz", "right": None}, start_grips={"left"}
    )
    evidence = runtime.InspectionEvidence(
        by_hand={
            "left": {
                "hand": "left",
                "sku": "Ritz",
                "step": 2,
                "label_legible": True,
                "best_effort_read": False,
                "image_b64": "frame",
            }
        }
    )
    outcome = SimpleNamespace(
        blocked_reason=False,
        center_message="stale center",
        last_reach="stale reach",
        grab_failed=True,
        checkout_result=None,
    )
    state, near = runtime.reconcile_after_actions(
        outcome=outcome,
        mode="navigation",
        last_inspection_result=prior["last_inspection"],
        store_map=SimpleNamespace(nearest_checkpoint=lambda _position: 9),
        visited={4},
        grip_tracker=tracker,
        inspection_evidence=evidence,
        metrics={"t_checkout": None, "t_grip": 1.0},
        started_at=0,
        read_state=lambda: dict(live_state),
        previous_state=prior,
    )
    return state, near


def test_recovery_increment_reconciles_authoritative_pose_and_clears_stale_targeting():
    state, near = _reconcile_recovery(3, 4)

    assert near == 9
    assert state["position_recovery"] == {
        "count": 4,
        "position": (1.5, 0.0, -2.0),
        "nearest_checkpoint": 9,
    }
    assert state["last_center"] is None
    assert state["last_reach"] is None
    assert state["last_grab_failed"] is False
    assert state["leftGrippedState"] is True
    assert state["gripped_name"] == "Ritz"
    assert state["inspection_evidence"][0]["sku"] == "Ritz"
    assert state["last_inspection"] == {"label_legible": True}
    assert state["last_checkout"] == {"scanned": True}
    assert state["visited_checkpoints"] == {4, 9}


def test_recovery_notice_is_not_repeated_for_unchanged_missing_or_reset_counts():
    for previous_count, live_count in ((4, 4), (4, None), (None, 4), (4, 0)):
        state, _near = _reconcile_recovery(previous_count, live_count)

        assert state["position_recovery"] is None
        assert state["last_center"] == "stale center"
        assert state["last_reach"] == "stale reach"
        assert state["last_grab_failed"] is True
