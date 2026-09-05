"""Validation and execution of actor-selected simulator actions."""

import ast
import json
import re

from sim.env import RequestLidarCenter, TransformHands
from manip.manipulation import plan_reach
from toolset.actions import (
    MANIPULATION_ACTIONS_REF,
    NAVIGATION_ACTIONS_REF,
    PERCEPTION_ACTIONS_REF,
)
from orchestrator.held_item_inspection import _run_held_item_inspection_macro


def _crouched_grab(action_ref, time_units, target_info, debug_dir, plan):
    """Crouch, re-center, re-measure, and grab if reachable; always stand back up.

    Crouching shifts the camera, so re-centering prevents grabbing a neighboring item.
    This call changes posture only; the router handles any required body movement.
    """
    from sim.env import SetCrouch
    from vision.perception import center_object_on_screen
    base = {"gripped": False, "reach_verdict": "crouch", "move_steps": 0}
    try:
        SetCrouch(True)
        c = center_object_on_screen(target_info, debug_dir=debug_dir) or {}
        if not c.get("centered"):
            base["last_reach"] = ("CROUCHED but could not RE-CENTER the target from the lower view "
                                  f"({c.get('outcome', 'no result')}) - stood back up; bring the target "
                                  "into view and retry the grab")
            return base
        plan2 = plan_reach(RequestLidarCenter())
        if plan2["verdict"] == "reachable":
            result = action_ref(time_units) or {}
            result["reach_verdict"] = "crouch"
            result["last_reach"] = (
                f"CROUCHED and GRABBED - {plan2['reason']}" if result.get("gripped") else
                f"CROUCHED into range but the grab MISSED - {plan2['reason']}; stood back up - "
                f"move a little closer, re-center, and retry")
            return result
        if plan2["verdict"] == "move":
            base["move_steps"] = plan2["move_steps"]
            base["last_reach"] = (f"CROUCHED but still too far - {plan2['reason']}; stood back up - "
                                  f"move that distance, re-center, and retry (it will crouch again)")
            return base
        base["last_reach"] = (f"CROUCHED but the target is still out of reach "
                              f"({plan2['verdict']}: {plan2['reason']}) - stood back up")
        return base
    finally:
        SetCrouch(False)   # ALWAYS stand back up - crouch must never leak past this call


def _last_reach_line(plan, gripped=None):
    """Format the measured reach verdict for the actor and learner."""
    v = plan["verdict"]
    if v == "reachable":
        if gripped:
            return f"REACHABLE and GRABBED - {plan['reason']}"
        return (f"REACHABLE by measure but the grab MISSED - {plan['reason']}; move a little closer, "
                f"re-center, and retry (the reach envelope may be slightly off)")
    if v == "move":
        return f"MOVE - {plan['reason']}; then re-center and retry the grab"
    if v == "crouch":
        return f"CROUCH (too low) - {plan['reason']}"
    if v == "bail":
        return f"UNREACHABLE (too high) - {plan['reason']}"
    if v == "recenter":
        return f"RE-CENTER - {plan['reason']}"
    return f"{v.upper()} - {plan.get('reason', '')}"
# Macros need the agent's live NavSession and share the manipulation-mode gate.
# Bare names select a hand from grip state; suffixed names pin the side.
_MACRO_ACTIONS = {"checkout_held_item": "auto",
                  "checkout_held_item_left": "left",
                  "checkout_held_item_right": "right"}
_GRAB_ACTIONS = {"extend_arm_until_grabbed",
                 "extend_arm_until_grabbed_left",
                 "extend_arm_until_grabbed_right"}
_INSPECT_MACRO_ACTIONS = {
    "inspect_held_item": "auto",
    "inspect_held_item_left": "left",
    "inspect_held_item_right": "right",
}
_INSPECT_HELD_ACTIONS = set(_INSPECT_MACRO_ACTIONS)
_INSPECT_VISUAL_ACTIONS = {
    "pan_left", "pan_right", "tilt_up", "tilt_down", "center_object_on_screen",
}
# Unheld inspections allow limited repositioning; graph navigation owns longer travel.
# Exhausting this budget directs the actor to report absence and STOP.
_INSPECT_APPROACH_ACTIONS = {"move_forward", "move_backward", "move_left", "move_right"}
_INSPECT_MOVE_BUDGET_STEPS = 20   # 20 x 0.1 m = 2.0 m of repositioning per unheld inspect leg
def _grab_ready(state):
    """Allow grab promotion only after measured centering or reach success."""
    lc = state.get("last_center") or ""
    lr = state.get("last_reach") or ""
    return lc.startswith("SUCCESS") or lr.startswith("REACHABLE")


# Apostrophes in free-text fields can break the actor's Python-literal response.
# Recover the flat actions/times lists when the full dict cannot be parsed.
_ACTOR_LIST_RE = lambda key: re.compile(r"['\"]%s['\"]\s*:\s*\[([^\[\]]*)\]" % key, re.DOTALL)
_ACTOR_ITEM_RE = re.compile(r"""['"]([^'"]+)['"]""")


def _salvage_actions_times(blob: str):
    """Recover flat actions/times lists from malformed quoting, or None on mismatch."""
    am, tm = _ACTOR_LIST_RE("actions").search(blob), _ACTOR_LIST_RE("times").search(blob)
    if not (am and tm):
        return None
    actions = [a.strip() for a in _ACTOR_ITEM_RE.findall(am.group(1)) if a.strip()]
    times = [int(x) for x in re.findall(r"-?\d+", tm.group(1))]
    if actions and len(actions) == len(times):
        return actions, times
    return None


def parse_actor_response(text: str, pattern) -> dict:
    """Try Python literals, JSON, then actions/times salvage; return None on failure."""
    m = re.search(pattern, text or "")
    blob = m.group(1) if m else (text or "").strip()
    if not blob:
        return None
    for parse in (ast.literal_eval, json.loads):
        try:
            d = parse(blob)
        except Exception:  # noqa: BLE001 - any parse failure just falls through to the next tier
            continue
        if isinstance(d, dict) and "actions" in d:
            return d
    salvaged = _salvage_actions_times(blob)
    if salvaged:
        actions, times = salvaged
        return {"actions": actions, "times": times, "notes": {}}
    return None


def dispatch_action(action: str, time_units: int, notes: dict, inline_arg: str = None,
                    mode: str = None, debug_dir: str = None, agent=None, leg_type: str = None,
                    state: dict = None, inspection_query: str = None,
                    inspection_log=None, inspection_frames_dir: str = None,
                    inspect_move_allowance: int = 0) -> dict:
    """Execute one action and return its result, including any grab or checkout verdict.

    Check the checkout leg restriction before the manipulation-mode gate so a wrong-leg
    action requests STOP instead of a mode switch. leg_type=None disables the leg gate.
    """
    inspect_move_steps = 0   # >0 only when THIS call spends part of an inspect leg's approach budget
    if leg_type == "inspect":
        inspect_state = state
        if not isinstance(inspect_state, dict):
            try:
                inspect_state = TransformHands(
                    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
            except Exception:
                inspect_state = {}
        held = bool(inspect_state.get("leftGrippedState")
                    or inspect_state.get("rightGrippedState"))
        allowed = _INSPECT_HELD_ACTIONS if held else _INSPECT_VISUAL_ACTIONS
        # The approach budget is UNHELD-ONLY: a held-item inspect reads a label already in hand, so
        # walking cannot help it, and the restricted macro owns that path end to end.
        approach = (not held) and action in _INSPECT_APPROACH_ACTIONS
        if approach and inspect_move_allowance > 0:
            time_units = max(1, min(int(time_units), int(inspect_move_allowance)))
            inspect_move_steps = time_units
        elif action not in allowed:
            if approach:
                reason = (f"{action} is outside inspect scope: this inspect leg has spent its "
                          "repositioning allowance. If the target still is not visible from here it "
                          "is not at this location - emit STOP and report that in reported_answer")
            else:
                reason = (
                    f"{action} is outside inspect scope: "
                    + ("a held-item inspect leg allows only a restricted inspect_held_item macro"
                       if held else
                       "an unheld inspect leg allows only camera pan/tilt, visual centering, and a "
                       "small metered amount of stepping closer")
                )
            print(f"[BLOCKED] {reason}.")
            return {"blocked": True, "executed": False, "reason": reason,
                    "inspect_scope_violation": True}
    if action in _MACRO_ACTIONS and leg_type is not None and leg_type != "checkout":
        print(f"[BLOCKED] '{action}' belongs to the *checkout* subtask, not this "
              f"'{leg_type}' leg. If the CURRENT GOAL is complete, choose STOP to hand off.")
        return {"blocked": True, "reason": (f"{action} belongs to the checkout subtask; "
                "if your CURRENT GOAL is complete choose STOP to hand off (do not check out here)")}
    if (action in MANIPULATION_ACTIONS_REF or action in _MACRO_ACTIONS
            or action in _INSPECT_MACRO_ACTIONS) \
            and mode is not None and mode != "manipulation":
        print(f"[BLOCKED] '{action}' only works in *manipulation* mode (current mode: {mode}); "
              f"the hands are inactive otherwise. Skipped - route to manipulation first.")
        return {"blocked": True, "reason": f"{action} requires manipulation mode (was {mode})"}
    if action in _MACRO_ACTIONS:
        # The 6.3 deterministic checkout macro - drive to the counter, align, scan, bag - one call.
        # Needs the agent for its live nav session; run_leg passes it through. The variant name pins
        # the hand: 'auto' checks out EVERY held item in one fused pass (both hands = scan-scan-bag-bag
        # off one drive+align, degrading to a single item when one hand holds); _left/_right pin one.
        if agent is None:
            print(f"[WARN] {action} dispatched without an agent - cannot reach the nav "
                  f"session; skipped.")
            return {"blocked": True, "reason": f"{action} needs the agent (no nav session)"}
        checkout = getattr(agent, "checkout_held_item", None)
        checkout = checkout or getattr(agent, "_checkout_held_item", None)
        if not callable(checkout):
            return {"blocked": True, "reason": "agent has no checkout service"}
        return checkout(hand=_MACRO_ACTIONS[action]) or {}
    if action in _INSPECT_MACRO_ACTIONS:
        return _run_held_item_inspection_macro(
            agent,
            inspection_query,
            state,
            log_event=inspection_log,
            frames_dir=inspection_frames_dir,
            hand=_INSPECT_MACRO_ACTIONS[action],
        )
    if action in NAVIGATION_ACTIONS_REF:
        action_ref = NAVIGATION_ACTIONS_REF[action]
    elif action in PERCEPTION_ACTIONS_REF:
        action_ref = PERCEPTION_ACTIONS_REF[action]
    elif action in MANIPULATION_ACTIONS_REF:
        action_ref = MANIPULATION_ACTIONS_REF[action]
    else:
        print(f"[WARN] Unknown action skipped: {action}")
        return {}

    main_goal = notes.get('main_goal', '')
    sub_goals = notes.get('sub_goal', '')
    key_info  = notes.get('key_info', '')
    checklist = notes.get('checklist', '')

    if action == "center_object_on_screen":
        target_info = f"main_goal={main_goal}\nsub_goals={sub_goals}\nkey_info={key_info}\nchecklist={checklist}"
        # debug_dir (when a runner passes one) makes center_object_on_screen drop its per-look
        # candidate/locked/aim frames there - see the runners' per-step screenshot logging.
        return action_ref(target_info, debug_dir=debug_dir) or {}
    elif action in _GRAB_ACTIONS:
        # Measure reach along camera gaze; hands are excluded from LiDAR.
        # Surface non-reachable verdicts for recovery. Transport failures or missing
        # pose data retain the blind-reach fallback.
        plan = None
        try:
            plan = plan_reach(RequestLidarCenter())
        except Exception as e:
            print(f"[REACH] RequestLidarCenter/plan_reach failed ({type(e).__name__}: {e}); "
                  f"falling back to a blind reach.")
        if plan is None or plan["verdict"] == "unavailable":
            result = action_ref(time_units) or {}
            if not result.get('gripped', False):
                print(f"[GRAB] {action} (hand={result.get('hand', '?')}) did not grip — "
                      f"item out of reach, reposition.")
            return result
        if plan["verdict"] == "reachable":
            result = action_ref(time_units) or {}
            result["reach_verdict"] = "reachable"
            result["last_reach"] = _last_reach_line(plan, gripped=result.get("gripped", False))
            if not result.get('gripped', False):
                print(f"[GRAB] plan_reach said reachable but the grab (hand="
                      f"{result.get('hand', '?')}) missed - {plan['reason']}; "
                      f"the reach envelope may need retuning.")
            return result
        if plan["verdict"] == "crouch":
            # AUTO-CROUCH: resolved inside this call (crouch -> re-center -> re-measure -> grab ->
            # ALWAYS stand). See _crouched_grab - posture never leaks to the router/VLM.
            print(f"[REACH] crouch: {plan['reason']} - auto-crouching")
            target_info = (f"main_goal={main_goal}\nsub_goals={sub_goals}\n"
                           f"key_info={key_info}\nchecklist={checklist}")
            result = _crouched_grab(action_ref, time_units, target_info, debug_dir, plan)
            print(f"[REACH] {result.get('last_reach')}")
            return result
        # move / bail / recenter: skip the blind reach, report the measured verdict.
        print(f"[REACH] {plan['verdict']}: {plan['reason']}")
        return {"gripped": False, "reach_verdict": plan["verdict"],
                "move_steps": plan["move_steps"], "last_reach": _last_reach_line(plan)}
    else:
        result = action_ref(time_units) or {}
        if inspect_move_steps:
            # run_leg decrements the leg's remaining allowance by this; reported only when the budget
            # was actually spent, so non-inspect legs' results are byte-identical to before.
            result["inspect_move_steps"] = inspect_move_steps
        return result
