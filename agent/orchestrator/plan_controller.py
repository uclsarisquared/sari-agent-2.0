"""Simulator-free bounded controller for experimental unfinished-plan revision."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from agent_core.prompt_loader import load_prompt
from orchestrator.subtask_completion import SUBTASK_TYPES
from orchestrator.subtask_planning import plan_legs

MAX_ACCEPTED_REVISIONS = 2
MAX_REVISION_REQUESTS = 4
MAX_INSERTED_LEGS = 4

REPLANNER_SYSTEM = load_prompt("orchestrator/replanner")

_RESOLVED_FIELDS = {
    "candidates", "candidate_sets", "target_checkpoint", "target_name", "feasible", "tier",
}


def _strategy_view(leg: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in leg.items() if key not in _RESOLVED_FIELDS}


def _parse_suffix(raw: str) -> list[Any] | None:
    match = re.search(r"```\s*json\s*([\s\S]*?)\s*```", raw or "", re.I)
    blob = match.group(1) if match else str(raw or "").strip()
    try:
        value = json.loads(blob)
    except (TypeError, ValueError):
        return None
    suffix = value.get("revised_suffix") if isinstance(value, dict) else None
    return suffix if isinstance(suffix, list) else None


def _validate_typed_leg(value: Any) -> str | None:
    if not isinstance(value, dict):
        return "every revised leg must be an object"
    leg_type = value.get("type")
    if leg_type not in SUBTASK_TYPES:
        return f"unsupported revised leg type: {leg_type!r}"
    if not isinstance(value.get("text"), str) or not value["text"].strip():
        return "every revised leg requires non-empty text"
    allowed_fields = {
        "pickup": {"type", "text", "goal_id", "target", "count"},
        "checkout": {"type", "text", "goal_id"},
        "compare": {"type", "text", "goal_id", "targets", "criterion"},
        "goto": {"type", "text", "goal_id", "location"},
        "inspect": {"type", "text", "goal_id", "query"},
    }[leg_type]
    unknown = sorted(set(value) - allowed_fields)
    if unknown:
        return f"unexpected field(s) for {leg_type}: {', '.join(unknown)}"
    requirements = {
        "pickup": ("target", str),
        "goto": ("location", str),
        "inspect": ("query", str),
        "compare": ("targets", list),
    }
    required = requirements.get(leg_type)
    if required:
        field_name, kind = required
        field_value = value.get(field_name)
        if not isinstance(field_value, kind) or not field_value:
            return f"{leg_type} leg requires non-empty {field_name}"
        if kind is str and not field_value.strip():
            return f"{leg_type} leg requires non-empty {field_name}"
    if leg_type == "compare" and (
        len(value["targets"]) < 2
        or any(not isinstance(item, str) or not item.strip() for item in value["targets"])
    ):
        return "compare leg requires at least two non-empty targets"
    if leg_type == "compare" and (
        not isinstance(value.get("criterion"), str) or not value["criterion"].strip()
    ):
        return "compare leg requires a non-empty criterion"
    if "count" in value and (
        not isinstance(value["count"], int)
        or isinstance(value["count"], bool)
        or not 1 <= value["count"] <= 2
    ):
        return "pickup count must be an integer from 1 to 2"
    goal_id = value.get("goal_id")
    if goal_id is not None and not isinstance(goal_id, str):
        return "goal_id must be a string or null"
    return None


@dataclass
class PlanController:
    """Own stable goal identity and atomically commit validated unfinished suffixes."""

    initial_legs: list[dict[str, Any]]
    store_map: Any
    resolve_call: Callable
    replanner_call: Callable[[str, str], str]
    pending: list[dict[str, Any]] = field(init=False)
    completed_goal_ids: set[str] = field(default_factory=set)
    accepted_revisions: int = 0
    total_requests: int = 0
    replanner_calls: int = 0
    inserted_legs: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    _committed_plan_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.initial_legs = copy.deepcopy(self.initial_legs)
        for index, leg in enumerate(self.initial_legs, 1):
            leg["goal_id"] = f"goal-{index:03d}"
        self.pending = copy.deepcopy(self.initial_legs)
        self._committed_plan_size = len(self.initial_legs)

    @property
    def initial_plan_size(self) -> int:
        return len(self.initial_legs)

    @property
    def final_plan_size(self) -> int:
        return self._committed_plan_size

    @property
    def can_request_revision(self) -> bool:
        return (
            self.total_requests < MAX_REVISION_REQUESTS
            and self.accepted_revisions < MAX_ACCEPTED_REVISIONS
        )

    @property
    def outstanding_goal_ids(self) -> list[str]:
        return [
            leg["goal_id"] for leg in self.initial_legs
            if leg["goal_id"] not in self.completed_goal_ids
        ]

    def complete(self, leg: dict[str, Any]) -> None:
        goal_id = leg.get("goal_id")
        if goal_id:
            self.completed_goal_ids.add(goal_id)

    def remove_current(self, leg: dict[str, Any]) -> None:
        if self.pending and self.pending[0] is leg:
            self.pending.pop(0)
            return
        for index, candidate in enumerate(self.pending):
            if candidate is leg:
                self.pending.pop(index)
                return

    def request_revision(self, request: dict[str, Any], *, trigger: str) -> dict[str, Any]:
        before = copy.deepcopy(self.pending)
        event = {"trigger": trigger, "request": copy.deepcopy(request), "before_suffix": before}
        if self.total_requests >= MAX_REVISION_REQUESTS:
            return {**event, "accepted": False, "feedback": "task revision-request limit exhausted"}
        self.total_requests += 1
        if self.accepted_revisions >= MAX_ACCEPTED_REVISIONS:
            return self._reject(event, "accepted revision limit exhausted")

        user = json.dumps(
            {
                "completed_goal_ids": sorted(self.completed_goal_ids),
                "outstanding_goal_ids": self.outstanding_goal_ids,
                "current_suffix": [_strategy_view(leg) for leg in self.pending],
                "revision_request": request,
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            self.replanner_calls += 1
            suffix = _parse_suffix(self.replanner_call(REPLANNER_SYSTEM, user))
        except Exception as error:  # model availability is a rejection, never a partial commit
            return self._reject(event, f"replanner unavailable: {type(error).__name__}")
        if suffix is None:
            return self._reject(event, "replanner output was malformed")
        return self._validate_and_commit(suffix, event)

    def _validate_and_commit(self, suffix: list[Any], event: dict[str, Any]) -> dict[str, Any]:
        for leg in suffix:
            problem = _validate_typed_leg(leg)
            if problem:
                return self._reject(event, problem)

        outstanding = self.outstanding_goal_ids
        goal_ids = [leg.get("goal_id") for leg in suffix if leg.get("goal_id") is not None]
        if sorted(goal_ids) != sorted(outstanding) or len(goal_ids) != len(set(goal_ids)):
            return self._reject(event, "revised suffix must cover every outstanding goal exactly once")
        inserted = sum(1 for leg in suffix if leg.get("goal_id") is None)
        if inserted > MAX_INSERTED_LEGS or len(suffix) > len(outstanding) + MAX_INSERTED_LEGS:
            return self._reject(event, "revised suffix exceeds the plan-growth limit")
        if [_strategy_view(leg) for leg in suffix] == [
            _strategy_view(leg) for leg in self.pending
        ]:
            return self._reject(event, "revised suffix makes no material change")

        candidate = copy.deepcopy(suffix)
        candidate, resolver_calls = plan_legs(self.store_map, self.resolve_call, candidate)
        if any(not leg.get("feasible", True) for leg in candidate):
            return self._reject(event, "revised suffix contains an infeasible map target")

        self.pending = candidate
        self._committed_plan_size = len(self.completed_goal_ids) + len(candidate)
        self.accepted_revisions += 1
        self.inserted_legs += inserted
        event.update(
            accepted=True,
            feedback="revision accepted",
            after_suffix=copy.deepcopy(candidate),
            inserted_legs=inserted,
            resolver_calls=resolver_calls,
        )
        self.events.append(event)
        return event

    def _reject(self, event: dict[str, Any], feedback: str) -> dict[str, Any]:
        event.update(accepted=False, feedback=feedback, after_suffix=copy.deepcopy(self.pending))
        self.events.append(event)
        return event
