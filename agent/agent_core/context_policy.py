"""Named context-window ablations shared by the standalone agent and Sari Bench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ContextPolicy:
    """Controls only context retention; baseline values preserve the historical behavior."""

    semantic_dedupe: float | None = None
    semantic_dedupe_window: int = 8
    semantic_keep_last: int | None = None
    findings_max_chars: int | None = None
    findings_enabled: bool = True
    episodic_in_actor: bool = True
    actor_image_history: int | None = None


CONTEXT_POLICIES: Mapping[str, ContextPolicy] = {
    "baseline": ContextPolicy(),
    "baseline-2img": ContextPolicy(actor_image_history=2),
    "a1": ContextPolicy(semantic_keep_last=0),
    "a1-2img": ContextPolicy(semantic_keep_last=0, actor_image_history=2),
    "a2c": ContextPolicy(semantic_dedupe=0.80, semantic_keep_last=12),
    "a2c-2img": ContextPolicy(
        semantic_dedupe=0.80, semantic_keep_last=12, actor_image_history=2
    ),
    "a3": ContextPolicy(findings_enabled=False),
    "a3-2img": ContextPolicy(findings_enabled=False, actor_image_history=2),
    "a4": ContextPolicy(findings_max_chars=600),
    "a4-2img": ContextPolicy(findings_max_chars=600, actor_image_history=2),
    "a5": ContextPolicy(episodic_in_actor=False),
    "a5-2img": ContextPolicy(episodic_in_actor=False, actor_image_history=2),
    "a6-2": ContextPolicy(actor_image_history=2),
    "a6-4": ContextPolicy(actor_image_history=4),
}
CONTEXT_POLICY_NAMES = tuple(CONTEXT_POLICIES)


def validate_context_policy(policy: ContextPolicy) -> ContextPolicy:
    """Validate a policy at the boundary where it becomes active."""

    if policy.semantic_dedupe is not None and not 0.0 <= policy.semantic_dedupe <= 1.0:
        raise ValueError("semantic_dedupe must be between 0.0 and 1.0")
    if policy.semantic_dedupe_window < 1:
        raise ValueError("semantic_dedupe_window must be at least 1")
    if policy.semantic_keep_last is not None and policy.semantic_keep_last < 0:
        raise ValueError("semantic_keep_last cannot be negative")
    if policy.findings_max_chars is not None and policy.findings_max_chars < 1:
        raise ValueError("findings_max_chars must be at least 1")
    if policy.actor_image_history is not None and policy.actor_image_history < 1:
        raise ValueError("actor_image_history must be at least 1")
    return policy


def resolve_context_policy(name: str) -> ContextPolicy:
    """Resolve and validate one registry name."""

    try:
        policy = CONTEXT_POLICIES[name]
    except KeyError as error:
        choices = ", ".join(CONTEXT_POLICY_NAMES)
        raise ValueError(f"unknown context policy {name!r}; choose one of: {choices}") from error
    return validate_context_policy(policy)
