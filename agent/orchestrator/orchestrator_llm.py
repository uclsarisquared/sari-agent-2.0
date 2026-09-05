"""Model configuration and orchestrator-level LLM calls."""

import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent_core import token_meter
from agent_core.llm import (
    EndpointProfile, MalformedContentError, agent_vlm_config, call_with_api_retries,
    effective_max_tokens,
)
from agent_core.context_policy import ContextPolicy
from agent_core.artifact_sanitize import semantic_artifact_view
from agent_core.prompt_loader import load_prompt
from orchestrator.subtask_completion import TYPED_DECOMPOSER_SYSTEM, parse_decomposition

load_dotenv(Path(__file__).resolve().parent.parent.parent / "secrets.env")


_ENDPOINT_PROFILE = EndpointProfile.from_env()
ORCHESTRATOR_MODEL = _ENDPOINT_PROFILE.model

# Every reasoner runs on the OpenAI API compatible endpoint from secrets.env (OpenRouter fully
# retired 2026-07-21). agent_vlm_config carries the load-bearing enable_thinking=False +
# max_tokens cap - see agent.agent_vlm_config. Mirrors the pickup navigation evaluation. The
# orchestrator LLM below (_llm_client) already targets the same endpoint.
VLM_CONFIG = agent_vlm_config(temperature=0.5)
ASSOCIATIVE_CONFIG = agent_vlm_config(temperature=0.3)

FINDINGS_INPUT_MAX_CHARS = 200_000
FINDINGS_STATE_MAX_CHARS = 40_000


# Orchestrator LLM helpers

def _llm_client() -> OpenAI:
    """Build the shared orchestrator client from configured endpoint credentials."""
    return OpenAI(base_url=_ENDPOINT_PROFILE.base_url, api_key=_ENDPOINT_PROFILE.api_key,
                  max_retries=0)


def _llm_call(client: OpenAI, system: str, user: str, role: str, validator=None) -> str:
    """One orchestrator-level completion. `role` is which reasoner to bill it to - this helper serves
    the decomposer, findings reporter, and final responder, which are separately measurable, so the
    caller must say which one it is rather than letting them pool into one unreadable number."""
    with token_meter.role(role):
        def request():
            kwargs = dict(
                model=ORCHESTRATOR_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.3,
                timeout=120,
                extra_body=_ENDPOINT_PROFILE.extra_body,
            )
            budget = effective_max_tokens(
                _ENDPOINT_PROFILE.provider, None, "reasoning",
                _ENDPOINT_PROFILE.thinking_level,
            )
            if budget is not None:
                kwargs["max_tokens"] = budget
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        def validate(content):
            if not isinstance(content, str) or not content.strip():
                raise MalformedContentError("model response was empty", content=content)
            return validator(content) if validator is not None else content

        return call_with_api_retries(
            request,
            call_name=role,
            validator=validate,
        )


def decompose_task(client: OpenAI, task: str) -> list:
    """Phase 6.3: returns a list of TYPED subtask dicts ({"type", "text", ...}), not free strings, so
    each leg's completion is checked by a code-side predicate keyed on its type instead of by grepping
    its prose (the pre-6.3 keyword guards). The type vocabulary is closed (pickup|checkout|compare|
    goto); any untypeable element degrades to `{"type": "unknown"}` inside parse_decomposition, which
    run_leg then handles with the OLD keyword guards. The A/B that validated this prompt lives in
    validation/evals/decomposition.py (11/11 clean on the four-family battery, 2026-07-23)."""
    def validate(raw):
        import re

        match = re.search(r"\[[\s\S]*\]", raw or "")
        try:
            items = json.loads(match.group(0)) if match else None
        except (json.JSONDecodeError, ValueError) as error:
            raise MalformedContentError(
                f"decomposition was not valid JSON: {error}", content=raw
            ) from error
        if not isinstance(items, list) or not items:
            raise MalformedContentError(
                "decomposition must be a non-empty JSON array", content=raw
            )
        return raw

    try:
        raw = _llm_call(
            client,
            TYPED_DECOMPOSER_SYSTEM,
            f"Task: {task}",
            token_meter.ROLE_DECOMPOSER,
            validator=validate,
        )
    except MalformedContentError as error:
        raw = str(error.content or "")
    subtasks = parse_decomposition(raw, task)
    if any(s.get("type") == "unknown" for s in subtasks):
        print("[WARN] Decomposition had untypeable element(s) — those legs fall back to keyword guards "
              "(logged as `untyped`).")
    return subtasks


def generate_findings_summary(
    client: OpenAI,
    completed_subtask: str,
    final_state: dict,
    new_semantic_entries: str,
    context_policy: ContextPolicy = ContextPolicy(),
) -> str:
    """
    Comprehensive summary of everything the agent found/learned during a subtask.
    Passed to the orchestrator so all future subtask agents receive accumulated context.
    """
    if context_policy.findings_max_chars is None:
        system = load_prompt("orchestrator/findings_full")
    else:
        system = load_prompt("orchestrator/findings_compact")
    # Findings are a semantic handoff, not an artifact transport. Inspection frames and primitive
    # traces can be multi-megabyte and are already persisted separately; state gets a fixed share,
    # while semantic memory keeps its newest tail because later entries supersede older beliefs.
    task_text = str(completed_subtask)[:4_000]
    state_text = json.dumps(
        semantic_artifact_view(final_state), indent=2, default=str, ensure_ascii=False,
    )[:FINDINGS_STATE_MAX_CHARS]
    prefix = (
        f"Completed subtask: {task_text}\n\n"
        f"Final agent state:\n{state_text}\n\n"
        "New semantic memory entries learned during this subtask:\n"
    )
    semantic_budget = max(0, FINDINGS_INPUT_MAX_CHARS - len(prefix))
    semantic_text = str(new_semantic_entries or "")[-semantic_budget:]
    user = prefix + semantic_text
    findings = _llm_call(client, system, user, token_meter.ROLE_FINDINGS)
    if context_policy.findings_max_chars is not None:
        findings = findings[: context_policy.findings_max_chars]
    return findings


def _generate_findings_if_enabled(
    policy: ContextPolicy,
    client: OpenAI,
    completed_subtask: str,
    final_state: dict,
    new_semantic_entries: str,
) -> str | None:
    """Return a retained handoff, or no work at all for A3."""
    if not policy.findings_enabled:
        return None
    return generate_findings_summary(
        client,
        completed_subtask=completed_subtask,
        final_state=final_state,
        new_semantic_entries=new_semantic_entries,
        context_policy=policy,
    )
