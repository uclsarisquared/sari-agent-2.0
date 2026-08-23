"""Process-wide token accounting for every LLM call the agent makes, split by which reasoner made it.

WHY A PATCH AND NOT PER-CALL-SITE COUNTERS: the token cost of one benchmark attempt is spread over
call sites in agent_core.agent (actor + semantic/episodic learner + advisor), vision.perception
(bbox/centering/OCR reasoning), vision.md_tools' qwen fallback, orchestrator.subtask_agents
(decomposer + findings summary + final responder), orchestrator.pickup_vlm_guard's completion guards,
mapping.drivers.vlm_planner and the map resolver. Most go through the OpenAI SDK's
``chat.completions.create`` against the same OpenAI-compatible endpoint. The response wrapper counts
token-bearing completions; a separate transport wrapper counts every actual HTTP send, including
failed requests and retries performed inside either the SDK or ``BaseAgent._api_call_with_retry``.

WHY ROLES ON TOP OF THAT: the totals alone answer "what did this attempt cost", never "what did the
advisor cost" - so an ablation that removes a component cannot tell from them whether the component
it cut was the expensive one. ``by_model`` does not help: every reasoner here runs on the same
Qwen checkpoint. So each call site declares itself with the ``role`` context manager and ``_record``
attributes the call to whatever role is current on this thread. A call made outside any ``role``
block is counted under ``UNATTRIBUTED`` rather than dropped or guessed at, so the role rows always
re-total to the whole and a new, untagged call site shows up as a visible gap instead of quietly
inflating somebody else's share.

The context variable is per-thread by construction (a fresh thread starts with the default), which
is the right failure mode: a background thread's call reads as unattributed, never as whatever role
the main thread happened to be inside at the time.

The runtime resolver/advisor/verifier, endpoint annotator, probe, and mapping planner all use the
SDK transport, so the patch is their single accounting path. ``claude_json`` (the claude-cli
backend) is billed to a different account altogether and is not counted here.

NOT COUNTED, deliberately: moondream (vision/md_tools) is a different API that reports no token
usage at all, so there is nothing to add up; its qwen ERROR-fallback path does go through the SDK
and is counted. Streamed responses carry no ``usage`` either and land in ``untracked_calls`` rather
than being guessed at - the repo streams nowhere today, so that counter should stay 0.

Usage (see run_agent.py):

    from agent_core import token_meter
    token_meter.install(run_dir)      # once, before any LLM traffic
    with token_meter.role(token_meter.ROLE_ACTOR):
        ...                           # every SDK call in here is billed to the actor
    before = token_meter.snapshot()
    ...
    token_meter.delta(before)         # adds api_calls beside token-bearing calls
    token_meter.dump()                # tokens.json, also auto-dumped every DUMP_INTERVAL_S

``install`` is idempotent, so importing this from two places cannot double-count.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import threading
import time
from typing import Any, Iterator, Optional

# tokens.json is rewritten at most this often from inside the patch. The file exists so an attempt
# that is SIGKILLed (harness timeout, operator kill) still leaves its token cost behind - summary.json
# is only written at exit, and those attempts never reach it. Cheap: one small atomic write.
DUMP_INTERVAL_S = 10.0

TOKENS_FILE = "tokens.json"

# The reasoners, one name per component an ablation can remove. Keep these stable: they are column
# names in the benchmark's CSV export and row labels in the watch dashboard, so renaming one
# silently splits a battery's history in two.
ROLE_ACTOR = "actor"                # VLMAgent.send_message - the per-step action chooser
ROLE_SEMANTIC = "semantic"          # associative learner, semantic memory pass
ROLE_EPISODIC = "episodic"          # associative learner, episodic memory pass
ROLE_ADVISOR = "advisor"            # graph-advised navigator's per-hop pick
ROLE_PERCEPTION = "perception"      # bbox / centering / OCR-bbox reasoning, incl. md_tools' fallback
ROLE_GUARD = "guard"                # pickup_vlm_guard's completion classifiers
ROLE_DECOMPOSER = "decomposer"      # task -> typed subtasks
ROLE_FINDINGS = "findings"          # between-leg findings summaries
ROLE_RESOLVER = "resolver"          # locate_task.resolve - plan-time map/product resolution
ROLE_RESPONDER = "responder"        # task journal -> final user-facing response
UNATTRIBUTED = "unattributed"       # a call made outside any role block; see the module docstring

ROLES = (
    ROLE_ACTOR, ROLE_SEMANTIC, ROLE_EPISODIC, ROLE_ADVISOR, ROLE_PERCEPTION,
    ROLE_GUARD, ROLE_DECOMPOSER, ROLE_FINDINGS, ROLE_RESOLVER, ROLE_RESPONDER, UNATTRIBUTED,
)

_lock = threading.Lock()
_installed = False
_run_dir: Optional[str] = None
_last_dump = 0.0

# Per-thread, not per-process: see the module docstring on why an untagged thread must read as
# unattributed rather than inherit whatever the main thread was doing.
_role_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "sari_token_meter_role", default=UNATTRIBUTED)

_totals: dict[str, Any] = {
    "tokens_in": 0,          # prompt tokens billed to the server
    "tokens_out": 0,         # completion tokens (reasoning tokens included, per the server's count)
    "calls": 0,              # SDK calls that reported usage
    "api_calls": 0,          # actual OpenAI-compatible HTTP attempts, including failures/retries
    "untracked_calls": 0,    # SDK calls with no usage on the response (streaming); see module docstring
    "by_model": {},          # model_id -> {"tokens_in", "tokens_out", "calls"}
    "by_role": {},           # role -> token-bearing calls plus optional api_calls
}


@contextlib.contextmanager
def role(name: str) -> Iterator[None]:
    """Bills every LLM call made inside the block to ``name``.

    Nests: the innermost block wins, which is what the call graph wants - perception called from
    inside an actor step is perception's cost, not the actor's. Resets on the way out even if the
    call raised, so a failed call cannot leak its role onto the next one.
    """
    token = _role_var.set(name or UNATTRIBUTED)
    try:
        yield
    finally:
        _role_var.reset(token)


def current_role() -> str:
    """The role calls would be billed to right now. Exposed for tests and debugging."""
    return _role_var.get()


def install(run_dir: Optional[str] = None) -> None:
    """Patches the OpenAI SDK's completion and transport paths. Safe to call more than once."""
    global _installed, _run_dir, _totals, _role_var

    if run_dir:
        _run_dir = run_dir

    if _installed:
        return

    try:
        from openai import _base_client
        from openai.resources.chat import completions as _completions
    except ImportError:  # noqa: BLE001 - counting is bookkeeping, never a reason to fail a run
        return

    original_create = _completions.Completions.create
    if getattr(original_create, "_sari_token_meter", False):
        # Already patched by another copy of this module (subtask_agents imports it as
        # `agent_core.token_meter`, a different sys.path root would import it again) - wrapping the
        # wrapper would double-count every call. Adopt the counters that copy is already filling,
        # AND the context variable it reads roles from, so reading the totals - or opening a `role`
        # block - through EITHER copy sees the same run. Adopting the counters without the var
        # would send this copy's roles nowhere and file all its calls as unattributed.
        _totals = original_create._sari_totals
        _totals.setdefault("api_calls", 0)
        _totals.setdefault("by_role", {})  # the installing copy may predate role accounting
        _role_var = getattr(original_create, "_sari_role_var", _role_var)
        _installed = True
        return

    def _patch_transport(client_class, *, asynchronous: bool) -> None:
        original_send = client_class.send
        if getattr(original_send, "_sari_api_call_meter", False):
            return

        if asynchronous:
            async def send(self, request, *args, **kwargs):
                _record_api_call()
                return await original_send(self, request, *args, **kwargs)
        else:
            def send(self, request, *args, **kwargs):
                _record_api_call()
                return original_send(self, request, *args, **kwargs)

        send.__wrapped__ = original_send
        send._sari_api_call_meter = True
        client_class.send = send

    # OpenAI's default clients inherit httpx.send without overriding it. Patching these SDK-owned
    # wrappers (rather than httpx globally) counts every retry send without touching OCR, depth,
    # Moondream, Discord, or any other HTTP traffic in this process.
    _patch_transport(_base_client.SyncHttpxClientWrapper, asynchronous=False)
    _patch_transport(_base_client.AsyncHttpxClientWrapper, asynchronous=True)

    def create(self, *args, **kwargs):
        response = original_create(self, *args, **kwargs)
        try:
            _record(kwargs.get("model"), getattr(response, "usage", None))
        except Exception:  # noqa: BLE001 - never let accounting break a call that succeeded
            pass
        return response

    create.__wrapped__ = original_create
    create._sari_token_meter = True  # the marker the double-install guard above reads
    create._sari_totals = _totals    # ...and the counters it adopts
    create._sari_role_var = _role_var  # ...and the role var it must share to attribute at all
    _completions.Completions.create = create
    _installed = True


def record_external(model: Any, usage: Any, role_name: Optional[str] = None) -> None:
    """Counts one token-bearing response that did NOT go through the patched SDK.

    Retained for unrelated legacy callers that do not use the OpenAI SDK. ``role_name`` overrides
    the ambient role for callers that know which reasoner they are serving but cannot wrap the call.

    The caller separately invokes ``record_api_call`` before its raw send. Never call this for a
    response that also went through the SDK - that double-counts its token-bearing ``calls`` row.
    """
    try:
        _record(model, usage, role_name=role_name)
    except Exception:  # noqa: BLE001 - accounting never breaks a call that succeeded
        pass


def record_api_call(role_name: Optional[str] = None) -> None:
    """Counts one raw-HTTP OpenAI-compatible request immediately before its transport send."""
    try:
        _record_api_call(role_name=role_name)
    except Exception:  # noqa: BLE001 - accounting never prevents the request itself
        pass


def _record_api_call(role_name: Optional[str] = None) -> None:
    """Internal request-attempt counter shared by the SDK and raw-HTTP paths."""
    global _last_dump
    billed_to = str(role_name or _role_var.get() or UNATTRIBUTED)
    with _lock:
        _totals["api_calls"] = int(_totals.get("api_calls") or 0) + 1
        row = _totals["by_role"].setdefault(
            billed_to, {"tokens_in": 0, "tokens_out": 0, "calls": 0, "api_calls": 0})
        row["api_calls"] = int(row.get("api_calls") or 0) + 1
        due = time.monotonic() - _last_dump >= DUMP_INTERVAL_S
        totals = _copy_locked()

    if due and _run_dir:
        _last_dump = time.monotonic()
        _write(totals)


def _record(model: Any, usage: Any, role_name: Optional[str] = None) -> None:
    global _last_dump

    # Read outside the lock: it is a per-thread lookup and taking it under the lock would pin the
    # role to whoever happens to be waiting.
    billed_to = str(role_name or _role_var.get() or UNATTRIBUTED)

    with _lock:
        if usage is None:
            _totals["untracked_calls"] += 1
            return

        # The vLLM server speaks the OpenAI schema, so usage is an object; a dict shows up when a
        # response has been round-tripped through JSON (annotate_qwen's envelope replay).
        if isinstance(usage, dict):
            tokens_in = usage.get("prompt_tokens") or 0
            tokens_out = usage.get("completion_tokens") or 0
        else:
            tokens_in = getattr(usage, "prompt_tokens", 0) or 0
            tokens_out = getattr(usage, "completion_tokens", 0) or 0

        _totals["tokens_in"] += int(tokens_in)
        _totals["tokens_out"] += int(tokens_out)
        _totals["calls"] += 1

        for bucket, key in (("by_model", str(model or "unknown")), ("by_role", billed_to)):
            row = _totals[bucket].setdefault(key, {"tokens_in": 0, "tokens_out": 0, "calls": 0})
            row["tokens_in"] += int(tokens_in)
            row["tokens_out"] += int(tokens_out)
            row["calls"] += 1

        due = time.monotonic() - _last_dump >= DUMP_INTERVAL_S
        totals = _copy_locked()

    if due and _run_dir:
        _last_dump = time.monotonic()
        _write(totals)


def _copy_locked() -> dict[str, Any]:
    """A deep-enough copy of the totals. Caller holds the lock."""
    snapshot = dict(_totals)
    for bucket in ("by_model", "by_role"):
        snapshot[bucket] = {key: dict(row) for key, row in _totals[bucket].items()}
    return snapshot


def snapshot() -> dict[str, Any]:
    """Totals so far. Take one before a leg and pass it to ``delta`` after."""
    with _lock:
        return _copy_locked()


def delta(before: dict[str, Any]) -> dict[str, Any]:
    """How many tokens were spent since ``before`` was taken, whole and per role.

    ``by_role`` holds only the roles that actually spent something in the window - a leg that never
    reached the advisor should have no advisor row, not a row of zeroes, so a reader can tell "did
    not run" from "ran and cost nothing".
    """
    now = snapshot()
    before_roles = before.get("by_role") or {}
    by_role: dict[str, dict[str, int]] = {}
    for name, row in now["by_role"].items():
        was = before_roles.get(name) or {}
        spent = {field: int(row.get(field, 0)) - int(was.get(field, 0))
                 for field in ("tokens_in", "tokens_out", "calls", "api_calls")}
        if any(spent.values()):
            by_role[name] = spent
    return {
        "tokens_in": now["tokens_in"] - int(before.get("tokens_in", 0)),
        "tokens_out": now["tokens_out"] - int(before.get("tokens_out", 0)),
        "calls": now["calls"] - int(before.get("calls", 0)),
        "api_calls": now["api_calls"] - int(before.get("api_calls", 0)),
        "by_role": by_role,
    }


def totals() -> dict[str, Any]:
    """Totals with the derived ``tokens_total``, shaped for summary.json."""
    payload = snapshot()
    payload["tokens_total"] = payload["tokens_in"] + payload["tokens_out"]
    return payload


def dump(run_dir: Optional[str] = None) -> None:
    """Writes tokens.json now. No-op until a run dir is known."""
    global _run_dir, _last_dump
    if run_dir:
        _run_dir = run_dir
    if not _run_dir:
        return
    _last_dump = time.monotonic()
    _write(totals())


def _write(payload: dict[str, Any]) -> None:
    """Atomic so the benchmark runner, which may read this while the agent is still running, never
    sees a half-written file."""
    if not _run_dir:
        return
    payload = dict(payload)
    payload.setdefault("tokens_total", payload["tokens_in"] + payload["tokens_out"])
    path = os.path.join(_run_dir, TOKENS_FILE)
    temp = path + ".tmp"
    try:
        os.makedirs(_run_dir, exist_ok=True)
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(temp, path)
    except OSError:
        pass  # a run dir that cannot be written to is not worth killing the attempt over


def reset() -> None:
    """Zeroes the counters (tests only - a real process meters one run from start to finish)."""
    with _lock:
        _totals["tokens_in"] = 0
        _totals["tokens_out"] = 0
        _totals["calls"] = 0
        _totals["api_calls"] = 0
        _totals["untracked_calls"] = 0
        _totals["by_model"] = {}
        _totals["by_role"] = {}
