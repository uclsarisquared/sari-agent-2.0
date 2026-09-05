"""Import-cheap, stateless VLM classifiers for completion guards.

The runtime injects its existing OpenAI-compatible client and model configuration. Calls go directly
to that client with a fresh two-message context, never through the actor's history-bearing
``send_message`` method.
"""

import json
import time

from agent_core import token_meter
from agent_core.llm import structured_chat_completion
from agent_core.prompt_loader import load_prompt


def _guard_call(client, kwargs, call_name):
    """Bill a fresh classifier request to the guard role, disabling SDK retries.

    Minimal test clients may omit with_options; the shared client is never mutated.
    """
    request_client = (client.with_options(max_retries=0)
                      if callable(getattr(client, "with_options", None)) else client)
    with token_meter.role(token_meter.ROLE_GUARD):
        extra_body = kwargs.get("extra_body") or {}
        provider = "vertex" if "google" in extra_body else "vllm"
        thinking_level = (
            extra_body.get("google", {}).get("thinking_config", {}).get("thinking_level")
        )
        structured = structured_chat_completion(
            client=request_client,
            provider=provider,
            thinking_level=thinking_level,
            default_extra_body=extra_body,
            messages=kwargs["messages"],
            schema=_SCHEMA,
            schema_name=call_name.replace(".", "_"),
            model=kwargs["model"],
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens"),
            workload="guard",
            call_name=call_name,
            timeout=kwargs.get("timeout"),
        )
        return structured.value


GUARD_SYSTEM = load_prompt("orchestrator/pickup_guard")

# Definite visual absence can complete inspection; inability to see the target cannot.
INSPECT_GUARD_SYSTEM = load_prompt("orchestrator/inspect_guard")

INSPECTION_VISIBILITY_SYSTEM = load_prompt("orchestrator/inspection_visibility")

INSPECTION_LABEL_PRESENCE_SYSTEM = load_prompt("orchestrator/inspection_label_presence")

COMPARE_GUARD_SYSTEM = load_prompt("orchestrator/compare_guard")

UNKNOWN_GUARD_SYSTEM = load_prompt("orchestrator/unknown_guard")

_SCHEMA = {
    "type": "object",
    "properties": {"match": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["match", "reason"],
    "additionalProperties": False,
}


def _guard_kwargs(model, config, messages, *, temperature=None):
    """Share request limits while allowing visibility checks to force temperature zero."""
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": getattr(config, "temperature", 0) if temperature is None else temperature,
        "max_tokens": min(getattr(config, "max_tokens", 256), 256),
        "timeout": 30,
    }
    extra_body = getattr(config, "extra_body", None)
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def _refusal(reason, latency_ms, sku):
    """Build a normalized inconclusive guard verdict without accepting a claim."""
    return {"match": False, "reason": reason, "conclusive": False,
            "latency_ms": round(latency_ms, 1), "sku": sku}


def classify_pickup(client, model, config, image_b64, held_sku, target,
                    image_media_type="image/png"):
    """Make exactly one 30-second attempt and return a normalized plain-dict verdict."""
    started = time.monotonic()
    messages = [
        {"role": "system", "content": GUARD_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}},
            {"type": "text",
             "text": f"HELD SKU: {held_sku}\nTARGET: {target}\nDoes the held SKU match the target?"},
        ]},
    ]
    kwargs = _guard_kwargs(model, config, messages)
    try:
        parsed = _guard_call(client, kwargs, "completion_guard.pickup")
        latency = (time.monotonic() - started) * 1000
        return {"match": parsed["match"], "reason": parsed["reason"].strip(),
                "conclusive": True, "latency_ms": round(latency, 1), "sku": held_sku}
    except Exception as exc:  # exhaustion and fail-fast errors retain the fail-closed contract
        latency = (time.monotonic() - started) * 1000
        return _refusal(f"VLM guard unavailable ({type(exc).__name__}: {exc})", latency, held_sku)


def classify_inspection(client, model, config, image_b64, query, answer, auxiliary_context,
                        image_media_type="image/png", evidence_frames=None):
    """Verify an answer against earlier labeled evidence followed by the current frame.

    Multi-item answers may need several views because only one held label faces the
    camera at a time. Empty evidence preserves the single-frame request.
    """
    started = time.monotonic()
    content = []
    for index, frame in enumerate(evidence_frames or [], 1):
        label = str((frame or {}).get("label") or "").strip()
        frame_b64 = str((frame or {}).get("image_b64") or "").strip()
        if not frame_b64:
            continue   # a ledger entry with no frame is context-only; never send an empty image
        content.extend([
            {"type": "text", "text": f"EVIDENCE {index}: {label or 'earlier observation'}"},
            {"type": "image_url",
             "image_url": {"url": f"data:{image_media_type};base64,{frame_b64}"}},
        ])
    if content:
        content.append({"type": "text", "text": "CURRENT FRAME:"})
    content.extend([
        {"type": "image_url",
         "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}},
        {"type": "text", "text": (
            f"QUERY: {query}\nREPORTED ANSWER: {answer}\n"
            f"AUXILIARY CONTEXT: {json.dumps(auxiliary_context, ensure_ascii=False, default=str)}\n"
            "Do the supplied images conclusively support the reported answer?"
        )},
    ])
    messages = [
        {"role": "system", "content": INSPECT_GUARD_SYSTEM},
        {"role": "user", "content": content},
    ]
    kwargs = _guard_kwargs(model, config, messages)
    try:
        parsed = _guard_call(client, kwargs, "completion_guard.inspection")
        latency = (time.monotonic() - started) * 1000
        return {"match": parsed["match"], "reason": parsed["reason"].strip(),
                "conclusive": True, "latency_ms": round(latency, 1)}
    except Exception as exc:  # timeout, API, response shape, and JSON all fail closed
        latency = (time.monotonic() - started) * 1000
        return {"match": False,
                "reason": f"VLM inspection guard unavailable ({type(exc).__name__}: {exc})",
                "conclusive": False, "latency_ms": round(latency, 1)}


def classify_inspection_visibility(client, model, config, image_b64, query,
                                   image_media_type="image/png", ocr_lines=None):
    """Fresh-context check for whether the requested printed information is currently legible."""
    started = time.monotonic()
    messages = [
        {"role": "system", "content": INSPECTION_VISIBILITY_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}},
            {"type": "text", "text": (
                f"INSPECTION REQUEST: {str(query or '').strip()}\n"
                f"PADDLEOCR AUXILIARY TEXT: "
                f"{json.dumps(list(ocr_lines or []), ensure_ascii=False)}\n"
                "Is the requested printed information directly visible and legible now?"
            )},
        ]},
    ]
    kwargs = _guard_kwargs(model, config, messages, temperature=0)
    try:
        parsed = _guard_call(client, kwargs, "completion_guard.inspection_visibility")
        latency = (time.monotonic() - started) * 1000
        return {
            "match": parsed["match"],
            "reason": parsed["reason"].strip(),
            "conclusive": True,
            "latency_ms": round(latency, 1),
        }
    except Exception as exc:
        latency = (time.monotonic() - started) * 1000
        return {
            "match": False,
            "reason": (
                f"VLM inspection visibility check unavailable "
                f"({type(exc).__name__}: {exc})"
            ),
            "conclusive": False,
            "latency_ms": round(latency, 1),
        }


def classify_inspection_label_presence(client, model, config, image_b64, query,
                                       image_media_type="image/png"):
    """Fresh-context check for a recognizable requested label, without requiring legibility."""
    started = time.monotonic()
    messages = [
        {"role": "system", "content": INSPECTION_LABEL_PRESENCE_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}},
            {"type": "text", "text": (
                f"INSPECTION REQUEST: {str(query or '').strip()}\n"
                "Is the specific requested label recognizably present on this facing side, even if "
                "its values are not yet legible?"
            )},
        ]},
    ]
    kwargs = _guard_kwargs(model, config, messages, temperature=0)
    try:
        parsed = _guard_call(client, kwargs, "completion_guard.inspection_label_presence")
        latency = (time.monotonic() - started) * 1000
        return {
            "match": parsed["match"],
            "reason": parsed["reason"].strip(),
            "conclusive": True,
            "latency_ms": round(latency, 1),
        }
    except Exception as exc:
        latency = (time.monotonic() - started) * 1000
        return {
            "match": False,
            "reason": (
                f"VLM inspection label-presence check unavailable "
                f"({type(exc).__name__}: {exc})"
            ),
            "conclusive": False,
            "latency_ms": round(latency, 1),
        }


def classify_compare(client, model, config, candidate_frames, criterion, answer,
                     auxiliary_context, image_media_type="image/png"):
    """Verify one reported choice against an ordered, labeled set of candidate frames."""
    started = time.monotonic()
    try:
        frames = list(candidate_frames or [])
        if len(frames) < 2:
            raise ValueError("at least two labeled candidate frames are required")
        content = []
        for index, frame in enumerate(frames, 1):
            if isinstance(frame, dict):
                target, image_b64 = frame.get("target"), frame.get("image_b64")
            else:
                target, image_b64 = frame
            target = str(target or "").strip()
            image_b64 = str(image_b64 or "").strip()
            if not target or not image_b64:
                raise ValueError(f"candidate {index} is missing a label or image")
            content.extend([
                {"type": "text", "text": f"CANDIDATE {index}: {target}"},
                {"type": "image_url",
                 "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}},
            ])
        content.append({"type": "text", "text": (
            f"CRITERION: {criterion}\nREPORTED CHOICE: {answer}\n"
            f"AUXILIARY CONTEXT: "
            f"{json.dumps(auxiliary_context, ensure_ascii=False, default=str)}\n"
            "Do the labeled candidate images conclusively support the reported choice?"
        )})
        messages = [
            {"role": "system", "content": COMPARE_GUARD_SYSTEM},
            {"role": "user", "content": content},
        ]
        kwargs = _guard_kwargs(model, config, messages)
        parsed = _guard_call(client, kwargs, "completion_guard.compare")
        latency = (time.monotonic() - started) * 1000
        return {"match": parsed["match"], "reason": parsed["reason"].strip(),
                "conclusive": True, "latency_ms": round(latency, 1)}
    except Exception as exc:  # timeout, API, input, response shape, and JSON all fail closed
        latency = (time.monotonic() - started) * 1000
        return {"match": False,
                "reason": f"VLM compare guard unavailable ({type(exc).__name__}: {exc})",
                "conclusive": False, "latency_ms": round(latency, 1)}


def classify_unknown(client, model, config, image_b64, task, claim, auxiliary_context,
                     image_media_type="image/png"):
    """Verify an unstructured task-completion claim against the actor-visible current frame."""
    started = time.monotonic()
    messages = [
        {"role": "system", "content": UNKNOWN_GUARD_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}},
            {"type": "text", "text": (
                f"TASK: {task}\nCOMPLETION CLAIM: {claim}\n"
                f"AUXILIARY CONTEXT: "
                f"{json.dumps(auxiliary_context, ensure_ascii=False, default=str)}\n"
                "Does the current image conclusively support completion of the task?"
            )},
        ]},
    ]
    kwargs = _guard_kwargs(model, config, messages)
    try:
        parsed = _guard_call(client, kwargs, "completion_guard.unknown")
        latency = (time.monotonic() - started) * 1000
        return {"match": parsed["match"], "reason": parsed["reason"].strip(),
                "conclusive": True, "latency_ms": round(latency, 1)}
    except Exception as exc:  # timeout, API, response shape, and JSON all fail closed
        latency = (time.monotonic() - started) * 1000
        return {"match": False,
                "reason": f"VLM unknown guard unavailable ({type(exc).__name__}: {exc})",
                "conclusive": False, "latency_ms": round(latency, 1)}


def make_inspect_guard(client, model, config, image_b64, on_verdict=None, evidence_frames=None):
    """Return an image-bound, per-step cached callback matching ``predicate_inspect``'s contract.

    ``evidence_frames`` is the leg's accumulated inspection ledger (see ``classify_inspection``); it
    is fixed for the lifetime of this per-step closure, so the cache key need not cover it.
    """
    frames = [
        {"label": str((frame or {}).get("label") or ""),
         "image_b64": str((frame or {}).get("image_b64") or "")}
        for frame in (evidence_frames or [])
    ]
    cache = {}

    def guard(query, answer, auxiliary_context):
        """Evaluate inspection completion once per distinct claim for this frame."""
        try:
            aux_key = json.dumps(auxiliary_context, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:  # defensive; classify_inspection still receives the original context
            aux_key = repr(auxiliary_context)
        key = (str(query), str(answer), aux_key)
        reused = key in cache
        if reused:
            cached = cache[key]
            verdict = dict(cached) if isinstance(cached, dict) else cached
        else:
            verdict = classify_inspection(
                client, model, config, image_b64, query, answer, auxiliary_context,
                evidence_frames=frames)
            cache[key] = dict(verdict) if isinstance(verdict, dict) else verdict
            guard.call_count += 1
        if callable(on_verdict):
            on_verdict(query, auxiliary_context, verdict, reused)
        return verdict

    guard.call_count = 0
    return guard


def make_compare_guard(client, model, config, candidate_frames, on_verdict=None):
    """Return a candidate-frame-bound cached callback for ``predicate_compare``."""
    frames = tuple(
        (str(frame.get("target") or ""), str(frame.get("image_b64") or ""))
        if isinstance(frame, dict) else (str(frame[0]), str(frame[1]))
        for frame in (candidate_frames or [])
    )
    cache = {}

    def guard(criterion, answer, auxiliary_context):
        """Evaluate comparison completion once per distinct claim across cached frames."""
        try:
            aux_key = json.dumps(auxiliary_context, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            aux_key = repr(auxiliary_context)
        key = (str(criterion), str(answer), aux_key)
        reused = key in cache
        if reused:
            verdict = dict(cache[key])
        else:
            verdict = classify_compare(
                client, model, config, frames, criterion, answer, auxiliary_context)
            cache[key] = dict(verdict)
            guard.call_count += 1
        if callable(on_verdict):
            on_verdict(criterion, auxiliary_context, verdict, reused)
        return verdict

    guard.call_count = 0
    return guard


def make_unknown_guard(client, model, config, image_b64, on_verdict=None):
    """Return an image-bound, per-step cached callback for ``predicate_unknown``."""
    cache = {}

    def guard(task, claim, auxiliary_context):
        """Evaluate an unknown-task completion claim once for the current frame."""
        try:
            aux_key = json.dumps(auxiliary_context, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            aux_key = repr(auxiliary_context)
        key = (str(task), str(claim), aux_key)
        reused = key in cache
        if reused:
            verdict = dict(cache[key])
        else:
            verdict = classify_unknown(
                client, model, config, image_b64, task, claim, auxiliary_context)
            cache[key] = dict(verdict)
            guard.call_count += 1
        if callable(on_verdict):
            on_verdict(task, auxiliary_context, verdict, reused)
        return verdict

    guard.call_count = 0
    return guard


def cache_compare_candidate_frames(cache, targets, candidate_sets, nearest_checkpoint,
                                   image_b64, step):
    """Capture the first frame seen at each candidate's resolved checkpoint.

    Candidate identity is positional: two targets may resolve to the same checkpoint and still
    receive separately labeled evidence entries. Malformed/unresolved sets simply capture nothing,
    leaving the VLM path to fail closed when no complete guard can be built.
    """
    if not isinstance(cache, dict) or not isinstance(candidate_sets, list):
        return 0
    targets = list(targets or [])
    if len(targets) != len(candidate_sets):
        return 0
    captured = 0
    for index, checkpoints in enumerate(candidate_sets):
        if (index in cache or not checkpoints
                or nearest_checkpoint not in set(checkpoints)):
            continue
        cache[index] = {
            "target": targets[index],
            "image_b64": image_b64,
            "checkpoint": nearest_checkpoint,
            "step": step,
        }
        captured += 1
    return captured


def evaluate_hands(client, model, config, image_b64, target, held_skus):
    """Classify each unique SKU once; return ``(per_hand_verdicts, call_count)``."""
    per_hand, by_sku, calls = {}, {}, 0
    for side in ("left", "right"):
        sku = (held_skus or {}).get(side)
        if not sku:
            continue
        if sku in by_sku:
            verdict = dict(by_sku[sku])
            verdict["reused"] = True
        else:
            verdict = classify_pickup(client, model, config, image_b64, sku, target)
            verdict["reused"] = False
            by_sku[sku] = dict(verdict)
            calls += 1
        per_hand[side] = verdict
    return per_hand, calls
