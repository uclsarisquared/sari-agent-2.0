"""OpenAI-compatible endpoint annotation implementation (legacy module name).

It has the same `annotate()` interface as annotate_claude_cli, so
annotate_pass can pick either with one `--backend` flag.

This is the batch counterpart to annotate_probe.py (which is a one-image CLI probe): it exposes a
reusable `annotate(image_path, system, schema, ...) -> (result_dict, envelope)` with the SAME
signature annotate_claude_cli.annotate has, using the shared SDK transport:

  * response_format (json_schema, strict), NOT guided_json - this vLLM build silently ignores the
    older spelling (negative-control "banana" test, see annotate_probe / annotator_sys_inst).
  * chat_template_kwargs={"enable_thinking": False} - Qwen3.x otherwise burns the whole token
    budget reasoning, loops, and returns content=None.
  * Each image is preceded by a text label part, per the annotator caller contract - the primary
    is the STANDING view, extra_views are the same shelf at a lower camera height (CROUCHED), every one
    item-bearing.

The annotator was un-pinned from claude on 2026-07-20 (user directive) - claude-cli stays the
DEFAULT because annotation quality was measured/frozen on sonnet, but endpoint is selectable.

    python mapping/annotate/annotate_endpoint.py mapping/output/captures/cp067_primary.png
"""
import argparse
import json
import mimetypes
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/annotate
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

# Reused verbatim so the batch path and the probe path share ONE set of measured server facts
# (data-URI block shape, OPENAI_API_URL -> /v1, OPENAI_API_KEY bearer resolution). These
# also trigger annotate_probe's module-level load_dotenv(repo-root secrets.env), so creds are present.
from agent_core.llm import (  # noqa: E402
    ChatEndpoint, EndpointProfile, image_url_part,
)
from agent_core.models import annotator_model  # noqa: E402
from annotator_sys_inst import (  # noqa: E402
    SYS_INST_CLASSIFY, CLASSIFY_SCHEMA,
    build_annotation_instructions, schema_for, effective_kind,
)

DEFAULT_MODEL = annotator_model()  # $OPENAI_ANNOTATOR_MODEL / $OPENAI_MODEL in secrets.env
DEFAULT_TIMEOUT_S = 300.0
_MAX_TOKENS = 4096


class EndpointAnnotateError(RuntimeError):
    """An endpoint annotation failed (API error, empty content, or unparseable JSON). Mirrors
    annotate_claude_cli.ClaudeCliError's role: annotate_pass catches both so one failed checkpoint
    is skipped and --resume continues, instead of killing the whole pass."""


def _image_part(label, path, model):
    with open(path, "rb") as f:
        raw = f.read()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return [
        {"type": "text", "text": f"{label} view:"},
        image_url_part(raw, mime),
    ]


def annotate(image_path, system, schema, *, model=DEFAULT_MODEL, effort=None,
             timeout=DEFAULT_TIMEOUT_S, extra_views=None, base_url=None, api_key=None,
             think=False, temperature=0.0, max_tokens=_MAX_TOKENS):
    """Run one annotation through the qwen/vLLM server and return (result_dict, envelope).

    Signature matches annotate_claude_cli.annotate so annotate_pass can call either interchangeably.
    `effort` is accepted and ignored (qwen has no effort knob; kept so the caller need not special-
    case the two backends). `extra_views` is [(label, path)] of additional same-spot views - every
    one item-bearing, labelled, per the annotator caller contract.

    The envelope mirrors the claude one's shape where it matters: `total_cost_usd` is None because
    this repository does not calculate endpoint pricing, and `usage` carries token counts.
    """
    profile = EndpointProfile.from_env(model=model, base_url=base_url, api_key=api_key)
    transport = ChatEndpoint(profile, timeout=timeout)
    model = profile.model

    content = _image_part("STANDING", image_path, model)
    for label, path in (extra_views or []):
        content += _image_part(label, path, model)

    messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
    extra = ({"chat_template_kwargs": {"enable_thinking": think}}
             if profile.provider == "vllm" else None)

    t0 = time.time()
    try:
        structured = transport.create_structured(
            messages=messages, schema=schema, schema_name="annotation", model=model,
            max_tokens=max_tokens, temperature=temperature, extra_body=extra,
            workload="annotation", call_name="annotation",
        )
    except Exception as error:
        raise EndpointAnnotateError(f"endpoint annotation failed: {error}") from error
    dt = time.time() - t0
    response = structured.completion.raw_response
    resp = transport.envelope(response)
    choice = response.choices[0]
    # finish_reason is the whole diagnostic when the JSON won't parse - "length" (truncated,
    # never terminated) needs a completely different response than "stop" (model finished but
    # emitted malformed output). The old error threw it away and reported every failure as "not
    # valid JSON", which read as a formatting bug when it was really a max_tokens truncation.
    finish = choice.finish_reason
    usage = resp.get("usage", {})
    result = structured.value

    envelope = {
        "backend": "endpoint",
        "provider": profile.provider,
        "model": model,
        "total_cost_usd": None,   # pricing is provider-dependent and deliberately not estimated
        "duration_ms": dt * 1000.0,
        "finish_reason": finish,
        "usage": usage,
        "structured_output_enforcement": structured.enforcement,
    }
    return result, envelope


def _build_request(args):
    if args.classify:
        return SYS_INST_CLASSIFY, CLASSIFY_SCHEMA, "classify"
    kind = "non_shelf" if args.kind == "non_shelf" else effective_kind(args.kind)
    return build_annotation_instructions(kind), schema_for(kind), f"annotate:{kind}"


def main():
    p = argparse.ArgumentParser(description="Annotate one capture via the configured endpoint.")
    p.add_argument("image", help="STANDING view (a PNG from the capture walk)")
    p.add_argument("--crouch", default=None, help="CROUCHED view of the same shelf (cp<id>_crouch.png)")
    p.add_argument("--classify", action="store_true", help="Run Stage 1 (shelf/non_shelf) instead of Stage 2")
    p.add_argument("--kind", default="shelf", help="Stage-2 effective kind: shelf | junction | end | doorway | non_shelf")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=None, help="Default: $OPENAI_API_URL, +/v1")
    p.add_argument("--api-key", default=None, help="Bearer (default: $OPENAI_API_KEY)")
    p.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = p.parse_args()

    system, schema, label = _build_request(args)
    views = [("CROUCHED", args.crouch)] if args.crouch else None
    print(f"[endpoint] {label}  model={args.model}  {os.path.basename(args.image)}")
    try:
        result, env = annotate(args.image, system, schema, model=args.model, timeout=args.timeout,
                               extra_views=views, base_url=args.base_url, api_key=args.api_key,
                               think=args.think)
    except EndpointAnnotateError as e:
        print(f"[endpoint] FAILED: {e}")
        sys.exit(1)
    u = env.get("usage", {})
    print(f"[endpoint] provider={env['provider']} {env['duration_ms']/1000:.1f}s  "
          f"prompt={u.get('prompt_tokens','?')} "
          f"completion={u.get('completion_tokens','?')}")
    print("-" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# Compatibility for imports from older scripts and persisted backend configurations.
QwenAnnotateError = EndpointAnnotateError
