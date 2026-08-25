"""Opt-in live smoke probe for Vertex's OpenAI-compatible Chat Completions endpoint.

Run with ``RUN_LIVE_VERTEX_PROBE=1 LLM_PROVIDER=vertex`` plus ADC/project configuration.
Without that explicit opt-in or configuration this script exits successfully with a skip message.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

_AGENT_DIR = Path(__file__).resolve().parents[2] / "agent"
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from agent_core.llm import (
    ChatEndpoint, EndpointConfigurationError, EndpointProfile, image_url_part,
)
from nav.locate_task import RESOLVE_SCHEMA
from annotator_sys_inst import SHELF_ANNOTATION_SCHEMA
from orchestrator.pickup_vlm_guard import _SCHEMA as GUARD_SCHEMA
from vision.perception import _bbox_schema


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
PUBLIC_IMAGE = "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/scones.jpg"


def main() -> int:
    if os.getenv("RUN_LIVE_VERTEX_PROBE") != "1":
        print("SKIP: set RUN_LIVE_VERTEX_PROBE=1 to enable the live Vertex probe")
        return 0
    if os.getenv("LLM_PROVIDER") != "vertex" or not os.getenv("GOOGLE_CLOUD_PROJECT"):
        print("SKIP: set LLM_PROVIDER=vertex and GOOGLE_CLOUD_PROJECT")
        return 0
    try:
        endpoint = ChatEndpoint(EndpointProfile.from_env(), timeout=120)
    except EndpointConfigurationError as error:
        print(f"SKIP: ADC is unavailable: {error}")
        return 0

    first = endpoint.create_result(
        messages=[{"role": "user", "content": "Remember the word sari."}],
        max_tokens=256, workload="reasoning",
    )
    # Force the already-initialized bearer down the expiry path so this probe covers an actual ADC
    # refresh as well as ordinary token reuse. This process is disposable; no credential file or
    # external state is changed.
    bearer = endpoint.profile.api_key
    credentials = getattr(bearer, "_credentials", None)
    if credentials is None:
        raise AssertionError("Vertex profile did not expose the shared ADC bearer provider")
    credentials.expiry = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert bearer(), "forced ADC refresh returned no access token"
    vision = endpoint.create_structured(
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "What food is visible?"},
            {"type": "image_url", "image_url": {"url": PUBLIC_IMAGE}},
        ]}], schema=SCHEMA, schema_name="vision_answer", workload="localization",
    )
    history = endpoint.create(messages=[
        {"role": "user", "content": "Remember the word sari."},
        first.assistant_message,
        {"role": "user", "content": "Which word did I ask you to remember?"},
    ], max_tokens=256, workload="reasoning")
    representative = [
        ("guard", GUARD_SCHEMA, "Does a red soda can match a red soda can?"),
        ("resolver", RESOLVE_SCHEMA, "Resolve chips to candidate checkpoint 1."),
        ("annotation", SHELF_ANNOTATION_SCHEMA, "Describe a shelf with one snack item."),
    ]
    for name, schema, prompt in representative:
        endpoint.create_structured(
            messages=[{"role": "user", "content": prompt}], schema=schema,
            schema_name=f"probe_{name}",
            workload="guard" if name == "guard" else (
                "annotation" if name == "annotation" else "reasoning"
            ),
        )
    probe_frames = [
        (_AGENT_DIR.parent / "validation/evidence/hand_pose/0722_192358/005_grab_piattos.png",
         "the visible Piattos bag"),
        (_AGENT_DIR.parent / "validation/evidence/carry/0722_195705/route1_leg1_cp3.png",
         "a visible packaged store product"),
    ]
    for frame, target in probe_frames:
        image = image_url_part(frame.read_bytes(), "image/png")
        replies = []
        for repeat in range(3):
            bbox = endpoint.create_structured(
                messages=[{"role": "user", "content": [image, {"type": "text", "text": (
                    f"Detect {target}. Return every matching box in normalized 0-1000 "
                    "Gemini [ymin,xmin,ymax,xmax] order."
                )}]}],
                schema=_bbox_schema(12), schema_name=f"probe_bbox_{frame.stem}_{repeat}",
                workload="localization", temperature=0.0,
            )
            replies.append(bbox.value)
        assert replies[0], f"no bbox found in representative frame {frame}"
        assert replies.count(replies[0]) == len(replies), f"unstable bbox replies for {frame}"
    usage = endpoint.envelope(vision.completion.raw_response).get("usage")
    assert vision.completion.text and usage, "missing structured content or usage"
    assert "sari" in (history.choices[0].message.content or "").lower(), "history was not retained"
    print(f"PASS: model={endpoint.profile.model} usage={usage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
