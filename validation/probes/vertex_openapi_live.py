"""Opt-in live smoke probe for Vertex's OpenAI-compatible Chat Completions endpoint.

Run with ``RUN_LIVE_VERTEX_PROBE=1 LLM_PROVIDER=vertex`` plus ADC/project configuration.
Without that explicit opt-in or configuration this script exits successfully with a skip message.
"""

import os
from pathlib import Path
import sys

_AGENT_DIR = Path(__file__).resolve().parents[2] / "agent"
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from agent_core.llm import ChatEndpoint, EndpointConfigurationError, EndpointProfile
from nav.locate_task import RESOLVE_SCHEMA
from annotator_sys_inst import SHELF_ANNOTATION_SCHEMA
from orchestrator.pickup_vlm_guard import _SCHEMA as GUARD_SCHEMA


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
    usage = endpoint.envelope(vision.completion.raw_response).get("usage")
    assert vision.completion.text and usage, "missing structured content or usage"
    assert "sari" in (history.choices[0].message.content or "").lower(), "history was not retained"
    print(f"PASS: model={endpoint.profile.model} usage={usage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
