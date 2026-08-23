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

    first = endpoint.create(messages=[{"role": "user", "content": "Remember the word sari."}])
    vision = endpoint.create(
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "What food is visible?"},
            {"type": "image_url", "image_url": {"url": PUBLIC_IMAGE}},
        ]}], schema=SCHEMA, schema_name="vision_answer",
    )
    history = endpoint.create(messages=[
        {"role": "user", "content": "Remember the word sari."},
        {"role": "assistant", "content": first.choices[0].message.content},
        {"role": "user", "content": "Which word did I ask you to remember?"},
    ])
    usage = endpoint.envelope(vision).get("usage")
    assert vision.choices[0].message.content and usage, "missing structured content or usage"
    assert "sari" in (history.choices[0].message.content or "").lower(), "history was not retained"
    print(f"PASS: model={endpoint.profile.model} usage={usage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
