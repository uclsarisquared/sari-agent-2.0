"""Shared removal of binary and implementation-only payloads from text artifacts."""

from __future__ import annotations

from typing import Any


_DROP_KEYS = {
    "steps",
    "primitive_steps",
    "debug",
    "debug_payload",
    "raw_debug",
}


def semantic_artifact_view(value: Any) -> Any:
    """Copy JSON-like data without images, primitive traces, or debug payloads."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if (
                normalized in _DROP_KEYS
                or normalized.endswith("_b64")
                or normalized in {"base64", "image", "image_bytes", "screenshot"}
            ):
                continue
            cleaned[key] = semantic_artifact_view(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [semantic_artifact_view(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "<binary omitted>"
    if isinstance(value, str) and value.lstrip().lower().startswith("data:image/"):
        return "<image omitted>"
    return value
