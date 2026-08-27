"""Run sequential OCR service calls and report end-to-end and inference timing."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image


def _png_bytes(path: Path) -> bytes:
    with Image.open(path) as opened:
        opened.load()
        buffer = BytesIO()
        opened.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _json_response(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected an object response, got {payload!r}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark sequential OCR HTTP calls.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--ocr-url", default="http://127.0.0.1:9100")
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if args.calls < 1:
        parser.error("--calls must be at least 1")

    base_url = args.ocr_url.rstrip("/")
    health = _json_response(requests.get(f"{base_url}/health", timeout=5))
    body = _png_bytes(args.image)
    records: list[dict[str, Any]] = []
    expected_lines: list[str] | None = None
    for call in range(1, args.calls + 1):
        started = time.perf_counter()
        payload = _json_response(
            requests.post(
                f"{base_url}/v1/ocr",
                data=body,
                headers={"Content-Type": "image/png"},
                timeout=args.timeout,
            )
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        lines = payload.get("lines")
        inference_ms = payload.get("elapsed_ms")
        if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
            raise RuntimeError(f"call {call} returned invalid lines: {payload!r}")
        if not isinstance(inference_ms, (int, float)) or isinstance(inference_ms, bool):
            raise RuntimeError(f"call {call} returned invalid elapsed_ms: {payload!r}")
        if expected_lines is None:
            expected_lines = lines
        elif lines != expected_lines:
            raise RuntimeError(f"call {call} returned inconsistent OCR text")
        record = {
            "call": call,
            "wall_ms": round(wall_ms, 2),
            "inference_ms": round(float(inference_ms), 2),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    result = {
        "health": health,
        "image": str(args.image),
        "calls": args.calls,
        "average_wall_ms": round(statistics.mean(r["wall_ms"] for r in records), 2),
        "average_inference_ms": round(
            statistics.mean(r["inference_ms"] for r in records), 2
        ),
        "records": records,
        "lines": expected_lines,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
