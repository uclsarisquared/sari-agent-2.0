"""Benchmark PaddleOCR's ONNX engine with CPU or DirectML.

The script deliberately verifies every ONNX Runtime session after PaddleOCR
initialization.  This catches the otherwise easy-to-miss case where an
accelerator provider is installed but ONNX Runtime silently falls back to CPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image
from paddleocr import PaddleOCR


def _parse_lines(result: Any) -> list[str]:
    if not result:
        return []
    page = result[0]
    if isinstance(page, dict):
        texts = page.get("rec_texts") or []
    else:
        data = getattr(page, "json", None)
        if callable(data):
            data = data()
        if isinstance(data, dict):
            data = data.get("res", data)
            texts = data.get("rec_texts") or []
        else:
            texts = []
    return [text.strip() for text in texts if isinstance(text, str) and text.strip()]


def _sessions() -> list[ort.InferenceSession]:
    return [obj for obj in gc.get_objects() if isinstance(obj, ort.InferenceSession)]


def _engine(provider: str) -> tuple[PaddleOCR, dict[str, Any]]:
    if provider == "cpu":
        required = "CPUExecutionProvider"
        engine_config: dict[str, Any] = {"providers": [required]}
    else:
        required = "DmlExecutionProvider"
        engine_config = {
            "providers": [required],
            "provider_options": {"device_id": 0},
            # Required by DirectML's execution provider.
            "execution_mode": "sequential",
            "enable_mem_pattern": False,
        }

    started = time.perf_counter()
    engine = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        # PaddleOCR 3.7's ONNX wrapper only recognizes CUDA when device is
        # "gpu".  Explicit providers still control the actual ORT device.
        device="cpu",
        engine="onnxruntime",
        engine_config=engine_config,
    )
    init_seconds = time.perf_counter() - started

    sessions = _sessions()
    provider_lists = [session.get_providers() for session in sessions]
    if not sessions:
        raise RuntimeError("PaddleOCR initialized without discoverable ONNX Runtime sessions")
    rejected = [providers for providers in provider_lists if not providers or providers[0] != required]
    if rejected:
        raise RuntimeError(
            f"provider verification failed: required {required!r} first, got {rejected!r}"
        )
    return engine, {
        "init_seconds": round(init_seconds, 6),
        "required_provider": required,
        "session_count": len(sessions),
        "session_providers": provider_lists,
    }


def _variants(frame_dir: Path, center_crops: bool) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = []
    for path in sorted(frame_dir.glob("*.png")):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            variants.append((f"full/{path.name}", np.asarray(image)))
            if center_crops:
                width, height = image.size
                crop = image.crop((width // 4, height // 4, 3 * width // 4, 3 * height // 4))
                variants.append((f"crop/{path.name}", np.asarray(crop)))
    if not variants:
        raise RuntimeError(f"no PNG inputs found in {frame_dir}")
    return variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("cpu", "dml"), required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--center-crops", action="store_true")
    args = parser.parse_args()

    variants = _variants(args.frames, args.center_crops)
    engine, engine_info = _engine(args.provider)
    records: list[dict[str, Any]] = []
    for pass_index in range(args.passes):
        for name, image in variants:
            started = time.perf_counter()
            pages = list(engine.predict(image))
            elapsed = time.perf_counter() - started
            record = {
                "pass": pass_index + 1,
                "name": name,
                "shape": list(image.shape),
                "seconds": round(elapsed, 6),
                "lines": _parse_lines(pages),
            }
            records.append(record)
            # Keep progress output compatible with legacy Windows console code pages.
            # The result file below remains UTF-8 and preserves recognized text.
            print(json.dumps(record, ensure_ascii=True), flush=True)

    steady = [record["seconds"] for record in records if record["pass"] == args.passes]
    summary = {
        "provider": args.provider,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "engine": engine_info,
        "input_count": len(variants),
        "passes": args.passes,
        "steady_seconds": round(sum(steady), 6),
        "steady_mean_seconds": round(statistics.mean(steady), 6),
        "steady_median_seconds": round(statistics.median(steady), 6),
        "steady_fps": round(len(steady) / sum(steady), 6),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
