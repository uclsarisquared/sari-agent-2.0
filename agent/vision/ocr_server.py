"""One-process, serialized PaddleOCR HTTP service for all local agent workers."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError

from .ocr_client import API_VERSION

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9100
DEFAULT_MAX_REQUEST_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_IN_FLIGHT = 32
MODEL_IDENTITY = "paddleocr:en:text-line-orientation"
OCR_BACKEND_ENV = "SARI_OCR_BACKEND"
OCR_BACKENDS = ("auto", "paddle", "onnx-cpu", "directml")
DEFAULT_OCR_BACKEND = "auto"
DEFAULT_DIRECTML_DEVICE_ID = 0


def resolve_ocr_backend(requested: str) -> str:
    """Resolve ``auto`` without importing ONNX Runtime on the normal Paddle path."""
    backend = requested.strip().lower()
    if backend not in OCR_BACKENDS:
        raise ValueError(f"unknown OCR backend {requested!r}; choose from {OCR_BACKENDS}")
    if backend != "auto":
        return backend
    if sys.platform != "win32":
        return "paddle"
    try:
        import onnxruntime as ort
    except ImportError:
        return "paddle"
    return (
        "directml"
        if "DmlExecutionProvider" in ort.get_available_providers()
        else "paddle"
    )


def _verify_onnx_provider(required_provider: str) -> list[list[str]]:
    """Fail startup if PaddleOCR silently created CPU-only ONNX sessions."""
    import onnxruntime as ort

    sessions = [
        candidate
        for candidate in gc.get_objects()
        if isinstance(candidate, ort.InferenceSession)
    ]
    provider_lists = [session.get_providers() for session in sessions]
    if not sessions:
        raise RuntimeError("PaddleOCR initialized without discoverable ONNX Runtime sessions")
    rejected = [
        providers
        for providers in provider_lists
        if not providers or providers[0] != required_provider
    ]
    if rejected:
        raise RuntimeError(
            f"provider verification failed: required {required_provider!r} first, "
            f"got {rejected!r}"
        )
    return provider_lists


def build_paddle_engine(
    backend: str = "paddle",
    *,
    directml_device_id: int = DEFAULT_DIRECTML_DEVICE_ID,
) -> Any:
    """Construct one PaddleOCR pipeline using the selected inference backend."""
    from paddleocr import PaddleOCR

    if backend == "paddle":
        return PaddleOCR(lang="en", use_textline_orientation=True, enable_mkldnn=False)
    if backend not in {"onnx-cpu", "directml"}:
        raise ValueError(f"build_paddle_engine requires a resolved backend, got {backend!r}")

    if backend == "directml":
        required_provider = "DmlExecutionProvider"
        engine_config: dict[str, Any] = {
            "providers": [required_provider],
            "provider_options": {"device_id": directml_device_id},
            # Required by ONNX Runtime's DirectML execution provider.
            "execution_mode": "sequential",
            "enable_mem_pattern": False,
        }
    else:
        required_provider = "CPUExecutionProvider"
        engine_config = {"providers": [required_provider]}

    engine = PaddleOCR(
        lang="en",
        use_textline_orientation=True,
        # PaddleOCR 3.7's ONNX wrapper only recognizes CUDA for device="gpu".
        # The explicit provider above still controls the actual inference device.
        device="cpu",
        engine="onnxruntime",
        engine_config=engine_config,
    )
    providers = _verify_onnx_provider(required_provider)
    print(
        f"[ocr-server] verified {len(providers)} ONNX sessions with "
        f"{required_provider} first",
        flush=True,
    )
    return engine


def model_identity(backend: str) -> str:
    return f"{MODEL_IDENTITY}:{backend}"


def parse_paddle_lines(result: Any) -> list[str]:
    """Normalize PaddleOCR 3.x dict pages and legacy 2.x tuple pages."""
    if not result or not result[0]:
        return []
    page = result[0]
    if isinstance(page, dict):
        texts = page.get("rec_texts") or []
    else:
        texts = [line[1][0] for line in page]
    return [text.strip() for text in texts if isinstance(text, str) and text.strip()]


class OcrApplication:
    """Owns one engine, one inference lock, and a bounded set of admitted requests."""

    def __init__(
        self,
        engine_factory: Callable[[], Any] = build_paddle_engine,
        *,
        model: str = MODEL_IDENTITY,
        backend: str | None = None,
        execution_provider: str | None = None,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    ) -> None:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        if max_request_bytes < 1:
            raise ValueError("max_request_bytes must be at least 1")
        self.model = model
        self.backend = backend
        self.execution_provider = execution_provider
        self.max_request_bytes = max_request_bytes
        self._engine = engine_factory()
        self._admission = threading.BoundedSemaphore(max_in_flight)
        self._inference_lock = threading.Lock()

    def health_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ready": True,
            "api_version": API_VERSION,
            "model": self.model,
        }
        if self.backend is not None:
            payload["backend"] = self.backend
        if self.execution_provider is not None:
            payload["execution_provider"] = self.execution_provider
        return payload

    def infer_png(self, body: bytes) -> tuple[list[str], float]:
        if not body:
            raise ValueError("request body is empty")
        if len(body) > self.max_request_bytes:
            raise OverflowError(
                f"request body exceeds {self.max_request_bytes} byte limit"
            )
        try:
            with Image.open(BytesIO(body)) as opened:
                opened.load()
                if opened.format != "PNG":
                    raise ValueError("request body is not a PNG image")
                image = opened.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("request body is not a valid PNG image") from error

        import numpy as np

        source = np.asarray(image)
        started = time.perf_counter()
        with self._inference_lock:
            predict = getattr(self._engine, "predict", None)
            result = predict(source) if callable(predict) else self._engine.ocr(source)
            if isinstance(result, dict):
                result = [result]
            elif not isinstance(result, (list, tuple)):
                result = list(result)
        return parse_paddle_lines(result), round((time.perf_counter() - started) * 1000.0, 2)

    def try_admit(self) -> bool:
        return self._admission.acquire(blocking=False)

    def release(self) -> None:
        self._admission.release()


def make_handler(application: OcrApplication) -> type[BaseHTTPRequestHandler]:
    class OcrHandler(BaseHTTPRequestHandler):
        server_version = "SariOCR/1"

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, code: str, message: str) -> None:
            self._json(status, {"error": {"code": code, "message": message}})

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/health":
                self._error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")
                return
            self._json(HTTPStatus.OK, application.health_payload())

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/v1/ocr":
                self._error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")
                return
            if not application.try_admit():
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "overloaded",
                    "OCR request queue is full",
                )
                return
            try:
                length_header = self.headers.get("Content-Length")
                if length_header is None:
                    self._error(
                        HTTPStatus.LENGTH_REQUIRED,
                        "length_required",
                        "Content-Length is required",
                    )
                    return
                try:
                    length = int(length_header)
                except ValueError:
                    self._error(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_length",
                        "Content-Length must be an integer",
                    )
                    return
                if length < 0:
                    self._error(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_length",
                        "Content-Length cannot be negative",
                    )
                    return
                if length > application.max_request_bytes:
                    self._error(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        "request_too_large",
                        f"PNG exceeds {application.max_request_bytes} byte limit",
                    )
                    return
                body = self.rfile.read(length)
                try:
                    lines, elapsed_ms = application.infer_png(body)
                except OverflowError as error:
                    self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large", str(error))
                    return
                except ValueError as error:
                    self._error(HTTPStatus.BAD_REQUEST, "invalid_image", str(error))
                    return
                except Exception as error:  # Paddle failures are server errors, never fake empties.
                    self._error(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "inference_failed",
                        f"{type(error).__name__}: {error}",
                    )
                    return
                self._json(
                    HTTPStatus.OK,
                    {"lines": lines, "model": application.model, "elapsed_ms": elapsed_ms},
                )
            finally:
                application.release()

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[ocr-server] {self.address_string()} {fmt % args}", flush=True)

    return OcrHandler


def create_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    engine_factory: Callable[[], Any] = build_paddle_engine,
    model: str = MODEL_IDENTITY,
    backend: str | None = None,
    execution_provider: str | None = None,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> ThreadingHTTPServer:
    application = OcrApplication(
        engine_factory,
        model=model,
        backend=backend,
        execution_provider=execution_provider,
        max_in_flight=max_in_flight,
        max_request_bytes=max_request_bytes,
    )
    server = ThreadingHTTPServer((host, port), make_handler(application))
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the central runner-local PaddleOCR service.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--backend",
        choices=OCR_BACKENDS,
        default=os.environ.get(OCR_BACKEND_ENV, DEFAULT_OCR_BACKEND),
        help="Inference backend (auto selects DirectML on Windows when available).",
    )
    parser.add_argument(
        "--directml-device-id",
        type=int,
        default=DEFAULT_DIRECTML_DEVICE_ID,
        help="DirectML adapter index (default: 0).",
    )
    parser.add_argument("--max-in-flight", type=int, default=DEFAULT_MAX_IN_FLIGHT)
    parser.add_argument("--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES)
    args = parser.parse_args(argv)

    backend = resolve_ocr_backend(args.backend)
    provider = {
        "paddle": "PaddlePaddle",
        "onnx-cpu": "CPUExecutionProvider",
        "directml": "DmlExecutionProvider",
    }[backend]
    identity = model_identity(backend)
    print(
        f"[ocr-server] loading PaddleOCR pipeline "
        f"(backend={backend}, provider={provider})...",
        flush=True,
    )
    server = create_server(
        args.host,
        args.port,
        engine_factory=lambda: build_paddle_engine(
            backend,
            directml_device_id=args.directml_device_id,
        ),
        model=identity,
        backend=backend,
        execution_provider=provider,
        max_in_flight=args.max_in_flight,
        max_request_bytes=args.max_request_bytes,
    )
    host, port = server.server_address[:2]
    print(
        f"[ocr-server] ready on http://{host}:{port} "
        f"(model={identity}, provider={provider}, max_in_flight={args.max_in_flight})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
