from __future__ import annotations

import os
import sys
import threading
import time
from io import BytesIO
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

_ROOT = os.path.dirname(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.vision import ocr_server
from agent.vision.ocr_server import (
    OcrApplication,
    OcrHttpServer,
    build_paddle_engine,
    create_server,
    make_handler,
    parse_paddle_lines,
    resolve_ocr_backend,
)


def png_bytes(size=(8, 6)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "white").save(buffer, "PNG")
    return buffer.getvalue()


def test_both_paddle_result_shapes_and_empty():
    assert parse_paddle_lines([{"rec_texts": [" A ", "", "B"]}]) == ["A", "B"]
    assert parse_paddle_lines([[(None, (" A ", 0.9)), (None, ("B", 0.8))]]) == ["A", "B"]
    assert parse_paddle_lines([]) == []
    assert parse_paddle_lines([[]]) == []


def test_invalid_images_and_request_limit():
    app = OcrApplication(lambda: object(), max_request_bytes=64)
    with pytest.raises(ValueError, match="valid PNG"):
        app.infer_png(b"not an image")
    with pytest.raises(OverflowError):
        app.infer_png(b"x" * 65)


def test_directml_backend_builds_onnx_engine_with_required_session_options(monkeypatch):
    captured = {}
    engine = object()

    def fake_paddle_ocr(**kwargs):
        captured.update(kwargs)
        return engine

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=fake_paddle_ocr),
    )
    monkeypatch.setattr(
        ocr_server,
        "_verify_onnx_provider",
        lambda required: [[required, "CPUExecutionProvider"]],
    )

    assert build_paddle_engine("directml", directml_device_id=2) is engine
    assert captured["engine"] == "onnxruntime"
    assert captured["device"] == "cpu"
    assert captured["use_doc_orientation_classify"] is False
    assert captured["use_doc_unwarping"] is False
    assert captured["engine_config"] == {
        "providers": ["DmlExecutionProvider"],
        "provider_options": {"device_id": 2},
        "execution_mode": "sequential",
        "enable_mem_pattern": False,
    }


def test_cuda_backend_builds_verified_onnx_engine(monkeypatch):
    captured = {}
    engine = object()
    preloaded = []

    def fake_paddle_ocr(**kwargs):
        captured.update(kwargs)
        return engine

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=fake_paddle_ocr),
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(preload_dlls=lambda **kwargs: preloaded.append(kwargs)),
    )
    monkeypatch.setattr(
        ocr_server,
        "_verify_onnx_provider",
        lambda required: [[required, "CPUExecutionProvider"]],
    )

    assert build_paddle_engine("cuda", device_id=3) is engine
    assert captured["engine"] == "onnxruntime"
    assert captured["device"] == "gpu:3"
    assert captured["use_doc_orientation_classify"] is False
    assert captured["use_doc_unwarping"] is False
    assert captured["engine_config"] == {
        "providers": ["CUDAExecutionProvider"],
        "provider_options": {"device_id": 3},
    }
    assert preloaded == [{"directory": ""}]


def test_cpu_backend_forces_cpu_even_when_a_gpu_is_present(monkeypatch):
    captured = {}
    engine = object()
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=lambda **kwargs: captured.update(kwargs) or engine),
    )

    assert build_paddle_engine("cpu") is engine
    assert captured["device"] == "cpu"
    assert captured["use_doc_orientation_classify"] is False
    assert captured["use_doc_unwarping"] is False


def test_auto_backend_selects_directml_only_on_windows_when_available(monkeypatch):
    monkeypatch.setattr(ocr_server.sys, "platform", "win32")
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(get_available_providers=lambda: ["DmlExecutionProvider"]),
    )
    assert resolve_ocr_backend("auto") == "directml"

    monkeypatch.setattr(ocr_server.sys, "platform", "linux")
    assert resolve_ocr_backend("auto") == "cpu"

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider"]),
    )
    assert resolve_ocr_backend("auto") == "cuda"


def test_health_reports_accelerator_identity_when_configured():
    app = OcrApplication(
        lambda: object(),
        model="paddleocr:test:directml",
        backend="directml",
        execution_provider="DmlExecutionProvider",
    )
    assert app.health_payload() == {
        "ready": True,
        "api_version": "v1",
        "model": "paddleocr:test:directml",
        "backend": "directml",
        "execution_provider": "DmlExecutionProvider",
    }


def test_main_reads_cuda_backend_and_device_from_runconfig(monkeypatch, tmp_path):
    config_path = tmp_path / "runconfig.toml"
    config_path.write_text('[ocr]\nbackend = "cuda"\ndevice_id = 4\n', encoding="utf-8")
    captured = {}

    class Server:
        server_address = ("127.0.0.1", 9100)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            captured["closed"] = True

    def fake_create_server(*args, **kwargs):
        captured.update(kwargs)
        return Server()

    monkeypatch.delenv(ocr_server.OCR_BACKEND_ENV, raising=False)
    monkeypatch.setattr(ocr_server, "create_server", fake_create_server)

    assert ocr_server.main(["--config", str(config_path)]) == 0
    assert captured["backend"] == "cuda"
    assert captured["execution_provider"] == "CUDAExecutionProvider"
    assert "cuda" in captured["model"]
    assert captured["closed"] is True

    captured_engine = {}
    monkeypatch.setattr(
        ocr_server,
        "build_paddle_engine",
        lambda backend, *, device_id: captured_engine.update(
            backend=backend, device_id=device_id
        ),
    )
    captured["engine_factory"]()
    assert captured_engine == {"backend": "cuda", "device_id": 4}


def test_windows_server_uses_exclusive_address_ownership(monkeypatch):
    monkeypatch.setattr(ocr_server.sys, "platform", "win32")
    monkeypatch.setattr(ocr_server.socket, "SO_EXCLUSIVEADDRUSE", 12345, raising=False)
    calls = []

    class FakeSocket:
        def setsockopt(self, *args):
            calls.append(args)

    server = object.__new__(OcrHttpServer)
    server.socket = FakeSocket()
    monkeypatch.setattr(ocr_server.ThreadingHTTPServer, "server_bind", lambda _self: None)

    server.server_bind()

    assert calls == [(ocr_server.socket.SOL_SOCKET, 12345, 1)]


def test_closed_log_stream_does_not_abort_request(monkeypatch):
    app = OcrApplication(lambda: object())
    handler = object.__new__(make_handler(app))
    monkeypatch.setattr(handler, "address_string", lambda: "127.0.0.1")

    def closed_stream(*_args, **_kwargs):
        raise OSError("invalid inherited stdout")

    monkeypatch.setattr("builtins.print", closed_stream)
    handler.log_message('%s "%s"', "GET", "/health")


def test_slow_engine_is_constructed_once_and_inference_is_serialized():
    state = {"constructed": 0, "active": 0, "peak": 0, "calls": 0}
    state_lock = threading.Lock()

    class Engine:
        def predict(self, _source):
            with state_lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.03)
            with state_lock:
                state["active"] -= 1
                state["calls"] += 1
            return [{"rec_texts": ["OK"]}]

    def factory():
        state["constructed"] += 1
        return Engine()

    app = OcrApplication(factory, max_in_flight=24)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(app.infer_png(png_bytes())[0]))
        for _ in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state == {"constructed": 1, "active": 0, "peak": 1, "calls": 20}
    assert results == [["OK"]] * 20


def test_http_health_structured_errors_size_limit_and_overload():
    entered = threading.Event()
    release = threading.Event()

    class Engine:
        def predict(self, _source):
            entered.set()
            release.wait(2)
            return [{"rec_texts": ["OK"]}]

    server = create_server(
        "127.0.0.1",
        0,
        engine_factory=lambda: Engine(),
        model="fake-model",
        max_in_flight=1,
        max_request_bytes=1024,
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        health = requests.get(f"{base}/health", timeout=2)
        assert health.json() == {"ready": True, "api_version": "v1", "model": "fake-model"}

        invalid = requests.post(f"{base}/v1/ocr", data=b"bad", timeout=2)
        assert invalid.status_code == 400
        assert invalid.json()["error"]["code"] == "invalid_image"

        large = requests.post(f"{base}/v1/ocr", data=b"x" * 1025, timeout=2)
        assert large.status_code == 413
        assert large.json()["error"]["code"] == "request_too_large"

        first = threading.Thread(
            target=lambda: requests.post(f"{base}/v1/ocr", data=png_bytes(), timeout=3)
        )
        first.start()
        assert entered.wait(1)
        overloaded = requests.post(f"{base}/v1/ocr", data=png_bytes(), timeout=2)
        assert overloaded.status_code == 503
        assert overloaded.json()["error"]["code"] == "overloaded"
        release.set()
        first.join()
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        serving.join()
