"""Regressions for vision/nav/sim bugs found in the 2026-09 partition audit."""
import threading
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from nav import locate_task, store_map
from sim import env
from vision import md_tools, ocr_server
from vision import perception


def _png(size=(64, 36)):
    buf = BytesIO()
    Image.new("RGB", size, "white").save(buf, "PNG")
    return buf.getvalue()


def _align(monkeypatch, lidar_samples):
    samples = iter(lidar_samples)
    monkeypatch.setattr(store_map, "RequestLidarCenter", lambda: next(samples))
    monkeypatch.setattr(store_map, "step_agent", lambda *_a: ((0, 0, 0), (0, 90.0, 0), False))
    monkeypatch.setattr(store_map, "perpendicular_yaw", lambda _cp: 90.0)
    monkeypatch.setattr(perception, "center_to_scanner",
                        lambda **_k: {"outcome": "success", "box": {"cx": 960, "cy": 540}})
    nav = SimpleNamespace(
        sm=SimpleNamespace(counter_checkpoint=lambda: 54, by_id={54: {}}),
        args=SimpleNamespace(uri=None), pos=(0, 0, 0), rot=(0, 90.0, 0),
    )
    return store_map.align_to_scanner(nav, max_advance_iters=2)


def test_align_without_any_lidar_hit_reports_instead_of_crashing(monkeypatch):
    hit = {"hit": True, "distance": 1.0, "pitch_deg": 20.0}
    result = _align(monkeypatch, [hit, {"hit": False}, {"hit": False}, {"hit": False}])
    assert result["aligned"] is False and result["slant"] is None
    assert "no LiDAR hit" in result["reason"]


def test_align_tolerates_sample_without_pitch(monkeypatch):
    no_pitch = {"hit": True, "distance": 0.8, "pitch_deg": None}
    result = _align(monkeypatch, [{"hit": True, "distance": 1.0, "pitch_deg": 20.0},
                                  no_pitch, no_pitch])
    assert result["slant"] == pytest.approx(0.8)


def test_checkout_baseline_ocrs_the_centering_frame(monkeypatch):
    frame = Image.new("RGB", (8, 8))
    seen = {}
    monkeypatch.setattr(store_map, "align_to_scanner", lambda *_a, **_k: {"aligned": True})
    monkeypatch.setattr(perception, "center_to_screen",
                        lambda **_k: {"box": {"xmin": 0}, "_source_image": frame})
    monkeypatch.setattr(perception, "center_to_scanner", lambda **_k: {})
    monkeypatch.setattr(perception, "read_text_in_box",
                        lambda box, source_image=None: seen.setdefault("src", source_image) and [])
    failure, aligned, _ = store_map._approach_checkout(None, [], {}, False, None)
    assert failure is None and aligned and seen["src"] is frame


def test_perception_client_has_bounded_timeout():
    assert perception.CLIENT.timeout == 180.0


@pytest.mark.parametrize("ymin_first, expected", [(False, (0.2, 0.4)), (True, (0.4, 0.2))])
def test_md_pointing_fallback_respects_provider_bbox_order(monkeypatch, ymin_first, expected):
    reply = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content='{"box_2d": [100, 200, 300, 600]}'))])
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_k: reply)))
    monkeypatch.setattr(perception, "CLIENT", fake)
    monkeypatch.setattr(perception, "BBOX_YMIN_FIRST", ymin_first)
    [point] = md_tools._point_via_qwen(Image.new("RGB", (4, 4)), "item")
    assert (point["x"], point["y"]) == pytest.approx(expected)


def test_md_reach_uses_configured_sandbox_uri(monkeypatch):
    sent = {}

    async def send(command, uri=None):
        sent["uri"] = uri

    monkeypatch.setattr(md_tools, "RequestScreenshot", lambda: {"image": _png()})
    monkeypatch.setattr(md_tools._model, "point", lambda *_a: {"points": [{"x": 0.5, "y": 0.5}]})
    monkeypatch.setattr(md_tools, "SendCommand", send)
    assert md_tools.reach_item_in_view("item", True) == {"reached": True}
    assert sent["uri"] is None  # None -> SendCommand resolves SARI_WS_URI


def test_ocr_handler_has_socket_timeout():
    app = ocr_server.OcrApplication(lambda: object())
    assert ocr_server.make_handler(app).timeout == ocr_server.REQUEST_TIMEOUT_S


def test_zoom_tiles_closes_source_image(monkeypatch, tmp_path):
    source = tmp_path / "frame.png"
    source.write_bytes(_png())
    opened = []
    real_open = Image.open
    monkeypatch.setattr(Image, "open", lambda *a, **k: opened.append(real_open(*a, **k)) or opened[-1])
    assert len(locate_task.zoom_tiles(str(source), str(tmp_path))) == 4
    assert opened[0].fp is None


def test_send_works_from_a_thread_without_an_event_loop(monkeypatch):
    async def reply(_command, _uri):
        return "ok"

    monkeypatch.setattr(env, "_send_command_once", reply)
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault("r", env._send({"command": "X"}, "ws://s")))
    worker.start()
    worker.join(5)
    assert out["r"] == "ok"
