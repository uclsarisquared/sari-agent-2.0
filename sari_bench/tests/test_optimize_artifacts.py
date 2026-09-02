from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from sari_bench import optimize_artifacts
from sari_bench.runner import BenchmarkRunner


def _png() -> bytes:
    out = BytesIO()
    Image.new("RGB", (2100, 1200), "steelblue").save(out, format="PNG")
    return out.getvalue()


def _attempt(root: Path, name: str = "try01", **manifest: object) -> Path:
    attempt = root / "battery" / "prompt" / name
    attempt.mkdir(parents=True)
    (attempt / "attempt.json").write_text(json.dumps(manifest), encoding="utf-8")
    return attempt


def test_finished_attempt_converts_only_safe_png_artifacts_atomically(tmp_path: Path) -> None:
    attempt = _attempt(tmp_path, state="finished")
    center = attempt / "leg00" / "step01_center"
    inspect = attempt / "leg00" / "step01_inspection"
    shots = attempt / "screenshots"
    for directory in (center, inspect, shots, attempt / "leg00", attempt / "capture"):
        directory.mkdir(parents=True, exist_ok=True)
    for path in (
        center / "look1_bbox.png", inspect / "check01_initial.png",
        shots / "annotated_target.png", shots / "ClientScreenshot.png",
        attempt / "leg00" / "step01.png",
    ):
        path.write_bytes(_png())
    legacy = attempt / "capture" / "frame000001-1.jpg"
    legacy.write_bytes(b"leave numbered captures alone")

    dry = optimize_artifacts.optimize_attempt(attempt, tmp_path.resolve())
    assert dry.converted == 5
    assert all(path.suffix == ".png" for path, _ in optimize_artifacts._convertible_pngs(attempt))

    applied = optimize_artifacts.optimize_attempt(attempt, tmp_path.resolve(), apply=True)
    assert applied.converted == 5
    assert not list(attempt.rglob("*.png"))
    assert legacy.read_bytes() == b"leave numbered captures alone"
    with Image.open(center / "look1_bbox.jpg") as image:
        assert image.format == "JPEG"
        assert image.size == (1890, 1080)
        assert all(layer[1:3] == (1, 1) for layer in image.layer)  # 4:4:4 sampling factors

    rerun = optimize_artifacts.optimize_attempt(attempt, tmp_path.resolve(), apply=True)
    assert rerun.converted == 0


def test_archived_requeue_purges_only_after_verified_replay(tmp_path: Path, monkeypatch) -> None:
    attempt = _attempt(tmp_path, "try01.requeue00", state="requeued", pending_retry=False)
    center = attempt / "leg00" / "step01_center"
    center.mkdir(parents=True)
    (center / "look1_bbox.jpg").write_bytes(b"debug")
    (attempt / "leg00" / "step01.jpg").write_bytes(b"step")
    (attempt / "replay.mp4").write_bytes(b"replay")
    (attempt / "agent.log").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(optimize_artifacts.video, "valid_replay", lambda _path: True)
    monkeypatch.setattr(optimize_artifacts, "_reencode_replay", lambda _path: (True, ""))

    result = optimize_artifacts.optimize_attempt(attempt, tmp_path.resolve(), apply=True)
    assert result.removed == 2
    assert not center.exists()
    assert not (attempt / "leg00" / "step01.jpg").exists()
    assert (attempt / "agent.log").read_text(encoding="utf-8") == "keep"


def test_runner_rotation_purges_only_a_replay_verified_archive(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "prompt" / "try01"
    center = run / "leg00" / "step01_center"
    center.mkdir(parents=True)
    (run / "attempt.json").write_text(
        json.dumps({"state": "running", "started_epoch": 1.0}), encoding="utf-8"
    )
    (run / "replay.mp4").write_bytes(b"verified by the patched validator")
    (center / "look1_bbox.jpg").write_bytes(b"debug")
    (run / "leg00" / "step01.jpg").write_bytes(b"step")
    (run / "agent.log").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(optimize_artifacts.video, "valid_replay", lambda _path: True)

    BenchmarkRunner._rotate_run_dir(run)

    archive = run.with_name("try01.requeue00")
    assert not (archive / "leg00" / "step01.jpg").exists()
    assert not (archive / "leg00" / "step01_center").exists()
    assert (archive / "agent.log").read_text(encoding="utf-8") == "keep"
    manifest = json.loads((archive / "attempt.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "requeued"
    assert manifest["pending_retry"] is False


def test_optimizer_skips_pending_and_symlinked_attempts(tmp_path: Path) -> None:
    pending = _attempt(tmp_path, "try01.requeue00", state="requeued", pending_retry=True)
    result = optimize_artifacts.optimize_attempt(pending, tmp_path.resolve(), apply=True)
    assert "not a closed" in result.skipped

    linked = _attempt(tmp_path / "other", state="finished")
    (linked / "leg00").mkdir()
    (linked / "leg00" / "linked.png").symlink_to(linked / "attempt.json")
    result = optimize_artifacts.optimize_attempt(linked, (tmp_path / "other").resolve(), apply=True)
    assert "symlink" in result.skipped
