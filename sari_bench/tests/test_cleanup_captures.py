from __future__ import annotations

import json
from pathlib import Path

from sari_bench import cleanup_captures, video


def _attempt(root: Path, *, state: str = "finished") -> Path:
    run = root / "battery" / "prompt" / "try01"
    capture = run / "capture"
    capture.mkdir(parents=True)
    (run / "attempt.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    (run / "replay.mp4").write_bytes(b"verified")
    (capture / "frame000001-123.jpg").write_bytes(b"one")
    (capture / "frame000002-456.jpg").write_bytes(b"two-two")
    (capture / "latest.json").write_text("{}", encoding="utf-8")
    (capture / "latest.jpg").write_bytes(b"live")
    return run


def _validated_replays():
    old_complete, old_duration = video.is_complete_mp4, video.ffprobe_duration
    video.is_complete_mp4 = lambda _path: True
    video.ffprobe_duration = lambda _path: 10.0
    return old_complete, old_duration


def test_cleanup_is_dry_run_then_explicit_apply(tmp_path: Path) -> None:
    run = _attempt(tmp_path)
    old_complete, old_duration = _validated_replays()
    try:
        dry = cleanup_captures.inspect_attempt(run, tmp_path.resolve(), apply=False)
        assert (dry.files, dry.bytes) == (3, 12)
        assert (run / "capture/frame000001-123.jpg").exists()
        applied = cleanup_captures.inspect_attempt(run, tmp_path.resolve(), apply=True)
        assert (applied.files, applied.bytes) == (3, 12)
        assert not (run / "capture/frame000001-123.jpg").exists()
        assert (run / "capture/latest.jpg").read_bytes() == b"live"
    finally:
        video.is_complete_mp4, video.ffprobe_duration = old_complete, old_duration


def test_cleanup_seeds_latest_jpg_for_attempts_without_one(tmp_path: Path) -> None:
    run = _attempt(tmp_path)
    (run / "capture/latest.jpg").unlink()
    old_complete, old_duration = _validated_replays()
    try:
        applied = cleanup_captures.inspect_attempt(run, tmp_path.resolve(), apply=True)
        assert applied.seeded_latest is True
        assert (run / "capture/latest.jpg").read_bytes() == b"two-two"
        assert not (run / "capture/frame000002-456.jpg").exists()
    finally:
        video.is_complete_mp4, video.ffprobe_duration = old_complete, old_duration


def test_cleanup_refuses_running_missing_and_symlinked_captures(tmp_path: Path) -> None:
    running_root = tmp_path / "running"
    running = _attempt(running_root, state="running")
    assert "not finished" in cleanup_captures.inspect_attempt(
        running, running_root.resolve(), apply=True
    ).skipped

    missing_root = tmp_path / "missing"
    missing = _attempt(missing_root)
    (missing / "replay.mp4").unlink()
    assert "missing" in cleanup_captures.inspect_attempt(
        missing, missing_root.resolve(), apply=True
    ).skipped

    link_root = tmp_path / "linked"
    linked = _attempt(link_root)
    real_capture = linked / "real-capture"
    (linked / "capture").rename(real_capture)
    (linked / "capture").symlink_to(real_capture, target_is_directory=True)
    assert "symlink" in cleanup_captures.inspect_attempt(
        linked, link_root.resolve(), apply=True
    ).skipped
