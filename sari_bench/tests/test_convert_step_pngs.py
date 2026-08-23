from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from sari_bench import convert_step_pngs


def _png_bytes(size: tuple[int, int] = (64, 48), seed: int = 0) -> bytes:
    """A noisy (not flat-color) image: representative enough that JPEG beats PNG on size, the
    way it does on the real photographic screenshots this tool exists for."""
    import random

    rng = random.Random(seed)
    pixels = [rng.randrange(256) for _ in range(size[0] * size[1] * 3)]
    image = Image.new("RGB", size)
    image.putdata(list(zip(pixels[0::3], pixels[1::3], pixels[2::3])))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _attempt(root: Path, *, state: str = "finished") -> Path:
    run = root / "battery" / "prompt" / "try01"
    leg = run / "leg00"
    leg.mkdir(parents=True)
    (run / "attempt.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    (leg / "step01.png").write_bytes(_png_bytes(seed=1))
    (leg / "step02_center.png").write_bytes(_png_bytes(seed=2))
    (leg / "step01.txt").write_text("not a frame", encoding="utf-8")
    return run


def test_convert_is_dry_run_then_explicit_apply(tmp_path: Path) -> None:
    run = _attempt(tmp_path)
    dry = convert_step_pngs.convert_attempt(run, tmp_path.resolve(), apply=False)
    assert dry.files == 2
    assert (run / "leg00/step01.png").exists()
    assert not (run / "leg00/step01.jpg").exists()

    applied = convert_step_pngs.convert_attempt(run, tmp_path.resolve(), apply=True)
    assert applied.files == 2
    assert applied.jpg_bytes > 0
    assert applied.jpg_bytes < applied.png_bytes

    assert not (run / "leg00/step01.png").exists()
    assert not (run / "leg00/step02_center.png").exists()
    jpg01 = run / "leg00/step01.jpg"
    jpg02 = run / "leg00/step02_center.jpg"
    assert jpg01.is_file()
    assert jpg02.is_file()
    with Image.open(jpg01) as img:
        assert img.format == "JPEG"
        assert img.size == (64, 48)
    # Untouched: not a step-frame PNG.
    assert (run / "leg00/step01.txt").exists()


def test_convert_covers_retried_leg_directories(tmp_path: Path) -> None:
    """leg{NN}_retry{M} dirs hold a leg's retried attempts (orchestration.py's `suffix`); their
    step frames are exactly as legacy as leg00's and must not be silently skipped."""
    run = tmp_path / "battery" / "prompt" / "try01"
    retry_leg = run / "leg00_retry1"
    retry_leg.mkdir(parents=True)
    (run / "attempt.json").write_text(json.dumps({"state": "finished"}), encoding="utf-8")
    (retry_leg / "step01.png").write_bytes(_png_bytes(seed=3))

    applied = convert_step_pngs.convert_attempt(run, tmp_path.resolve(), apply=True)
    assert applied.files == 1
    assert not (retry_leg / "step01.png").exists()
    assert (retry_leg / "step01.jpg").is_file()


def test_convert_refuses_running_and_missing_manifest(tmp_path: Path) -> None:
    running_root = tmp_path / "running"
    running = _attempt(running_root, state="running")
    result = convert_step_pngs.convert_attempt(running, running_root.resolve(), apply=True)
    assert "not finished" in result.skipped
    assert (running / "leg00/step01.png").exists()

    no_manifest_root = tmp_path / "no_manifest"
    run = no_manifest_root / "battery" / "prompt" / "try01"
    (run / "leg00").mkdir(parents=True)
    (run / "attempt.json").write_text("not json", encoding="utf-8")
    result = convert_step_pngs.convert_attempt(run, no_manifest_root.resolve(), apply=True)
    assert "missing or unreadable" in result.skipped


def test_convert_skips_when_jpg_already_exists(tmp_path: Path) -> None:
    run = _attempt(tmp_path)
    (run / "leg00/step01.jpg").write_bytes(b"already here")
    result = convert_step_pngs.convert_attempt(run, tmp_path.resolve(), apply=True)
    assert "already exists" in result.skipped
    # Refused before touching anything, so both original PNGs survive.
    assert (run / "leg00/step01.png").exists()
    assert (run / "leg00/step02_center.png").exists()
