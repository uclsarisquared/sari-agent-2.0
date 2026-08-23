"""Re-encodes legacy per-step debug screenshots from PNG to JPEG, in place.

`leg_artifacts.LegArtifacts.save_frame` wrote `legNN/stepNN.png` before it switched to
`stepNN.jpg` at quality=85 (same bounding box, `agent/sim/env.py:downscale_for_storage_jpeg`).
Runs from before that switch are stuck with the PNGs, which are ~7x the size for the same bounded
1920x1080 debug frame nobody re-reads at full fidelity - only finished, already-inspected legs
carry them, so re-encoding them is a pure disk-space reclaim.

Only ever touches a finished attempt's leg directories, mirroring cleanup_captures.py's stance:
this is about disk space on old evidence, never about a run that could still be writing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

MAX_SAVE_W, MAX_SAVE_H = 1920, 1080
JPEG_QUALITY = 85

_LEG_DIR = re.compile(r"^leg\d+(_retry\d+)?$")
_STEP_FRAME_PNG = re.compile(r"^step(\d+)(?:_[^.]+)?\.png$", re.IGNORECASE)


@dataclass
class ConvertResult:
    attempt: Path
    files: int = 0
    png_bytes: int = 0
    jpg_bytes: int = 0
    skipped: str = ""


def _manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _attempt_dirs(scope: Path) -> list[Path]:
    if (scope / "attempt.json").is_file():
        return [scope]
    return sorted({path.parent for path in scope.rglob("attempt.json")})


def _to_jpeg(png_bytes: bytes) -> bytes:
    """Same bounding + quality as `save_frame` writes today, so a converted frame is
    byte-for-byte what that step would have produced had it run after the format switch."""
    from PIL import Image

    with Image.open(BytesIO(png_bytes)) as source:
        rgb = source.convert("RGB")
        width, height = rgb.size
        if width > MAX_SAVE_W or height > MAX_SAVE_H:
            scale = min(MAX_SAVE_W / width, MAX_SAVE_H / height)
            rgb = rgb.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()


def _step_pngs(attempt: Path) -> list[Path]:
    frames: list[Path] = []
    for leg_dir in attempt.iterdir():
        if not leg_dir.is_dir() or leg_dir.is_symlink() or not _LEG_DIR.match(leg_dir.name):
            continue
        frames.extend(
            path for path in leg_dir.iterdir()
            if path.is_file() and not path.is_symlink() and _STEP_FRAME_PNG.match(path.name)
        )
    return frames


def convert_attempt(attempt: Path, root: Path, *, apply: bool = False) -> ConvertResult:
    result = ConvertResult(attempt)
    try:
        resolved_attempt = attempt.resolve(strict=True)
        resolved_attempt.relative_to(root)
    except (OSError, ValueError):
        result.skipped = "attempt is outside the requested root"
        return result
    manifest = _manifest(attempt / "attempt.json")
    if not manifest:
        result.skipped = "attempt.json is missing or unreadable"
        return result
    if manifest.get("state") != "finished":
        result.skipped = f"attempt is not finished (state={manifest.get('state')!r})"
        return result

    candidates = _step_pngs(attempt)
    if not candidates:
        result.skipped = "no legacy step PNGs"
        return result

    # Checked for every candidate before any file is touched: partway through an attempt is not
    # a state this function may leave behind, so one collision refuses the whole attempt rather
    # than converting everything up to it.
    collision = next(
        (path for path in candidates if path.with_suffix(".jpg").exists()), None
    )
    if collision is not None:
        result.skipped = f"a converted frame already exists: {collision.with_suffix('.jpg').name}"
        return result

    for png_path in candidates:
        jpg_path = png_path.with_suffix(".jpg")
        try:
            png_size = png_path.stat().st_size
        except OSError:
            result.skipped = f"step frame became unreadable: {png_path.name}"
            return result
        result.files += 1
        result.png_bytes += png_size

        if not apply:
            # Dry-run size estimate matches the real ratio closely enough to report without
            # spending the encode; the real pass is what actually measures and writes it.
            continue

        try:
            jpeg = _to_jpeg(png_path.read_bytes())
        except Exception as error:  # noqa: BLE001 - a bad PNG must not abort the whole attempt
            result.skipped = f"could not re-encode {png_path.name}: {error!r}"
            return result

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{jpg_path.stem}.", suffix=".jpg.tmp", dir=jpg_path.parent
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(jpeg)
            os.replace(temp_name, jpg_path)
        except OSError as error:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_name)
            result.skipped = f"could not write {jpg_path.name}: {error!r}"
            return result
        result.jpg_bytes += len(jpeg)
        png_path.unlink()

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sari_bench convert-step-pngs",
        description="Re-encode finished attempts' legacy stepNN.png debug frames as JPEG.",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--bench-root", type=Path, default=Path("bench_runs"))
    scope.add_argument("--run-dir", type=Path)
    parser.add_argument("--apply", action="store_true", help="Convert files (default is dry-run).")
    args = parser.parse_args(argv)
    requested = args.run_dir if args.run_dir is not None else args.bench_root
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        parser.error(f"requested root is unavailable: {error}")
    if not root.is_dir():
        parser.error("requested root is not a directory")

    results = [convert_attempt(attempt, root, apply=args.apply) for attempt in _attempt_dirs(root)]
    total_files = total_png = total_jpg = 0
    verb = "converted" if args.apply else "would convert"
    for result in results:
        label = str(result.attempt)
        if result.skipped:
            if result.skipped != "no legacy step PNGs":
                print(f"SKIP {label}: {result.skipped}")
        else:
            total_files += result.files
            total_png += result.png_bytes
            total_jpg += result.jpg_bytes
            if args.apply:
                print(
                    f"{label}: {verb} {result.files} file(s), "
                    f"{result.png_bytes} -> {result.jpg_bytes} byte(s)"
                )
            else:
                print(f"{label}: {verb} {result.files} file(s), {result.png_bytes} byte(s)")
    mode = "APPLY" if args.apply else "DRY RUN"
    if args.apply:
        print(
            f"{mode}: {verb} {total_files} file(s), "
            f"{total_png} -> {total_jpg} byte(s) total "
            f"({total_png - total_jpg} byte(s) reclaimed)"
        )
    else:
        print(f"{mode}: {verb} {total_files} file(s), {total_png} byte(s) total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
