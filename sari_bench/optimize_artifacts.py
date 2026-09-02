"""Compact closed distributed-benchmark artifacts without touching executable attempts.

The command is deliberately dry-run by default.  It is for historical Sari Bench batteries, not
for generic agent output: only terminal ``finished`` attempts and fully archived requeues are
eligible, and any symlink, malformed manifest, bad replay, or legacy numbered capture archive is
left alone.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from sari_bench import video
from sari_bench.storage import write_json_atomic

PROFILE = "sari-bench-artifacts-v1"
_REQUEUE = re.compile(r"\.requeue\d+$")
_STEP_PNG = re.compile(r"^step\d+(?:_[^.]+)?\.png$", re.IGNORECASE)
_BBOX_PNG = re.compile(r"^look\d+_bbox\.png$", re.IGNORECASE)
_CHECK_PNG = re.compile(r"^check\d+.*\.png$", re.IGNORECASE)
_CLIENT_PNG = re.compile(r"^ClientScreenshot.*\.png$", re.IGNORECASE)


@dataclass
class OptimizeResult:
    attempt: Path
    converted: int = 0
    removed: int = 0
    replay_reencoded: bool = False
    skipped: str = ""
    notes: list[str] = field(default_factory=list)


def _manifest(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _attempt_dirs(scope: Path) -> list[Path]:
    if (scope / "attempt.json").is_file():
        return [scope]
    attempts: list[Path] = []
    # rglob follows neither directory symlinks nor inaccessible directories in normal use, but the
    # later per-attempt symlink gate remains authoritative.
    for path in scope.rglob("attempt.json"):
        if path.is_file():
            attempts.append(path.parent)
    return sorted(set(attempts))


def _has_symlink(attempt: Path) -> bool:
    if attempt.is_symlink():
        return True
    try:
        return any(path.is_symlink() for path in attempt.rglob("*"))
    except OSError:
        return True


def _closed_kind(attempt: Path, manifest: dict) -> str | None:
    state = manifest.get("state")
    if state == "finished":
        return "finished"
    if (
        state == "requeued"
        and manifest.get("pending_retry") is False
        and _REQUEUE.search(attempt.name)
    ):
        return "archived_requeue"
    return None


def _png_quality(path: Path) -> int | None:
    """Return the JPEG profile for a known visual artifact, never for arbitrary PNG data."""
    name = path.name
    parent = path.parent.name
    if _BBOX_PNG.match(name) or name.lower() == "annotated_target.png":
        return 85
    if _CHECK_PNG.match(name):
        return 90
    if _CLIENT_PNG.match(name):
        return 85
    if parent.startswith("leg") and _STEP_PNG.match(name):
        return 85
    return None


def _convertible_pngs(attempt: Path) -> list[tuple[Path, int]]:
    candidates: list[tuple[Path, int]] = []
    for path in attempt.rglob("*.png"):
        if not path.is_file() or path.is_symlink():
            continue
        quality = _png_quality(path)
        if quality is not None:
            candidates.append((path, quality))
    return candidates


def _visual_artifacts(attempt: Path) -> list[Path]:
    """Debug/evidence imagery removable from a replay-verified archived requeue only."""
    candidates: list[Path] = []
    for path in attempt.rglob("*"):
        if path.is_symlink():
            continue
        relative = path.relative_to(attempt)
        parts = relative.parts
        if not parts:
            continue
        name = path.name
        # Entire per-step diagnostic directories contain bbox and inspection evidence only.
        if path.is_dir() and len(parts) == 2 and parts[0].startswith("leg") and (
            name.endswith("_center") or name.endswith("_inspection")
        ):
            candidates.append(path)
        elif path.is_dir() and len(parts) == 1 and name == "annotations":
            candidates.append(path)
        elif path.is_file() and (
            _CLIENT_PNG.match(name)
            or name.lower().startswith("clientscreenshot") and name.lower().endswith((".jpg", ".jpeg"))
            or name.lower() in {"annotated_target.png", "annotated_target.jpg", "annotated_target.jpeg"}
            or (path.parent.name.startswith("leg") and re.match(r"^step\d+(?:_[^.]+)?\.(?:png|jpe?g)$", name, re.I))
        ):
            candidates.append(path)
    # Avoid unlinking a file which lives below a directory we will remove.
    directory_set = set(path for path in candidates if path.is_dir())
    return [path for path in candidates if not any(parent in directory_set for parent in path.parents)]


def _jpeg_bytes(source: Path, quality: int) -> bytes:
    from PIL import Image

    with Image.open(source) as opened:
        image = opened.convert("RGB")
        if image.width > 1920 or image.height > 1080:
            scale = min(1920 / image.width, 1080 / image.height)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        out = BytesIO()
        image.save(out, format="JPEG", quality=quality, subsampling=0)
        return out.getvalue()


def _write_atomic(path: Path, data: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".jpg.tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _video_stream(path: Path) -> tuple[int, int, float] | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height:format=duration",
             "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=15,
        )
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [None])[0]
        duration = float((payload.get("format") or {}).get("duration"))
        width, height = int(stream["width"]), int(stream["height"])
        if width <= 0 or height <= 0 or duration <= 0:
            return None
        return width, height, duration
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError):
        return None


def _reencode_replay(replay: Path) -> tuple[bool, str]:
    """Atomically replace a valid completed replay with the CRF-28 archival profile."""
    source = _video_stream(replay)
    if not video.valid_replay(replay) or source is None:
        return False, "replay.mp4 is unreadable, incomplete, or has no video stream"
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg is unavailable"
    fd, temp_name = tempfile.mkstemp(prefix=".replay.optimize.", suffix=".part.mp4", dir=replay.parent)
    os.close(fd)
    temporary = Path(temp_name)
    try:
        print(
            f"[optimize-artifacts] re-encoding {replay} "
            f"({source[0]}x{source[1]}, {source[2]:.1f}s) -> CRF 28",
            flush=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(replay), "-map", "0:v:0", "-an",
             "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,"
                    "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black",
             "-r", "4", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(temporary)],
            check=True, capture_output=True, text=True, timeout=900,
        )
        replacement = _video_stream(temporary)
        if not video.valid_replay(temporary) or replacement is None:
            return False, "re-encoded replay failed validation"
        width, height, duration = replacement
        if (width, height) != (1280, 720):
            return False, f"re-encoded replay has unexpected dimensions {width}x{height}"
        # ffmpeg can round by one frame. Two percent (with a 0.25s floor) catches truncated clips
        # without rejecting an otherwise equivalent 4fps archive.
        if abs(duration - source[2]) > max(0.25, source[2] * 0.02):
            return False, "re-encoded replay duration does not match source"
        os.replace(temporary, replay)
        print(
            f"[optimize-artifacts] validated {replay} "
            f"({width}x{height}, {duration:.1f}s)",
            flush=True,
        )
        return True, ""
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"could not re-encode replay.mp4: {error}"
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def purge_archived_visual_artifacts(attempt: Path, *, apply: bool = False) -> tuple[int, str]:
    """Remove replay-covered visuals from one archived requeue, only after replay validation."""
    replay = attempt / video.REPLAY_NAME
    if not video.valid_replay(replay):
        return 0, "replay.mp4 is missing, incomplete, or unreadable; preserved visual artifacts"
    files = _visual_artifacts(attempt)
    if apply:
        for path in files:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return len(files), ""


def optimize_attempt(attempt: Path, root: Path, *, apply: bool = False) -> OptimizeResult:
    result = OptimizeResult(attempt=attempt)
    try:
        attempt.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        result.skipped = "attempt is outside the requested root"
        return result
    if _has_symlink(attempt):
        result.skipped = "attempt contains a symlink"
        return result
    manifest_path = attempt / "attempt.json"
    manifest = _manifest(manifest_path)
    if not manifest:
        result.skipped = "attempt.json is missing or unreadable"
        return result
    kind = _closed_kind(attempt, manifest)
    if kind is None:
        result.skipped = f"attempt is not a closed terminal attempt (state={manifest.get('state')!r})"
        return result

    replay = attempt / video.REPLAY_NAME
    # An already marked replay is not encoded again. PNG conversion/deletion remains independently
    # idempotent, so an interrupted optimizer invocation can safely be re-run.
    optimized = (manifest.get("artifact_optimization") or {}).get("profile") == PROFILE
    if replay.exists() and not optimized:
        if not video.valid_replay(replay):
            result.skipped = "replay.mp4 is missing, incomplete, or unreadable"
            return result
        if apply:
            ok, reason = _reencode_replay(replay)
            if not ok:
                result.skipped = reason
                return result
            result.replay_reencoded = True
        else:
            result.notes.append("would re-encode replay.mp4 to CRF 28")
    elif replay.exists() and not video.valid_replay(replay):
        result.skipped = "replay.mp4 is missing, incomplete, or unreadable"
        return result

    if kind == "archived_requeue":
        files, reason = purge_archived_visual_artifacts(attempt, apply=apply)
        if reason:
            result.skipped = reason
            return result
        result.removed = files
    else:
        candidates = _convertible_pngs(attempt)
        for png, quality in candidates:
            jpg = png.with_suffix(".jpg")
            if jpg.exists():
                result.notes.append(f"kept {png.name}: {jpg.name} already exists")
                continue
            if apply:
                try:
                    _write_atomic(jpg, _jpeg_bytes(png, quality))
                    png.unlink()
                except Exception as error:  # do not delete the source if any conversion step fails
                    result.skipped = f"could not convert {png.name}: {type(error).__name__}: {error}"
                    return result
            result.converted += 1

    if apply and not optimized:
        manifest["artifact_optimization"] = {
            "profile": PROFILE,
            "replay_crf": 28,
        }
        write_json_atomic(manifest_path, manifest)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sari_bench optimize-artifacts",
        description="Dry-run first compaction for safe, closed distributed Sari Bench attempts.",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--bench-root", type=Path, default=Path("bench_runs"))
    scope.add_argument("--run-dir", type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run).")
    args = parser.parse_args(argv)
    requested = args.run_dir if args.run_dir is not None else args.bench_root
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        parser.error(f"requested root is unavailable: {error}")
    if not root.is_dir():
        parser.error("requested root is not a directory")

    attempts = _attempt_dirs(root)
    results = []
    for index, attempt in enumerate(attempts, start=1):
        print(
            f"[optimize-artifacts] [{index}/{len(attempts)}] inspecting {attempt}",
            flush=True,
        )
        results.append(optimize_attempt(attempt, root, apply=args.apply))
    converted = removed = replays = 0
    for result in results:
        if result.skipped:
            print(f"SKIP {result.attempt}: {result.skipped}")
            continue
        converted += result.converted
        removed += result.removed
        replays += int(result.replay_reencoded or (not args.apply and bool(result.notes)))
        verb = "optimized" if args.apply else "would optimize"
        detail = f"{result.converted} PNG->JPEG, {result.removed} visual artifact(s) removed"
        print(f"{result.attempt}: {verb}: {detail}")
        for note in result.notes:
            print(f"  {note}")
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"{mode}: {converted} PNG->JPEG, {removed} visual artifact(s), {replays} replay(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
