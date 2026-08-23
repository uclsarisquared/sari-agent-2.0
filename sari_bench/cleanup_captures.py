"""Safely remove legacy numbered capture archives after a verified replay exists."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sari_bench import capture, video

_LEGACY_FRAME = re.compile(r"^frame\d+-\d+\.jpg$")


@dataclass
class CleanupResult:
    attempt: Path
    files: int = 0
    bytes: int = 0
    skipped: str = ""
    seeded_latest: bool = False


def _manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _seed_latest(capture_dir: Path, candidates: list[Path]) -> bool:
    """Writes the newest legacy frame out as latest.jpg before it is deleted.

    Old attempts that predate the fixed live-preview file have no latest.jpg, only the numbered
    frames this cleanup is about to remove. Without this, deleting them leaves the dashboard's
    thumbnail for that attempt with nothing to fall back to.
    """
    if (capture_dir / capture.LATEST_CAPTURE).is_file():
        return False
    frames = [path for path in candidates if _LEGACY_FRAME.match(path.name)]
    if not frames:
        return False
    newest = max(frames, key=capture.frame_timestamp_ns)
    try:
        data = newest.read_bytes()
    except OSError:
        return False
    fd, temp_name = tempfile.mkstemp(prefix=".latest.", suffix=".jpg.tmp", dir=capture_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temp_name, capture_dir / capture.LATEST_CAPTURE)
    except OSError:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        return False
    return True


def _attempt_dirs(scope: Path) -> list[Path]:
    if (scope / "attempt.json").is_file():
        return [scope]
    return sorted({path.parent for path in scope.rglob("attempt.json")})


def inspect_attempt(attempt: Path, root: Path, *, apply: bool = False) -> CleanupResult:
    result = CleanupResult(attempt)
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
    capture_dir = attempt / "capture"
    if not capture_dir.exists():
        result.skipped = "capture directory is missing"
        return result
    if capture_dir.is_symlink():
        result.skipped = "capture directory is a symlink"
        return result
    try:
        capture_dir.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        result.skipped = "capture directory resolves outside the requested root"
        return result
    replay = attempt / video.REPLAY_NAME
    if not replay.is_file():
        result.skipped = "replay.mp4 is missing"
        return result
    if not video.is_complete_mp4(replay):
        result.skipped = "replay.mp4 is structurally incomplete"
        return result
    if video.ffprobe_duration(replay) is None:
        result.skipped = "replay.mp4 is unreadable or has zero duration"
        return result

    candidates = [
        path for path in capture_dir.iterdir()
        if path.is_file() and not path.is_symlink()
        and (_LEGACY_FRAME.match(path.name) or path.name == "latest.json")
    ]
    for path in candidates:
        try:
            size = path.stat().st_size
        except OSError:
            result.skipped = f"capture file became unreadable: {path.name}"
            result.files = result.bytes = 0
            return result
        result.files += 1
        result.bytes += size
    if apply:
        result.seeded_latest = _seed_latest(capture_dir, candidates)
        for path in candidates:
            path.unlink()
        try:
            capture_dir.rmdir()
        except OSError:
            pass
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sari_bench cleanup-captures",
        description="Remove old numbered capture JPEGs only after a valid finished replay exists.",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--bench-root", type=Path, default=Path("bench_runs"))
    scope.add_argument("--run-dir", type=Path)
    parser.add_argument("--apply", action="store_true", help="Delete eligible files (default is dry-run).")
    args = parser.parse_args(argv)
    requested = args.run_dir if args.run_dir is not None else args.bench_root
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        parser.error(f"requested root is unavailable: {error}")
    if not root.is_dir():
        parser.error("requested root is not a directory")

    results = [inspect_attempt(attempt, root, apply=args.apply) for attempt in _attempt_dirs(root)]
    total_files = total_bytes = 0
    verb = "removed" if args.apply else "would remove"
    for result in results:
        label = str(result.attempt)
        if result.skipped:
            print(f"SKIP {label}: {result.skipped}")
        else:
            total_files += result.files
            total_bytes += result.bytes
            seeded = " (seeded latest.jpg)" if result.seeded_latest else ""
            print(f"{label}: {verb} {result.files} file(s), {result.bytes} byte(s){seeded}")
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"{mode}: {verb} {total_files} file(s), {total_bytes} byte(s) total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
