"""Small, process-safe persistence helpers shared by the runner and watcher."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # Windows: fall back to msvcrt byte-range locks
    fcntl = None  # type: ignore[assignment]
    import msvcrt

ATTEMPTS_LOCK = ".attempts.lock"
BATTERY_LOCK = ".battery.lock"
RUNNER_LOCK = ".runner.lock"

# A battery's id IS its directory name: it is joined onto bench_root, comes back out in URLs and in
# report filenames, and is what a browser pins. So the set of names allowed is the set that survives
# all three - no separators, no leading dot, nothing that could climb out of bench_runs/.
BATTERY_NAME_MAX = 64
_UNSAFE_IN_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def battery_dir_name(name: str) -> str:
    """One free-form battery name reduced to a single safe path component, or "" if nothing survives.

    Deliberately shared by `--name` and the dashboard's rename: a run named at launch and a run
    renamed later have to be the same kind of thing, and both reach the same filesystem.
    """
    cleaned = _UNSAFE_IN_NAME.sub("-", name.strip()).strip("-._")
    return cleaned[:BATTERY_NAME_MAX].strip("-._")


def read_json_object(path: Path) -> dict[str, Any]:
    """A JSON object from disk, or {} when missing, unreadable, or not an object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_bytes_atomic(path: Path, data: bytes, *, fsync: bool = True) -> None:
    """Publish bytes via a unique temp file and rename (fsynced unless `fsync=False`)."""
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=f"{path.suffix}.tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def attempt_dirs(scope: Path) -> list[Path]:
    """Every attempt dir (one holding attempt.json) at or below `scope`."""
    if (scope / "attempt.json").is_file():
        return [scope]
    return sorted({path.parent for path in scope.rglob("attempt.json") if path.is_file()})


def resolve_scope_root(parser: Any, args: Any) -> Path:
    """The existing --run-dir/--bench-root directory a maintenance CLI operates on."""
    requested = args.run_dir if args.run_dir is not None else args.bench_root
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        parser.error(f"requested root is unavailable: {error}")
    if not root.is_dir():
        parser.error("requested root is not a directory")
    return root


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish a JSON object with rename, so polling readers never see a partial write.

    The temp name is unique, so concurrent writers from different processes cannot clobber it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    write_bytes_atomic(path, data, fsync=False)


def _lock_handle(handle: Any, blocking: bool) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return
    while True:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as error:
            if not blocking:
                raise BlockingIOError(str(error)) from error
            time.sleep(0.05)


def _unlock_handle(handle: Any) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def file_lock(path: Path, *, blocking: bool = True) -> Iterator[None]:
    """Hold an advisory process lock for the duration of the context."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        _lock_handle(handle, blocking)
        try:
            yield
        finally:
            _unlock_handle(handle)


@contextlib.contextmanager
def edit_json_locked(path: Path, lock_name: str = BATTERY_LOCK) -> Iterator[dict[str, Any]]:
    """Read-modify-write one JSON object while holding its sibling lock."""
    with file_lock(path.parent / lock_name):
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(payload, dict):
            raise ValueError(f"{path} is not a JSON object")
        yield payload
        write_json_atomic(path, payload)


def json_lock_name(path: Path) -> str:
    """The per-file sibling lock guarding read-modify-writes of `path`."""
    return f".{path.name}.lock"


def patch_json(
    path: Path,
    fields: dict[str, Any] | None = None,
    *,
    drop: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Locked merge of `fields` (after removing `drop`) into one JSON object; returns it."""
    with edit_json_locked(path, json_lock_name(path)) as payload:
        for key in drop:
            payload.pop(key, None)
        payload.update(fields or {})
        return dict(payload)


def read_jsonl_rows(path: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    """Read JSON-object lines, optionally rejecting malformed content."""
    if not path.exists():
        return []
    return _parse_jsonl_rows(path, path.read_text(encoding="utf-8"), strict=strict)


def _parse_jsonl_rows(path: Path, text: str, *, strict: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            if strict:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            continue
        if not isinstance(row, dict):
            if strict:
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            continue
        rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False, default=str)}\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temp, path)


def canonical_attempt_rows(
    output_dir: Path,
    *,
    strict: bool = False,
    rewrite: bool = False,
) -> list[dict[str, Any]]:
    """Return the latest row per logical attempt, optionally compacting the file."""
    path = output_dir / "attempts.jsonl"
    with file_lock(output_dir / ATTEMPTS_LOCK):
        exists = path.exists()
        text = path.read_text(encoding="utf-8") if exists else ""
        rows = _parse_jsonl_rows(path, text, strict=strict)
        latest: dict[tuple[str, int], dict[str, Any]] = {}
        order: list[tuple[str, int]] = []
        for index, row in enumerate(rows, 1):
            prompt_id = str(row.get("prompt_id") or "")
            try:
                attempt = int(row.get("attempt"))
            except (TypeError, ValueError) as error:
                if strict:
                    raise ValueError(
                        f"{path}: row {index} has an invalid attempt number"
                    ) from error
                continue
            if not prompt_id or attempt < 1:
                if strict:
                    raise ValueError(f"{path}: row {index} has no valid logical attempt key")
                continue
            key = (prompt_id, attempt)
            if key not in latest:
                order.append(key)
            latest[key] = row
        canonical = [latest[key] for key in order]
        if rewrite and (rows != canonical or (exists and not text.endswith("\n"))):
            write_jsonl_atomic(path, canonical)
        return canonical


def upsert_attempt_row(output_dir: Path, row: dict[str, Any]) -> None:
    """Replace one logical attempt row atomically, preserving other rows' order."""
    prompt_id = str(row.get("prompt_id") or "")
    attempt = int(row.get("attempt") or 0)
    if not prompt_id or attempt < 1:
        raise ValueError("attempt result has no valid prompt_id/attempt key")

    path = output_dir / "attempts.jsonl"
    with file_lock(output_dir / ATTEMPTS_LOCK):
        rows = read_jsonl_rows(path)
        replacement = (prompt_id, attempt)
        kept: list[dict[str, Any]] = []
        replaced = False
        for existing in rows:
            try:
                existing_attempt = int(existing.get("attempt") or 0)
            except (TypeError, ValueError):
                existing_attempt = 0
            key = (str(existing.get("prompt_id") or ""), existing_attempt)
            if key == replacement:
                if not replaced:
                    kept.append(row)
                    replaced = True
                continue
            kept.append(existing)
        if not replaced:
            kept.append(row)
        write_jsonl_atomic(path, kept)


def purge_attempt_rows(output_dir: Path, prompt_id: str, attempt: int) -> None:
    path = output_dir / "attempts.jsonl"
    with file_lock(output_dir / ATTEMPTS_LOCK):
        rows = read_jsonl_rows(path)
        kept: list[dict[str, Any]] = []
        for row in rows:
            try:
                row_attempt = int(row.get("attempt") or 0)
            except (TypeError, ValueError):
                row_attempt = 0
            if str(row.get("prompt_id") or "") == prompt_id and row_attempt == attempt:
                continue
            kept.append(row)
        if path.exists():
            write_jsonl_atomic(path, kept)
