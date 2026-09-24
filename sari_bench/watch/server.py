"""The live benchmark dashboard: `python -m sari_bench watch`.

Runs beside the RUNNER, not the coordinator. Screenshots and step logs are written by agent
subprocesses the runner spawns, so they are on the runner's local disk; the coordinator is allowed
to be a third machine entirely.

It reads the filesystem, and optionally opens ONE read-only connection to the coordinator's /bench
route to show the sandbox pool. That connection only ever sends `bench.status`, never
`bench.acquire`, so it cannot take a lease away from a worker.

Stdlib only - ThreadingHTTPServer plus one HTML file, polled. This is a handful of tiles refreshing
every couple of seconds; it does not warrant a web framework, and pyproject.toml is deliberately
kept to what is actually imported.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gzip
import json
import os
import shutil
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from sari_bench import capture, storage, video
from sari_bench.protocol import DEFAULT_COORDINATOR_PORT, STATE_READY
from sari_bench.storage import edit_json_locked
from sari_bench.watch import health, notify, replay as replay_mod, scan
from sari_watchconfig import WatchConfigError, load_watch_config

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BENCH_ROOT = REPO_ROOT / "bench_runs"
STATIC_DIR = Path(__file__).resolve().parent / "static"

POOL_REFRESH_SECONDS = 5.0
# Notifications must not depend on a browser keeping `/api/state` open. This is separate from the
# coordinator poller because an unavailable coordinator must never stop filesystem-based watching.
WATCH_POLL_SECONDS = 1.0
LOG_TAIL_LINES = 25
LOG_MAX_LINES = 2000
LOG_BOOTSTRAP_BYTES = 16_384
GZIP_MIN_BYTES = 1024
# Same string the runner stamps as a cancelled try's end_reason and the scanner classifies as the
# `already_successful` verdict; one definition so the three can never drift.
ALREADY_SUCCESSFUL = scan.ALREADY_SUCCESSFUL
# How long the watcher waits for a runner to close an attempt out before deciding nobody will.
FINALIZE_GRACE_SECONDS = 3.0
# The manifest records the agent subprocess, which exits before its runner has drained and
# finalized the continuous encoder. Do not let a replay request race that bounded cleanup. The
# extra margin also covers the manifest write which follows it.
REPLAY_FINALIZE_GRACE_SECONDS = capture.ENCODER_GRACE_SECONDS + 5.0
# The end_reason written for an attempt nobody recorded: not something the agent decided, and
# deliberately not blank, so a row that exists only because the watcher adopted it says so.
RUNNER_GONE = "runner_gone"
# Every field a verdict writes, so un-reviewing an attempt leaves nothing of it behind. Listed once:
# a clear that missed one field would leave the row looking judged by a ghost.
VERDICT_FIELDS = (
    "verified_verdict", "verified_success", "verified_at", "verified_by", "verified_note",
)
FLEET_FIELDS = (
    "capacity_limit", "effective_capacity", "active_leases",
    "registered_sandboxes", "eligible_sandboxes", "quarantined_sandboxes",
)
COORDINATOR_TIMEOUT_SECONDS = 10.0


def _log(message: str) -> None:
    print(f"[sari-bench watch] {message}", flush=True)


class WatchState:
    """Everything the HTTP handlers read. Rebuilt on demand, at most once per `min_interval`."""

    def __init__(
        self,
        *,
        bench_root: Path,
        fixed_battery: Path | None,
        discord: notify.Discord,
        replay: replay_mod.ReplayNotifier | None = None,
        backfill: bool = False,
        min_interval: float = 1.0,
        coordinator_url: str | None = None,
        retry_agent_entry: str | None = None,
        retry_agent_cwd: Path | None = None,
        retry_ocr_health_check: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.bench_root = bench_root
        self.fixed_battery = fixed_battery
        self.discord = discord
        self.replay = replay
        self.backfill = backfill
        self.min_interval = min_interval
        self.coordinator_url = coordinator_url
        self.retry_agent_entry = retry_agent_entry
        self.retry_agent_cwd = retry_agent_cwd
        self.retry_ocr_health_check = retry_ocr_health_check
        self._lock = threading.Lock()
        self._notify_lock = threading.Lock()
        self._seeded = False
        self._cached: dict[str, Any] = {}
        self._cached_at = 0.0
        # Read-only views of batteries this watcher is not the one following, one cache entry per
        # battery id. A browser can look at any run on disk without moving what the server watches.
        self._views: dict[str, tuple[float, dict[str, Any]]] = {}
        self._pool: list[dict[str, Any]] = []
        self._pool_error = ""
        # Acquires parked at the coordinator, from the last pool poll. Includes the battery runner's
        # own workers, which is the point: a retry queues behind them and nothing else can say so.
        self._pool_waiting = 0
        self._fleet_status: dict[str, Any] = {
            name: None if name == "capacity_limit" else 0 for name in FLEET_FIELDS
        }
        # (monotonic time, find_batteries result), reused within one poll window.
        self._found: tuple[float, list[Path]] = (float("-inf"), [])
        self._announced_start = False
        self._announced_done = False
        self.battery: Path | None = None
        self._retry_jobs: dict[str, dict[str, Any]] = {}
        self._runner_retries: dict[tuple[str, str], dict[str, Any]] = {}

    def resolve_battery(self) -> Path | None:
        """The `--run-dir` pin, else the newest battery under `bench_root`."""
        if self.fixed_battery is not None:
            return self.fixed_battery
        batteries = self._batteries()
        return batteries[0] if batteries else None

    def _batteries(self) -> list[Path]:
        """`find_batteries`, cached briefly: every request and poll resolves a battery through it."""
        now = time.monotonic()
        found_at, found = self._found
        if now - found_at >= min(self.min_interval, 1.0):
            found = scan.find_batteries(self.bench_root)
            self._found = (now, found)
        return found

    def battery_for(self, battery_id: str | None) -> Path | None:
        """The battery one request names, matched by dir name (never joined onto a path).

        Unknown ids fall back to the watched battery; `newest` opts out of a `--run-dir` pin.
        """
        if not battery_id:
            return self.resolve_battery()
        batteries = self._batteries()
        if battery_id == "newest":
            return batteries[0] if batteries else None
        for battery in batteries:
            if battery.name == battery_id:
                return battery
        return self.resolve_battery()

    def rename_battery(self, battery_id: str, new_name: str) -> dict[str, Any]:
        """Renames a battery's directory (its id). Unknown ids are errors, never a fallback.

        Refused while a runner or retry is working: they hold absolute paths into the battery.
        """
        wanted = storage.battery_dir_name(new_name)
        if not wanted:
            return {"ok": False, "error": "a name needs letters, digits, dot, dash or underscore"}

        with self._lock:
            match = next(
                (b for b in scan.find_batteries(self.bench_root) if b.name == battery_id), None
            )
            if match is None:
                return {"ok": False, "error": f"no bench run named {battery_id!r}"}
            if match.name == wanted:
                return {"ok": True, "battery_id": wanted, "renamed": False}
            target = self.bench_root / wanted
            if target.exists():
                return {"ok": False, "error": f"{wanted} already exists under bench_runs/"}
            if scan.battery_live(match):
                return {"ok": False, "error": "its runner is still working - stop it first"}
            busy = next(
                (key for key, job in self._retry_jobs.items()
                 if job.get("battery_id") == battery_id and job.get("state") != "failed"),
                None,
            )
            if busy:
                return {"ok": False, "error": f"a retry is still {self._retry_jobs[busy]['state']} "
                                              f"{busy} in this run"}
            try:
                match.rename(target)
            except OSError as error:
                return {"ok": False, "error": f"{error!r}"}

            # Carry the paths across so `snapshot` does not read the rename as a new battery.
            if self.fixed_battery == match:
                self.fixed_battery = target
            if self.battery == match:
                self.battery = target
            for job in self._retry_jobs.values():
                if job.get("battery_id") == battery_id:
                    job["battery_id"] = wanted
            # Re-derived under the new id on its next scan.
            for identity in [i for i in self._runner_retries if i[0] == battery_id]:
                self._runner_retries.pop(identity)
            self._invalidate_locked()
        _log(f"renamed bench run {battery_id} -> {wanted}")
        return {"ok": True, "battery_id": wanted, "renamed": True}

    def _invalidate_locked(self) -> None:
        """Drops every cached payload. The caller must hold `self._lock`."""
        self._cached_at = 0.0
        self._views.clear()
        self._found = (float("-inf"), [])

    def view(self, battery_id: str | None = None) -> dict[str, Any]:
        """`/api/state` for one battery. Only the watched one goes through `snapshot` (and notifies)."""
        battery = self.battery_for(battery_id)
        watched = self.resolve_battery()
        if battery is None or battery == watched:
            return self.snapshot()

        # Still rebuilt for its notification side effects; only the payload is discarded.
        self.snapshot()

        with self._lock:
            now = time.time()
            cached_at, cached = self._views.get(battery.name, (0.0, {}))
            if cached and now - cached_at < self.min_interval:
                return cached
            view = self._scan_locked(
                battery, now, mode="pinned", watching_id=watched.name if watched else None
            )
            self._views[battery.name] = (now, view)
            return view

    def _scan_locked(
        self, battery: Path, now: float, *, mode: str, watching_id: str | None
    ) -> dict[str, Any]:
        """One battery's `/api/state` payload. The caller must hold `self._lock`."""
        # Every battery on disk, pinned or not: the header's picker reaches runs the server was not
        # started on.
        view = scan.scan_battery(battery, now, discovered=self._batteries()).as_dict()
        self._sync_runner_retries(view, battery.name)
        view.update({
            "pool": self._pool,
            "pool_error": self._pool_error,
            "fleet": dict(self._fleet_status),
            "queue": self._queue_locked(),
            "now": now,
            "bench_root": str(self.bench_root),
            "mode": mode,
            "watching_id": watching_id,
        })
        self._merge_retry_jobs(view, battery.name)
        return view

    def snapshot(self, force: bool = False) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            if not force and self._cached and now - self._cached_at < self.min_interval:
                return self._cached

            battery = self.resolve_battery()
            if battery is None:
                self._cached = {
                    "battery_id": None,
                    "error": f"no battery found under {self.bench_root}",
                    "attempts": [],
                    "counts": {},
                    "pool": self._pool,
                    "pool_error": self._pool_error,
                    "fleet": dict(self._fleet_status),
                    "queue": self._queue_locked(),
                    "discovered": [],
                    "watching_id": None,
                    "bench_root": str(self.bench_root),
                    "now": now,
                }
                self._cached_at = now
                return self._cached

            if battery != self.battery:
                _log(f"watching battery {battery}")
                self.battery = battery
                self._announced_start = False
                self._announced_done = False
                self._seeded = False

            view = self._scan_locked(
                battery, now, mode="pinned" if self.fixed_battery else "auto",
                watching_id=battery.name,
            )
            self._cached = view
            self._cached_at = now

        self._notify(view)
        return view

    def _merge_retry_jobs(self, view: dict[str, Any], battery_id: str | None = None) -> None:
        """Keeps a logical try visible while its old directory is gone and replacement is queued."""
        attempts = view.get("attempts")
        if not isinstance(attempts, list):
            attempts = []
            view["attempts"] = attempts
        by_key = {str(attempt.get("key") or ""): attempt for attempt in attempts}
        for key, job in self._retry_jobs.items():
            # Retries are tracked for the whole process but belong to one run; a placeholder card
            # must not surface in a different battery's grid just because the keys collide.
            if battery_id and job.get("battery_id") and job["battery_id"] != battery_id:
                continue
            current = by_key.get(key)
            logical = job["attempt"]
            related = [
                attempt for attempt in attempts
                if attempt.get("prompt_id") == logical.get("prompt_id")
                and attempt.get("attempt") == logical.get("attempt")
            ]
            for attempt in related:
                attempt["retry_state"] = job["state"]
                attempt["retry_error"] = job.get("error", "")
            if current is not None:
                continue
            placeholder = dict(job["attempt"])
            placeholder.update({
                "key": key,
                "run_id": job["run_id"],
                "state": "retrying",
                "retry_state": job["state"],
                "retry_error": job.get("error", ""),
                "alive": False,
                "verifiable": False,
                "verified": False,
                "frame": "",
                "log_bytes": 0,
            })
            attempts.append(placeholder)

    def _sync_runner_retries(self, view: dict[str, Any], battery_id: str) -> None:
        """Mirror durable automatic retries into the same queue model as watcher retries."""
        for identity in [item for item in self._runner_retries if item[0] == battery_id]:
            self._runner_retries.pop(identity, None)
        winners = (view.get("battery") or {}).get("human_verified_winners") or {}
        for attempt in view.get("attempts") or []:
            if not attempt.get("pending_retry"):
                continue
            if str(attempt.get("prompt_id") or "") in winners:
                continue
            key = f"{attempt.get('prompt_id')}/try{int(attempt.get('attempt') or 0):02d}"
            self._runner_retries[(battery_id, key)] = {
                "key": key,
                "battery_id": battery_id,
                "prompt_id": str(attempt.get("prompt_id") or ""),
                "attempt": int(attempt.get("attempt") or 0),
                "kind": "automatic_retry",
                "state": "waiting",
                "error": str(attempt.get("retry_wait_reason") or ""),
                "waiting": True,
                "position": None,
                "since": attempt.get("retry_queued_at") or 0.0,
            }

    def _notify(self, view: dict[str, Any]) -> None:
        """Queues finished-run replays and diffs notifications outside the snapshot lock."""
        # One completion pass at a time. Two concurrent /api/state polls could otherwise race while
        # claiming the same render or notification. Non-blocking is safe because the next poll
        # re-derives the same durable filesystem state.
        if not self._notify_lock.acquire(blocking=False):
            return
        try:
            attempts = view.get("attempts") or []
            if self.discord.enabled and not self._announced_start and attempts:
                self._announced_start = True
                self.discord.battery_started(view, len(self._pool))

            if not self._seeded:
                self._seeded = True
                if self.replay is not None and not self.backfill:
                    self.replay.seed(attempts)

            for attempt in attempts:
                if attempt.get("state") == "finished":
                    if attempt.get("end_reason") == ALREADY_SUCCESSFUL:
                        # Administrative sibling cancellation is bookkeeping, not an attempt halt:
                        # do not spend encoder capacity or post one Discord message per skipped try.
                        if self.discord.enabled:
                            self.discord.suppress_finished([attempt.get("key") or ""])
                        continue
                    # Give a time-sensitive notification first place in the serial worker's queue.
                    # It only declines when posting is off or backed up, in which case announce the
                    # halt here without a clip.
                    if self.discord.enabled and (
                        self.replay is None or not self.replay.submit(attempt)
                    ):
                        self.discord.attempt_finished(attempt)
                    # Full dashboard replays are prepared as soon as a run finishes, independently
                    # of Discord. `enqueue` deduplicates repeated state polls and never blocks here.
                    key = str(attempt.get("key") or "")
                    run_dir_value = attempt.get("run_dir")
                    if self.replay is not None and key and run_dir_value:
                        self.replay.enqueue(key, Path(str(run_dir_value)))
                elif self.discord.enabled and (
                    (attempt.get("health") or {}).get("level") == health.LEVEL_ALERT
                ):
                    frame = attempt.get("frame")
                    path = (self.battery / frame) if (self.battery and frame) else None
                    self.discord.collapse(attempt, path)

            planned = (view.get("battery") or {}).get("planned_attempts")
            finished = sum(1 for a in attempts if a.get("state") in {"finished", "requeued"})
            live = sum(1 for a in attempts if a.get("state") in {"starting", "running"})
            if (self.discord.enabled and planned and finished >= planned and not live
                    and not self._announced_done):
                self._announced_done = True
                self.discord.battery_finished(view)
        finally:
            self._notify_lock.release()

    def set_pool(
        self,
        pool: list[dict[str, Any]],
        error: str = "",
        waiting: int = 0,
        fleet: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._pool = pool
            self._pool_error = error
            self._pool_waiting = waiting
            if fleet is not None:
                self._fleet_status = {name: fleet.get(name) for name in FLEET_FIELDS}
            self._invalidate_locked()

    def set_fleet_cap(self, limit: Any) -> dict[str, Any]:
        """Update the coordinator's process-local active lease cap."""
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            return {"ok": False, "error": "limit must be null or a non-negative integer"}
        if not self.coordinator_url:
            return {"ok": False, "error": "watcher has no coordinator configured"}
        try:
            status = _coordinator_call(self.coordinator_url, lambda c: c.set_capacity(limit))
        except Exception as error:  # noqa: BLE001 - report the coordinator failure to the caller
            return {"ok": False, "error": repr(error)}
        with self._lock:
            self._fleet_status.update(status)
            self._pool_error = ""
            self._invalidate_locked()
        return {"ok": True, **status}

    def fleet_status(self) -> dict[str, Any]:
        """Stable JSON backend for a dedicated sandbox-status UI."""
        with self._lock:
            return {
                "sandboxes": list(self._pool),
                "waiting": self._pool_waiting,
                "error": self._pool_error,
                **dict(self._fleet_status),
            }

    def change_quarantine(
        self, selector: str, *, clear: bool, reason: str = "web_ui"
    ) -> dict[str, Any]:
        if not selector:
            return {"ok": False, "error": "sandbox alias or ID is required"}
        if not self.coordinator_url:
            return {"ok": False, "error": "watcher has no coordinator configured"}
        try:
            status = _coordinator_call(
                self.coordinator_url,
                lambda c: c.unquarantine(selector) if clear
                else c.quarantine_sandbox(selector, reason=reason, source="watch_api"),
            )
        except Exception as error:  # noqa: BLE001
            return {"ok": False, "error": repr(error)}
        return {"ok": True, **status}

    # Queue visibility covers watcher retries and coordinator wait counts.
    # The battery's pending jobs live in another process and have no run directory
    # until leased, so their order is unavailable here.

    # Retry states that are not yet holding a sandbox, in the order `_retry_worker` moves through
    # them. "running" has one; "failed" has stopped needing one.
    _QUEUE_WAITING_STATES = ("stopping", "cleaning", "waiting")

    def _free_sandboxes_locked(self) -> int:
        """Sandboxes the coordinator would hand out right now - its own `Sandbox.leasable` rule."""
        free = sum(
            1 for sandbox in self._pool
            if not sandbox.get("lease_id")
            and sandbox.get("state") == STATE_READY
            and bool(sandbox.get("store_loaded", True))
            and not sandbox.get("quarantined")
        )
        limit = self._fleet_status.get("capacity_limit")
        if limit is None:
            return free
        try:
            headroom = max(0, int(limit) - int(self._fleet_status.get("active_leases") or 0))
        except (TypeError, ValueError):
            return free
        return min(free, headroom)

    def _queue_locked(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        waiting_count = 0
        for key, job in self._retry_jobs.items():
            waiting = job["state"] in self._QUEUE_WAITING_STATES
            if waiting:
                waiting_count += 1
            entries.append({
                "key": key,
                "battery_id": job.get("battery_id", ""),
                "prompt_id": str(job["attempt"].get("prompt_id") or ""),
                "attempt": job["attempt"].get("attempt"),
                "kind": "retry",
                "state": job["state"],
                "error": job.get("error", ""),
                "waiting": waiting,
                # 1-based among the waiting entries, and absent for the ones that are not waiting:
                # a number beside a running row would read as a place in a line it has already left.
                "position": None,
                "since": job.get("queued_at", 0.0),
            })
        forced = {(entry["battery_id"], entry["key"]) for entry in entries}
        for identity, entry in self._runner_retries.items():
            if identity in forced:
                continue
            entries.append(dict(entry))
            waiting_count += 1
        return {
            "entries": entries,
            "waiting": waiting_count,
            "running": sum(1 for entry in entries if entry["state"] == "running"),
            "failed": sum(1 for entry in entries if entry["state"] == "failed"),
            "free_sandboxes": self._free_sandboxes_locked(),
            "coordinator_waiting": self._pool_waiting,
            "pool_error": self._pool_error,
        }

    def queue(self) -> dict[str, Any]:
        with self._lock:
            return self._queue_locked()

    def _placement_locked(self, key: str) -> dict[str, Any]:
        """Return a conservative availability hint without inventing a global queue position."""
        queue = self._queue_locked()
        return {
            "position": None,
            "ahead": None,
            "free_sandboxes": queue["free_sandboxes"],
            "coordinator_waiting": queue["coordinator_waiting"],
            "pool_error": queue["pool_error"],
            "immediate": bool(
                not self._pool_error
                and queue["free_sandboxes"] > 0
                and queue["coordinator_waiting"] == 0
            ),
        }

    # -- actions ---------------------------------------------------------------------------

    def kill(self, key: str, *, battery_id: str | None = None) -> dict[str, Any]:
        """SIGTERMs the agent's process group; its runner records the attempt and frees the lease.

        `killed_by` keeps the row from scoring as a crash. A dead pid closes the attempt out instead.
        """
        battery, run_dir = self.run_dir_for(key, battery_id)
        if battery is None:
            return {"ok": False, "error": "no battery"}
        if run_dir is None:
            return {"ok": False, "error": "unknown attempt"}

        manifest_path = run_dir / scan.ATTEMPT_MANIFEST
        manifest = scan._read_json(manifest_path)
        pid = manifest.get("pid")
        if manifest.get("state") == "finished":
            return {"ok": False, "error": "attempt already finished"}
        if not pid:
            return {"ok": False, "error": "no pid recorded"}

        # An orphan: nothing to signal, so adopt the attempt instead of stranding the tile.
        if not scan.agent_is_alive(manifest, pid):
            return self._close_out_abandoned(battery, key, run_dir, manifest)

        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError) as error:
            return {"ok": False, "error": f"{error!r}"}

        _stamp(manifest_path, {"killed_by": "watcher",
                               "killed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        _log(f"killed {key} (pid {pid}); the runner will release its lease")
        return {"ok": True, "pid": pid}

    def _close_out_abandoned(
        self, battery: Path, key: str, run_dir: Path, manifest: dict[str, Any]
    ) -> dict[str, Any]:
        """Writes the terminal record an abandoned (orphaned) attempt's runner never wrote.

        A slow runner gets FINALIZE_GRACE_SECONDS first, and still wins if it lands later: both paths
        merge into the manifest and upsert the same attempts.jsonl key.
        """
        from dataclasses import asdict

        from sari_bench.runner import AttemptResult  # deferred: watch starts without the runner
        from sari_bench.storage import upsert_attempt_row

        prompt_id = str(manifest.get("prompt_id") or run_dir.parent.name)
        attempt = _try_number(manifest, run_dir) or 0
        if attempt < 1:
            # attempts.jsonl is keyed on (prompt_id, attempt); without one there is nothing to record.
            return {"ok": False, "error": "attempt has no valid try number"}
        with self._lock:
            job = self._retry_jobs.get(f"{prompt_id}/try{attempt:02d}")
        if job is not None and job.get("state") != "error":
            # A retry runs its runner inside this process and owns this try's manifest; let it finish.
            return {"ok": False, "error": f"a retry is already {job['state']} this attempt"}

        manifest_path = run_dir / scan.ATTEMPT_MANIFEST
        deadline = time.monotonic() + FINALIZE_GRACE_SECONDS
        while True:
            manifest = scan._read_json(manifest_path)
            if manifest.get("state") == "finished":
                return {"ok": True, "closed_out": False,
                        "note": "the runner recorded the attempt itself"}
            if time.monotonic() >= deadline:
                break
            time.sleep(0.25)

        summary = scan._read_json(run_dir / "summary.json")
        legs = summary.get("legs") if isinstance(summary.get("legs"), list) else []
        end_reasons = [str(leg.get("end_reason") or "") for leg in legs if isinstance(leg, dict)]
        # Same chain the runner uses: the agent's summary when it wrote one, else the tokens.json it
        # rewrites as it goes - which is the only place a killed attempt's cost survives at all.
        tokens = summary.get("tokens") if isinstance(summary.get("tokens"), dict) else (
            summary if "tokens_in" in summary else scan._read_json(run_dir / "tokens.json"))
        started = manifest.get("started_epoch")
        ended = _last_sign_of_life(run_dir)
        stamped = time.strftime("%Y-%m-%dT%H:%M:%S")

        result = AttemptResult(
            prompt_id=prompt_id,
            attempt=attempt,
            prompt=str(manifest.get("prompt") or ""),
            family=str(manifest.get("family") or ""),
            # The outcome the runner itself would have recorded for a dashboard kill, so a killed
            # orphan reads exactly like a killed attempt whose runner survived to say so.
            outcome="operator_kill" if manifest.get("killed_by") else "orphaned",
            context_policy=str(
                manifest.get("context_policy")
                or (summary.get("run_config") or {}).get("context_policy")
                or scan._read_json(battery / scan.BATTERY_MANIFEST).get("context_policy")
                or "baseline"
            ),
            success=bool(summary.get("success")),
            end_reason=end_reasons[-1] if end_reasons else RUNNER_GONE,
            sandbox_id=str(manifest.get("sandbox_id") or ""),
            sandbox_alias=str(manifest.get("sandbox_alias") or ""),
            lease_alias=str(manifest.get("lease_alias") or ""),
            commands_uri=str(manifest.get("commands_uri") or ""),
            exit_code=None,
            wall_seconds=(max(0.0, round(ended - float(started), 1))
                          if isinstance(started, (int, float)) else 0.0),
            run_dir=str(run_dir),
            error="its runner exited without recording this attempt",
            winning_attempt_key=str(manifest.get("winning_attempt_key") or ""),
            legs={"planned": summary.get("legs_planned"),
                  "completed": summary.get("legs_completed"),
                  "end_reasons": end_reasons} if summary else {},
            tokens_in=int(tokens.get("tokens_in") or 0),
            tokens_out=int(tokens.get("tokens_out") or 0),
            api_calls=(int(tokens.get("api_calls") or 0)
                       if "api_calls" in tokens and tokens.get("api_calls") is not None else None),
            tokens_by_role=scan.normalize_by_role(tokens.get("by_role")),
            llm_calls=int(summary.get("llm_calls") or 0),
        )

        _stamp(manifest_path, {
            "state": "finished",
            "outcome": result.outcome,
            "context_policy": result.context_policy,
            "success": result.success,
            "end_reason": result.end_reason,
            "exit_code": None,
            "wall_seconds": result.wall_seconds,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "api_calls": result.api_calls,
            "tokens_by_role": result.tokens_by_role,
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ended)),
            "finalized_at": stamped,
            # So the record is never mistaken for one its runner produced: the runtime is measured to
            # the attempt's last sign of life rather than to its exit, and there is no exit code.
            "closed_out_by": "watcher",
            "closed_out_at": stamped,
        })
        upsert_attempt_row(battery, asdict(result))
        self._release_abandoned_lease(battery, manifest)
        with self._lock:
            self._invalidate_locked()
        _log(f"closed out {key} as {result.outcome}: its runner never recorded it")
        return {"ok": True, "closed_out": True, "outcome": result.outcome,
                "wall_seconds": result.wall_seconds}

    def _release_abandoned_lease(self, battery: Path, manifest: dict[str, Any]) -> None:
        """Best-effort release of the dead runner's lease; the coordinator reaps it anyway."""
        lease_id = str(manifest.get("lease_id") or "")
        plan = scan._read_json(battery / scan.BATTERY_MANIFEST)
        coordinator = str(plan.get("coordinator") or self.coordinator_url or "")
        if not lease_id or not coordinator:
            return
        try:
            known = _coordinator_call(
                coordinator,
                lambda c: c.release_lease_id(lease_id, outcome="watcher_close_out"),
            )
        except Exception as error:  # noqa: BLE001 - the sandbox is reclaimable without us
            _log(f"could not release abandoned lease {lease_id}: {error!r}")
            return
        _log(f"released abandoned lease {lease_id} ({'known' if known else 'already reaped'})")

    def retry(self, key: str, *, battery_id: str | None = None) -> dict[str, Any]:
        """Schedules a destructive replacement of one logical prompt/try."""
        battery = self.battery_for(battery_id)
        if battery is None:
            return {"ok": False, "error": "no battery"}
        selected = _safe_run_dir(battery, key)
        previous_error: dict[str, Any] | None = None
        if selected is None:
            with self._lock:
                candidate = self._retry_jobs.get(key)
                if candidate and candidate.get("state") == "failed":
                    previous_error = dict(candidate)
            if previous_error is None:
                return {"ok": False, "error": "unknown attempt"}
            manifest = dict(previous_error["source_manifest"])
            prompt_id = str(manifest.get("prompt_id") or "")
            attempt = int(manifest.get("attempt") or 0)
            canonical_key = f"{prompt_id}/try{attempt:02d}"
            if key != canonical_key:
                return {"ok": False, "error": "invalid retry key"}
            attempt_view = dict(previous_error["attempt"])
        else:
            manifest = scan._read_json(selected / scan.ATTEMPT_MANIFEST)
            prompt_id = str(manifest.get("prompt_id") or selected.parent.name)
            attempt = _try_number(manifest, selected)
            if attempt is None:
                return {"ok": False, "error": "attempt has no valid try number"}
            if attempt < 1 or prompt_id != selected.parent.name:
                return {"ok": False, "error": "invalid attempt metadata"}
            canonical_key = f"{prompt_id}/try{attempt:02d}"
            attempt_view = scan.scan_attempt(selected, battery, time.time()).as_dict()

        canonical_manifest = scan._read_json(
            battery / prompt_id / f"try{attempt:02d}" / scan.ATTEMPT_MANIFEST
        )
        if canonical_manifest.get("pending_retry"):
            return {
                "ok": False,
                "error": "battery runner retry already pending",
                "key": canonical_key,
            }

        with self._lock:
            existing = self._retry_jobs.get(canonical_key)
            if existing and existing.get("state") != "failed":
                return {"ok": False, "error": "retry already in progress", "key": canonical_key}
            # Accepting a retry deliberately reopens a previously won prompt. Do it while holding
            # the same lock verdict writes use, before cleanup starts, so a later sibling pass can
            # never be erased by the retry thread.
            self._clear_prompt_winner(battery, prompt_id)
            job = {
                "key": canonical_key,
                "battery_id": battery.name,
                "run_id": f"retry-{uuid.uuid4().hex}",
                "state": "stopping",
                "error": "",
                "attempt": attempt_view,
                "source_manifest": manifest,
                "queued_at": time.time(),
            }
            # Re-queued rather than overwritten in place: re-dispatching a job that errored an hour
            # ago must take its turn at the back, not inherit the position it had then.
            self._retry_jobs.pop(canonical_key, None)
            self._retry_jobs[canonical_key] = job
            placement = self._placement_locked(canonical_key)
            self._invalidate_locked()
            self._announced_done = False
            if self.replay is not None:
                self.replay.forget_attempt(canonical_key)
            else:
                self.discord.forget_attempt(canonical_key)

        threading.Thread(
            target=self._retry_worker,
            args=(battery, prompt_id, attempt, manifest, job["run_id"]),
            name=f"retry-{prompt_id}-{attempt}",
            daemon=True,
        ).start()
        _log(
            f"retry requested for {canonical_key}; prior execution will be archived on replacement"
            + ("; a sandbox appears available" if placement["immediate"] else "; waiting to dispatch")
        )
        return {
            "ok": True,
            "key": canonical_key,
            "retry_state": "stopping",
            "queue": placement,
        }

    def _set_retry_state(self, key: str, state: str, error: str = "") -> None:
        with self._lock:
            job = self._retry_jobs.get(key)
            if job is None:
                return
            job["state"] = state
            job["error"] = error
            self._invalidate_locked()

    def _retry_worker(
        self,
        battery: Path,
        prompt_id: str,
        attempt: int,
        source_manifest: dict[str, Any],
        retry_run_id: str,
    ) -> None:
        key = f"{prompt_id}/try{attempt:02d}"
        try:
            config = self._retry_config(battery, source_manifest)
            self._stop_logical_try(battery, prompt_id, attempt)
            self._set_retry_state(key, "cleaning")

            from sari_bench.runner import (
                BenchmarkRunner,
                ORCHESTRATOR_ENTRY,
                OVERHAUL_DIR,
                Prompt,
                purge_attempt_records,
            )

            purge_attempt_records(battery, prompt_id, attempt)
            with contextlib.suppress(OSError):
                (battery / "summary.json").unlink()

            self._set_retry_state(key, "waiting")
            runner = BenchmarkRunner(
                prompts=[Prompt(
                    id=prompt_id,
                    prompt=str(source_manifest.get("prompt") or ""),
                    family=str(source_manifest.get("family") or ""),
                    looking_for=str(source_manifest.get("looking_for") or ""),
                )],
                coordinator_url=config["coordinator"],
                output_dir=battery,
                tries=max(attempt, 1),
                time_limit_minutes=config["time_limit_minutes"],
                per_leg_minutes=config["per_leg_minutes"],
                concurrency=1,
                max_steps=config["max_steps"],
                arm=config["arm"],
                map_dir=config["map_dir"],
                leg_retries=config["leg_retries"],
                completion_guard=config["completion_guard"],
                refusal_cap_action=config["refusal_cap_action"],
                adaptive_leg_replanning=config["adaptive_leg_replanning"],
                context_policy=config["context_policy"],
                api_max_attempts=config["api_max_attempts"],
                max_api_requeues=config["max_api_requeues"],
                timeout_grace=config["timeout_grace"],
                sandbox_startup_timeout=config["sandbox_startup_timeout"],
                lease_acquire_timeout=config["lease_acquire_timeout"],
                sandbox_command_timeout=config["sandbox_command_timeout"],
                capture_interval=config["capture_interval"],
                python_executable=sys.executable,
                agent_entry=self.retry_agent_entry or ORCHESTRATOR_ENTRY,
                agent_cwd=self.retry_agent_cwd or OVERHAUL_DIR,
                work_items=[(prompt_id, attempt)],
                retry_work_items=True,
                initialize_battery=False,
                ocr_url=config["ocr_url"],
                attempt_started_callback=lambda started_key, _run_dir: self._set_retry_state(
                    started_key, "running"
                ),
                **(
                    {"ocr_health_check": self.retry_ocr_health_check}
                    if self.retry_ocr_health_check is not None
                    else {}
                ),
            )
            asyncio.run(runner.run())
        except Exception as error:  # noqa: BLE001 - surfaced on the tile; watcher stays alive
            _log(f"retry failed for {key}: {error!r}")
            self._set_retry_state(key, "failed", repr(error))
            return

        with self._lock:
            current = self._retry_jobs.get(key)
            if current and current.get("run_id") == retry_run_id:
                self._retry_jobs.pop(key, None)
            self._invalidate_locked()
        _log(f"retry completed for {key}")

    def _retry_config(
        self, battery: Path, source_manifest: dict[str, Any]
    ) -> dict[str, Any]:
        plan = scan._read_json(battery / scan.BATTERY_MANIFEST)
        command = source_manifest.get("command")
        command = command if isinstance(command, list) else []

        def option(name: str, default: Any) -> Any:
            try:
                return command[command.index(name) + 1]
            except (ValueError, IndexError):
                return default

        coordinator = str(plan.get("coordinator") or self.coordinator_url or "")
        if not coordinator:
            raise RuntimeError("battery does not record its coordinator")
        return {
            "coordinator": coordinator,
            "time_limit_minutes": float(
                source_manifest.get("time_limit_minutes")
                or plan.get("time_limit_minutes")
                or 40.0
            ),
            "per_leg_minutes": float(
                source_manifest.get("per_leg_minutes")
                or plan.get("per_leg_minutes")
                or option("--max-minutes", 40.0)
            ),
            "max_steps": int(
                source_manifest.get("max_steps")
                or plan.get("max_steps")
                or option("--max-steps", 150)
            ),
            "arm": str(source_manifest.get("arm") or plan.get("arm") or "graph"),
            "map_dir": plan.get("map_dir") or option("--output-dir", None),
            "leg_retries": int(plan.get("leg_retries") or option("--leg-retries", 1)),
            "completion_guard": str(
                source_manifest.get("completion_guard")
                or plan.get("completion_guard")
                or option("--completion-guard", "deterministic")
            ),
            "context_policy": str(
                source_manifest.get("context_policy")
                or plan.get("context_policy")
                or option("--context-policy", "baseline")
            ),
            # Batteries predating the option aborted the attempt on the refusal cap; a retry of one
            # must keep doing that rather than inherit today's default.
            "refusal_cap_action": str(
                plan.get("refusal_cap_action")
                or option("--refusal-cap-action", "halt")
            ),
            "adaptive_leg_replanning": bool(
                source_manifest.get("adaptive_leg_replanning")
                if source_manifest.get("adaptive_leg_replanning") is not None
                else plan.get("adaptive_leg_replanning")
                if plan.get("adaptive_leg_replanning") is not None
                else "--adaptive-leg-replanning" in command
            ),
            "api_max_attempts": int(
                source_manifest.get("api_max_attempts")
                or plan.get("api_max_attempts")
                or option("--api-max-attempts", 10)
            ),
            "max_api_requeues": int(
                source_manifest.get("max_api_requeues")
                if source_manifest.get("max_api_requeues") is not None
                else plan.get("max_api_requeues")
                if plan.get("max_api_requeues") is not None
                else option("--max-api-requeues", 3)
            ),
            "ocr_url": str(
                source_manifest.get("ocr_url")
                or plan.get("ocr_url")
                or option("--ocr-url", "http://127.0.0.1:9100")
            ),
            "timeout_grace": float(plan.get("timeout_grace_seconds") or 120.0),
            "sandbox_startup_timeout": max(
                1.0, float(plan.get("sandbox_startup_timeout_seconds") or 30.0)
            ),
            "lease_acquire_timeout": max(
                0.001, float(plan.get("lease_acquire_timeout_seconds") or 30.0)
            ),
            "sandbox_command_timeout": float(
                source_manifest.get("sandbox_command_timeout")
                if source_manifest.get("sandbox_command_timeout") is not None
                else plan.get("sandbox_command_timeout_seconds", 10.0)
            ),
            "capture_interval": float(
                source_manifest.get("capture_interval_seconds")
                if source_manifest.get("capture_interval_seconds") is not None
                else plan.get("capture_interval_seconds", capture.DEFAULT_INTERVAL_SECONDS)
            ),
        }

    def _stop_logical_try(self, battery: Path, prompt_id: str, attempt: int) -> None:
        canonical = battery / prompt_id / f"try{attempt:02d}"
        manifest_path = canonical / scan.ATTEMPT_MANIFEST
        manifest = scan._read_json(manifest_path)
        pid = manifest.get("pid")
        was_live = manifest.get("state") in {"starting", "running"}
        was_orphaned = bool(was_live and pid and not scan.agent_is_alive(manifest, pid))
        if was_live:
            _stamp(manifest_path, {
                "retry_requested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "killed_by": "watcher_retry",
            })
        if was_orphaned:
            return

        deadline = time.monotonic() + 45.0
        signalled = False
        while was_live and time.monotonic() < deadline:
            manifest = scan._read_json(manifest_path)
            pid = manifest.get("pid")
            if scan.agent_is_alive(manifest, pid):
                if not signalled:
                    try:
                        os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    signalled = True
                time.sleep(0.1)
                continue
            if manifest.get("finalized_at") or manifest.get("state") == "finished":
                return
            time.sleep(0.1)
        if was_live and scan.agent_is_alive(manifest, pid):
            raise RuntimeError("agent did not stop within 45 seconds")

    def _clear_prompt_winner(self, battery: Path, prompt_id: str) -> None:
        battery_path = battery / scan.BATTERY_MANIFEST
        winner_key = ""
        with edit_json_locked(battery_path) as plan:
            winners = plan.get("human_verified_winners")
            if isinstance(winners, dict):
                winner = winners.pop(prompt_id, None)
                if isinstance(winner, dict):
                    winner_key = str(winner.get("winning_attempt_key") or "")
                plan["human_verified_winners"] = winners
        if winner_key:
            winner_dir = _safe_run_dir(battery, winner_key)
            if winner_dir is not None:
                _clear_verdict_fields(
                    winner_dir / scan.ATTEMPT_MANIFEST,
                    scan._read_json(winner_dir / scan.ATTEMPT_MANIFEST),
                )
        # Older batteries may have only the per-attempt verdict and no battery-level winner map.
        prompt_dir = battery / prompt_id
        if prompt_dir.is_dir():
            for run_dir in prompt_dir.iterdir():
                if not run_dir.is_dir() or ".requeue" in run_dir.name:
                    continue
                manifest_path = run_dir / scan.ATTEMPT_MANIFEST
                manifest = scan._read_json(manifest_path)
                if manifest.get("verified_success") is True:
                    _clear_verdict_fields(manifest_path, manifest)

    @staticmethod
    def _delete_logical_try(battery: Path, prompt_id: str, attempt: int) -> None:
        prompt_dir = battery / prompt_id
        if not prompt_dir.is_dir():
            return
        base = f"try{attempt:02d}"
        for child in list(prompt_dir.iterdir()):
            suffix = child.name[len(base + ".requeue"):] if child.name.startswith(base + ".requeue") else ""
            if child.name == base or (suffix.isdigit() and suffix):
                if child.is_dir():
                    shutil.rmtree(child)

    def verdict(
        self,
        key: str,
        verdict: str,
        *,
        note: str = "",
        by: str = "",
        battery_id: str | None = None,
    ) -> dict[str, Any]:
        """Records a human verdict on one finished attempt, stamped beside `success`, never over it.

        Excluded verdicts (invalid, already_successful) write no `verified_success` and cancel no
        siblings; only a pass does. Eligibility is re-checked here, not trusted from the UI.
        """
        battery, run_dir = self.run_dir_for(key, battery_id)
        if battery is None:
            return {"ok": False, "error": "no battery"}
        if run_dir is None:
            return {"ok": False, "error": "unknown attempt"}

        if verdict not in scan.VERDICTS:
            return {"ok": False, "error": f"unknown verdict {verdict!r}; expected one of "
                                          f"{', '.join(scan.VERDICTS)}"}

        manifest_path = run_dir / scan.ATTEMPT_MANIFEST
        with self._lock:
            manifest = scan._read_json(manifest_path)
            state = str(manifest.get("state") or "")
            end_reason = str(manifest.get("end_reason") or "")
            if not scan.is_verifiable(state, end_reason):
                return {
                    "ok": False,
                    "error": f"not reviewable (state={state or '?'}); only finished attempts can be judged",
                }
            fields = {
                "verified_verdict": verdict,
                "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "verified_by": by or os.environ.get("USER") or "watcher",
                "verified_note": note,
            }
            if verdict in scan.EXCLUDED_VERDICTS:
                # Drop any stale boolean from an earlier pass/fail.
                stale = scan._read_json(manifest_path)
                stale.pop("verified_success", None)
                stale.update(fields)
                _write_json(manifest_path, stale)
            else:
                _stamp(manifest_path, {**fields, "verified_success": verdict == "pass"})
            cancellations = {"stopped": 0, "skipped": 0}
            if verdict == "pass":
                cancellations = self._cancel_successful_siblings(
                    battery, key, manifest, fields
                )
            # So the reviewer's next poll shows the new badge.
            self._invalidate_locked()

        disagrees = (verdict not in scan.EXCLUDED_VERDICTS
                     and bool(manifest.get("success")) != (verdict == "pass"))
        _log(f"verdict on {key}: {verdict.upper()} by {fields['verified_by']}"
             f"{' (predicate disagreed)' if disagrees else ''}")
        return {
            "ok": True,
            **fields,
            "verified_success": None if verdict in scan.EXCLUDED_VERDICTS else verdict == "pass",
            "siblings_stopped": cancellations["stopped"],
            "siblings_skipped": cancellations["skipped"],
            "sibling_cancellations": cancellations,
        }

    def _cancel_successful_siblings(
        self,
        battery: Path,
        winner_key: str,
        winner_manifest: dict[str, Any],
        verdict_fields: dict[str, Any],
    ) -> dict[str, int]:
        """Durably cancels only tries of the winner's prompt ID, then signals published PIDs."""
        prompt_id = str(winner_manifest.get("prompt_id") or winner_key.split("/", 1)[0])
        cancellation = {
            "stop_reason": ALREADY_SUCCESSFUL,
            "stop_requested_at": verdict_fields["verified_at"],
            "stop_requested_by": verdict_fields["verified_by"],
            "winning_attempt_key": winner_key,
        }

        # This battery-level entry covers queued and waiting-for-lease work that has no run
        # manifest yet. It is intentionally never removed by clear_verdict(): administrative stops
        # are irreversible and must not be silently requeued later.
        battery_path = battery / scan.BATTERY_MANIFEST
        with edit_json_locked(battery_path) as battery_manifest:
            winners = battery_manifest.get("human_verified_winners")
            if not isinstance(winners, dict):
                winners = {}
            winners[prompt_id] = {
                "winning_attempt_key": winner_key,
                "stop_requested_at": verdict_fields["verified_at"],
                "stop_requested_by": verdict_fields["verified_by"],
            }
            battery_manifest["human_verified_winners"] = winners

        for identity, retry in list(self._runner_retries.items()):
            if identity[0] == battery.name and retry.get("prompt_id") == prompt_id:
                self._runner_retries.pop(identity, None)

        # Watcher-launched retries have no separate durable queue file: remove their in-memory rows
        # immediately. Their partial runner (or the cleanup thread just before it) observes the
        # durable winner above and uses the runner's ordinary already-successful materialization.
        for retry_key, job in list(self._retry_jobs.items()):
            logical = job.get("attempt") or {}
            if (
                job.get("battery_id") == battery.name
                and str(logical.get("prompt_id") or "") == prompt_id
                and retry_key != winner_key
            ):
                if job.get("state") == "waiting":
                    source = job.get("source_manifest") or {}
                    _materialize_cancelled(
                        battery, battery_manifest, prompt_id,
                        int(logical.get("attempt") or source.get("attempt") or 0),
                        source, cancellation, fallback=logical,
                    )
                self._retry_jobs.pop(retry_key, None)

        stopped = 0
        known_cancellable = 0
        known_attempts: set[int] = set()
        for run_dir in scan.run_dirs_of(battery):
            if run_dir.parent.name != prompt_id or ".requeue" in run_dir.name:
                continue
            sibling_key = f"{prompt_id}/{run_dir.name}"
            sibling = scan._read_json(run_dir / scan.ATTEMPT_MANIFEST)
            try:
                sibling_attempt = int(sibling.get("attempt") or run_dir.name[3:])
                known_attempts.add(sibling_attempt)
            except (TypeError, ValueError):
                sibling_attempt = 0
            if sibling_key != winner_key and sibling.get("pending_retry") and sibling_attempt > 0:
                # The battery runner has published this retry but has not replaced the failed
                # execution yet. Materialize the cancellation now; its acquire-side winner poll
                # cancels the coordinator request and reaches the same transaction idempotently.
                _materialize_cancelled(
                    battery, battery_manifest, prompt_id, sibling_attempt, sibling, cancellation
                )
                known_cancellable += 1
                continue
            if sibling_key == winner_key or sibling.get("state") in {"finished", "requeued"}:
                continue

            known_cancellable += 1
            sibling_path = run_dir / scan.ATTEMPT_MANIFEST
            _stamp(sibling_path, cancellation)
            pid = sibling.get("pid")
            if not pid:
                continue
            try:
                os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                # The runner's post-PID check sees the durable stop request even if publication and
                # process exit raced this signal.
                continue
            _stamp(
                sibling_path,
                {
                    "killed_by": ALREADY_SUCCESSFUL,
                    "killed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            stopped += 1

        tries = int(battery_manifest.get("tries") or 0)
        queued = max(0, tries - len({n for n in known_attempts if n > 0}))
        skipped = max(0, known_cancellable - stopped) + queued
        _log(
            f"{winner_key} verified successful: requested stop for "
            f"{stopped} running and {skipped} unstarted sibling(s)"
        )
        return {"stopped": stopped, "skipped": skipped}

    def clear_verdict(self, key: str, *, battery_id: str | None = None) -> dict[str, Any]:
        """Un-reviews an attempt, for a misclick. Leaves no trace, so the row reads as never looked at
        rather than as a verdict of False."""
        battery, run_dir = self.run_dir_for(key, battery_id)
        if battery is None:
            return {"ok": False, "error": "no battery"}
        if run_dir is None:
            return {"ok": False, "error": "unknown attempt"}

        manifest_path = run_dir / scan.ATTEMPT_MANIFEST
        with self._lock:
            manifest = scan._read_json(manifest_path)
            if not scan.verdict_of(manifest):
                return {"ok": True, "cleared": False}
            _clear_verdict_fields(manifest_path, manifest)
            self._invalidate_locked()
        _log(f"verdict cleared on {key}")
        return {"ok": True, "cleared": True}

    def replay_status(self, key: str, *, battery_id: str | None = None) -> tuple[str, Path | None]:
        """Returns the auto-rendered clip, queueing a missing older one as a fallback."""
        if self.replay is None:
            return replay_mod.UNAVAILABLE, None
        _, run_dir = self.run_dir_for(key, battery_id)
        if run_dir is None:
            return replay_mod.UNAVAILABLE, None
        manifest = scan._read_json(run_dir / scan.ATTEMPT_MANIFEST)
        if manifest.get("end_reason") == ALREADY_SUCCESSFUL:
            return replay_mod.UNAVAILABLE, None
        if manifest.get("state") in {"starting", "running"}:
            # `pid` is the agent subprocess, not the runner. A missing pid is the normal startup
            # window, and a dead one begins the runner's normal replay-finalization window. Only a
            # quiescent manifest is evidence that the runner itself is gone and fallback rendering
            # can no longer race it.
            if scan.agent_is_alive(manifest, manifest.get("pid")):
                return replay_mod.UNAVAILABLE, None
            quiet_for = time.time() - _last_sign_of_life(run_dir)
            if quiet_for < REPLAY_FINALIZE_GRACE_SECONDS:
                # Keep an orphaned overlay polling while the runner gets first chance to publish
                # its clip. Returning UNAVAILABLE would strand that already-open overlay on a 409.
                return replay_mod.RENDERING, None
        status = self.replay.request(key, run_dir)
        clip = run_dir / video.REPLAY_NAME
        return status, (clip if status == replay_mod.READY and clip.is_file() else None)

    def log_tail(
        self,
        key: str,
        lines: int = LOG_TAIL_LINES,
        since: int | None = None,
        full: bool = False,
        battery_id: str | None = None,
    ) -> dict[str, Any]:
        """Reads agent.log as a full log, a tail, or the delta since byte offset `since`.

        An unterminated trailing line comes back as `partial` and does not advance the cursor.
        """
        empty = {"lines": [], "offset": 0, "size": 0, "partial": "", "reset": False}
        _, run_dir = self.run_dir_for(key, battery_id)
        if run_dir is None:
            return empty
        path = run_dir / "agent.log"
        if not path.exists():
            return empty
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                # A cursor past the end means truncation/rotation: start over and tell the client.
                reset = since is not None and since > size
                bootstrap = since is None or reset
                start = (
                    0
                    if bootstrap and full
                    else max(0, size - LOG_BOOTSTRAP_BYTES)
                    if bootstrap
                    else since
                )
                handle.seek(start)
                raw = handle.read()
        except OSError as error:
            return {**empty, "lines": [f"<unreadable: {error!r}>"]}

        if bootstrap and start > 0:
            # The bootstrap window lands mid-line; drop that fragment rather than show half of it.
            head = raw.find(b"\n")
            start += len(raw) if head < 0 else head + 1
            raw = b"" if head < 0 else raw[head + 1:]

        partial = b""
        if raw and not raw.endswith(b"\n"):
            tail = raw.rfind(b"\n")
            partial, raw = (raw, b"") if tail < 0 else (raw[tail + 1:], raw[:tail + 1])

        text = raw.decode("utf-8", errors="replace")
        return {
            # Only a tail bootstrap is windowed; a delta must return every line it advances past.
            "lines": text.splitlines()[-lines:] if bootstrap and not full else text.splitlines(),
            "offset": start + len(raw),
            "size": size,
            "partial": partial.decode("utf-8", errors="replace"),
            "reset": reset,
        }

    def frame_path(self, key: str, *, battery_id: str | None = None) -> Path | None:
        _, run_dir = self.run_dir_for(key, battery_id)
        return None if run_dir is None else scan._latest_frame(run_dir)

    def run_dir_for(self, key: str, battery_id: str | None) -> tuple[Path | None, Path | None]:
        """(battery, run dir) for one request's attempt key; either may be None."""
        battery = self.battery_for(battery_id)
        return battery, (_safe_run_dir(battery, key) if battery is not None else None)


def _coordinator_call(url: str, call: Callable[[Any], Any]) -> Any:
    """Runs one `CoordinatorClient` coroutine to completion, bounded by the coordinator timeout."""
    from sari_bench.client import CoordinatorClient

    async def run() -> Any:
        async with CoordinatorClient(url) as client:
            return await call(client)

    return asyncio.run(asyncio.wait_for(run(), timeout=COORDINATOR_TIMEOUT_SECONDS))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic replace, so a poller mid-read never sees a half-written manifest."""
    try:
        temp = path.with_name(f".{path.name}.tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)
    except OSError as error:  # noqa: BLE001
        _log(f"could not write {path}: {error!r}")


def _last_sign_of_life(run_dir: Path) -> float:
    """Newest mtime anywhere in the run dir: a lower-bound end time for an abandoned attempt."""
    newest = 0.0
    for path in [run_dir, *run_dir.rglob("*")]:
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest or time.time()


def _try_number(manifest: dict[str, Any], run_dir: Path) -> int | None:
    """The manifest's try number, else the dir's: `try07` and `try07.requeue00` are both 7."""
    try:
        return int(manifest.get("attempt") or run_dir.name[3:].split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def _materialize_cancelled(
    battery: Path,
    plan: dict[str, Any],
    prompt_id: str,
    attempt: int,
    source: dict[str, Any],
    winner: dict[str, Any],
    *,
    fallback: dict[str, Any] | None = None,
) -> None:
    """Records one unstarted try as already_successful, configured from `source` then the plan."""
    from sari_bench.runner import Prompt, materialize_already_successful

    fallback = fallback or {}

    def text(name: str) -> str:
        return str(source.get(name) or fallback.get(name) or "")

    def given(name: str, default: Any) -> Any:
        return source.get(name) if source.get(name) is not None else default

    materialize_already_successful(
        output_dir=battery,
        prompt=Prompt(id=prompt_id, prompt=text("prompt"), family=text("family"),
                      looking_for=text("looking_for")),
        attempt=attempt,
        winner=winner,
        arm=str(source.get("arm") or plan.get("arm") or "graph"),
        context_policy=str(source.get("context_policy") or plan.get("context_policy") or "baseline"),
        adaptive_leg_replanning=bool(
            given("adaptive_leg_replanning", plan.get("adaptive_leg_replanning", False))
        ),
        api_max_attempts=int(source.get("api_max_attempts") or plan.get("api_max_attempts") or 10),
        max_api_requeues=int(given("max_api_requeues", plan.get("max_api_requeues") or 0)),
        ocr_url=str(source.get("ocr_url") or plan.get("ocr_url") or ""),
        requeues=int(source.get("requeues") or 0),
    )


def _clear_verdict_fields(path: Path, manifest: dict[str, Any]) -> None:
    for field in VERDICT_FIELDS:
        manifest.pop(field, None)
    _write_json(path, manifest)


def _stamp(path: Path, fields: dict[str, Any]) -> None:
    payload = scan._read_json(path)
    payload.update(fields)
    _write_json(path, payload)


def _int_param(
    query: dict[str, list[str]], name: str, default: int | None, low: int, high: int | None
) -> int | None:
    """Reads one clamped integer out of a query string, falling back on anything unparseable."""
    values = query.get(name)
    if not values:
        return default
    try:
        value = int(values[0])
    except ValueError:
        return default
    value = max(low, value)
    return value if high is None else min(high, value)


def _safe_run_dir(battery: Path, key: str) -> Path | None:
    """Resolves an HTTP-supplied attempt key to a run dir, refusing anything outside the battery."""
    root = battery.resolve()
    candidate = (battery / key).resolve()
    # Exactly <prompt>/<tryNN[.requeueNN]>: not the battery, a prompt dir, or anything deeper.
    if candidate.parent.parent != root or not scan.is_try_dir_name(candidate.name):
        return None
    return candidate if candidate.is_dir() else None


class _PoolPoller(threading.Thread):
    """Keeps the sandbox pool fresh on its own event loop.

    Read-only: it sends `bench.status` and nothing else, so it never competes with a worker for a
    sandbox. A coordinator that is down is reported in the UI, not raised.
    """

    daemon = True

    def __init__(self, state: WatchState, coordinator_url: str) -> None:
        super().__init__(name="pool-poller")
        self.state = state
        self.coordinator_url = coordinator_url
        self._stop = threading.Event()

    def run(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        from sari_bench.client import CoordinatorClient

        while not self._stop.is_set():
            try:
                async with CoordinatorClient(self.coordinator_url) as client:
                    fleet = await client.fleet_status()
                    self.state.set_pool(
                        fleet["sandboxes"], waiting=fleet["waiting"], fleet=fleet
                    )
            except Exception as error:  # noqa: BLE001 - the dashboard survives a dead coordinator
                self.state.set_pool([], f"{error!r}")
            await asyncio.sleep(POOL_REFRESH_SECONDS)

    def stop(self) -> None:
        self._stop.set()


class _StatePoller(threading.Thread):
    """Continuously scan the watched battery so completion notifications are never UI-driven."""

    daemon = True

    def __init__(self, state: WatchState, interval: float = WATCH_POLL_SECONDS) -> None:
        super().__init__(name="watch-state-poller")
        self.state = state
        self.interval = interval
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                # Force a scan: the state cache is for HTTP readers, whereas this is the durable
                # completion detector that drives Discord and replay preparation.
                self.state.snapshot(force=True)
            except Exception as error:  # noqa: BLE001 - a malformed run must not kill the watcher
                _log(f"state poll failed: {error!r}")
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()


class Handler(BaseHTTPRequestHandler):
    state: WatchState = None  # type: ignore[assignment]

    def log_message(self, *_args: Any) -> None:  # noqa: D102 - silence per-request stderr spam
        pass

    def _write_body(self, body: bytes) -> None:
        """Writes a response body without reporting an ordinary client disconnect."""
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # browsers abandon video responses when seeking

    def _send(
        self,
        code: int,
        body: bytes,
        content_type: str,
        extra: dict[str, str] | None = None,
        *,
        cache: str = "no-store",
        compress: bool = False,
    ) -> None:
        headers = dict(extra or {})
        if compress:
            headers["Vary"] = "Accept-Encoding"
            if len(body) >= GZIP_MIN_BYTES and "gzip" in self.headers.get("Accept-Encoding", ""):
                body = gzip.compress(body, compresslevel=5)
                headers["Content-Encoding"] = "gzip"
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self._write_body(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(code, json.dumps(payload, default=str).encode("utf-8"), "application/json",
                   compress=True)

    def _result(self, result: dict[str, Any], ok_code: int = 200) -> None:
        self._json(result, code=ok_code if result.get("ok") else 400)

    def _send_validated(
        self,
        path: Path,
        content_type: str,
        *,
        missing: bytes = b"not found",
        cache: str = "no-store",
        compress: bool = False,
    ) -> None:
        """Serves a small file under an ETag, answering a matching If-None-Match with an empty 304."""
        try:
            stat = path.stat()
            etag = f'"{path.name}-{stat.st_mtime_ns:x}-{stat.st_size:x}"'
            if self.headers.get("If-None-Match") == etag:
                self._send(304, b"", content_type, {"ETag": etag}, cache=cache)
                return
            body = path.read_bytes()
        except OSError:
            self._send(404, missing, "text/plain")
            return
        self._send(200, body, content_type, {"ETag": etag}, cache=cache, compress=compress)

    def _send_file(self, path: Path, content_type: str) -> None:
        """Serves a file, honouring a single byte range, which `<video>` needs to seek."""
        try:
            total = path.stat().st_size
        except OSError as error:
            self._send(404, f"unreadable: {error!r}".encode("utf-8"), "text/plain")
            return

        start, end = 0, total - 1
        partial = False
        header = self.headers.get("Range", "")
        if header.startswith("bytes=") and "," not in header and total:
            first, _, last = header[len("bytes="):].partition("-")
            try:
                if first:
                    start = int(first)
                    end = int(last) if last else total - 1
                elif last:  # a suffix range: the LAST n bytes
                    start = max(0, total - int(last))
                partial = True
            except ValueError:
                partial = False
        end = min(end, total - 1)

        if partial and (start > end or start >= total):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{total}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if not partial:
            start, end = 0, total - 1
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                chunk = handle.read(max(0, end + 1 - start))
        except OSError as error:
            self._send(404, f"unreadable: {error!r}".encode("utf-8"), "text/plain")
            return
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        # Cacheable briefly: browsers that cannot keep the clip refuse to seek or re-fetch per seek.
        self.send_header("Cache-Control", "private, max-age=60")
        self.end_headers()
        self._write_body(chunk)

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _battery_param(self, query: str) -> str:
        """The `?battery=` a request carries, naming the run the browser is looking at."""
        return parse_qs(query).get("battery", [""])[0]

    def _report_csv(self, battery_id: str = "", grain: str = "attempts") -> None:
        """The `python -m sari_bench report` CSV for `grain` "attempts" (default) or "roles"."""
        battery = self.state.battery_for(battery_id)
        if battery is None:
            self._send(404, b"no battery found", "text/plain")
            return

        import csv
        import io

        from sari_bench import report

        rows, _legs = report.collect(battery)
        if grain == "roles":
            rows, columns = report.role_rows(rows), report.ROLE_COLUMNS
        else:
            grain, columns = "attempts", report.ATTEMPT_COLUMNS
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        self._send(
            200,
            buffer.getvalue().encode("utf-8"),
            "text/csv; charset=utf-8",
            {"Content-Disposition": f'attachment; filename="{battery.name}-{grain}.csv"'},
        )

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        battery_id = self._battery_param(parsed.query)

        if path in {"/", "/index.html"}:
            # Revalidated rather than stored, so an edited page is picked up on the next load.
            self._send_validated(STATIC_DIR / "dashboard.html", "text/html; charset=utf-8",
                                 cache="no-cache", compress=True)
            return

        if path == "/api/state":
            self._json(self.state.view(battery_id))
            return

        if path == "/api/fleet/status":
            self._json(self.state.fleet_status())
            return

        if path == "/api/report.csv":
            self._report_csv(battery_id, parse_qs(parsed.query).get("grain", [""])[0])
            return

        if path.startswith("/api/attempt/"):
            rest = path[len("/api/attempt/"):]
            key, _, action = rest.rpartition("/")
            if action == "frame.png":
                frame = self.state.frame_path(key, battery_id=battery_id)
                if frame is None:
                    self._send(404, b"no frame yet", "text/plain")
                    return
                # Polled at 4 FPS; the ETag turns an unchanged frame into an empty 304.
                self._send_validated(
                    frame,
                    "image/jpeg" if frame.suffix.lower() in {".jpg", ".jpeg"} else "image/png",
                    missing=b"no frame yet",
                )
                return
            if action == "log":
                query = parse_qs(parsed.query)
                self._json(self.state.log_tail(
                    key,
                    lines=_int_param(query, "lines", LOG_TAIL_LINES, 1, LOG_MAX_LINES),
                    since=_int_param(query, "since", None, 0, None),
                    full=query.get("full", ["0"])[0] == "1",
                    battery_id=battery_id,
                ))
                return
            if action == "replay.mp4":
                status, clip = self.state.replay_status(key, battery_id=battery_id)
                if clip is not None:
                    self._send_file(clip, "video/mp4")
                elif status == replay_mod.RENDERING:
                    # 202: the encode is queued on the replay worker. The page polls until 200.
                    self._json({"status": status}, code=202)
                else:
                    self._json({"status": replay_mod.UNAVAILABLE,
                                "reason": "no frames or ffmpeg is unavailable"},
                               code=409)
                return
            if action == "replay.vtt":
                _, run_dir = self.state.run_dir_for(key, battery_id)
                subtitles = run_dir / "replay.vtt" if run_dir is not None else None
                if subtitles is None or not subtitles.is_file():
                    self._send(404, b"no subtitles", "text/plain")
                else:
                    self._send_validated(subtitles, "text/vtt; charset=utf-8",
                                         missing=b"no subtitles")
                return

        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        battery_id = self._battery_param(parsed.query)
        if path == "/api/battery/rename":
            body = self._body()
            result = self.state.rename_battery(
                str(body.get("battery") or battery_id), str(body.get("name") or "")
            )
            self._result(result)
            return
        if path == "/api/fleet/cap":
            body = self._body()
            if "limit" not in body:
                self._json({"ok": False, "error": "body needs 'limit'"}, code=400)
                return
            result = self.state.set_fleet_cap(body.get("limit"))
            self._result(result)
            return
        if path in {"/api/fleet/quarantine", "/api/fleet/unquarantine"}:
            body = self._body()
            result = self.state.change_quarantine(
                str(body.get("sandbox") or ""),
                clear=path.endswith("/unquarantine"),
                reason=str(body.get("reason") or "web_ui"),
            )
            self._result(result)
            return
        if path.startswith("/api/attempt/"):
            rest = path[len("/api/attempt/"):]
            if rest.endswith("/verdict/clear"):
                result = self.state.clear_verdict(
                    rest[:-len("/verdict/clear")], battery_id=battery_id
                )
                self._result(result)
                return
            if rest.endswith("/verdict"):
                body = self._body()
                # `verdict` is the full four-value field; `success` remains accepted so anything
                # scripted against the original boolean route keeps working unchanged.
                verdict = body.get("verdict")
                if verdict is None and isinstance(body.get("success"), bool):
                    verdict = "pass" if body["success"] else "fail"
                if verdict not in scan.VERDICTS:
                    self._json({"ok": False, "error": "body needs a boolean 'success' or a "
                                                      f"'verdict' of {', '.join(scan.VERDICTS)}"},
                               code=400)
                    return
                result = self.state.verdict(
                    rest[:-len("/verdict")],
                    verdict,
                    note=str(body.get("note") or ""),
                    by=str(body.get("by") or ""),
                    battery_id=battery_id,
                )
                self._result(result)
                return
            if rest.endswith("/kill"):
                result = self.state.kill(rest[:-len("/kill")], battery_id=battery_id)
                self._result(result)
                return
            if rest.endswith("/retry"):
                result = self.state.retry(rest[:-len("/retry")], battery_id=battery_id)
                self._result(result, ok_code=202)
                return
        self._send(404, b"not found", "text/plain")


def serve(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sari_bench watch", description="Live dashboard for a running prompt battery."
    )
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT,
                        help="Directory holding battery dirs. The newest is followed automatically.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Pin one battery dir instead of auto-discovering the newest.")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address. 0.0.0.0 exposes the dashboard - including its destructive "
                             "kill and retry endpoints - to the network.")
    parser.add_argument("--coordinator", default=f"ws://localhost:{DEFAULT_COORDINATOR_PORT}",
                        help="Coordinator to read the sandbox pool from. --no-pool to skip.")
    parser.add_argument("--no-pool", action="store_true", help="Do not connect to the coordinator.")
    parser.add_argument("--discord", action="store_true",
                        help=f"Send Discord notifications (needs {notify.WEBHOOK_ENV}).")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "watchconfig.toml",
                        help="Watcher TOML configuration (defaults to watchconfig.toml).")
    parser.add_argument("--no-replay", action="store_true",
                        help="Announce halts without rendering and attaching a replay clip.")
    parser.add_argument("--replay-fps", type=float, default=video.UPLOAD_FPS)
    parser.add_argument("--replay-width", type=int, default=video.UPLOAD_WIDTH)
    parser.add_argument(
        "--review-max-frames",
        type=int,
        default=video.MAX_REPLAY_FRAMES,
        help=f"Maximum frames in a dashboard replay (default: {video.MAX_REPLAY_FRAMES}; "
             "0 keeps every capture). Step frames are always retained.",
    )
    parser.add_argument("--replay-max-mb", type=float,
                        default=video.DISCORD_BUDGET_BYTES / 1e6,
                        help="Size budget for an attached clip.")
    parser.add_argument("--replay-backfill", action="store_true",
                        help="Also announce attempts that had already finished when the watcher "
                             "started. Off by default: a restart mid-battery would otherwise replay "
                             "the whole run into the channel.")
    args = parser.parse_args(argv)
    if args.review_max_frames < 0:
        parser.error("--review-max-frames cannot be negative")

    _load_api_env()

    collapse_alerts = False
    discord_enabled = args.discord
    if args.config.is_file():
        try:
            watch_config = load_watch_config(args.config)
        except WatchConfigError as error:
            parser.error(str(error))
        collapse_alerts = bool(watch_config.get("discord", "collapse_alerts", False))
        discord_enabled = discord_enabled or bool(watch_config.get("discord", "enable", False))

    discord = notify.Discord(enabled=discord_enabled, collapse_alerts=collapse_alerts)
    if discord_enabled and not discord.enabled:
        _log(f"Discord enabled via --discord or [discord].enable but {notify.WEBHOOK_ENV} is unset; "
             "notifications are OFF")

    replay = replay_mod.ReplayNotifier(
        discord,
        enabled=not args.no_replay,
        max_bytes=int(args.replay_max_mb * 1e6),
        width=args.replay_width,
        fps=args.replay_fps,
        review_max_frames=args.review_max_frames or None,
    )
    # Started whenever rendering is on, not only when Discord is: the dashboard's review flow asks
    # this same worker for clips, and it must be running with no webhook configured.
    if replay.render_enabled:
        if shutil.which("ffmpeg") is None:
            _log("replay clips are ON but ffmpeg is not on PATH; halts will post without one "
                 "and the dashboard cannot show replays")
        replay.start()

    state = WatchState(
        bench_root=args.bench_root.resolve(),
        fixed_battery=args.run_dir.resolve() if args.run_dir else None,
        discord=discord,
        replay=replay,
        backfill=args.replay_backfill,
        coordinator_url=args.coordinator,
    )
    state_poller = _StatePoller(state)
    state_poller.start()

    battery = state.resolve_battery()
    if args.run_dir:
        _log(f"PINNED to battery dir {args.run_dir} (single-battery mode)")
        if not args.run_dir.exists():
            _log("  ...which does not exist yet; it will appear once the runner creates it")
    elif battery is not None:
        _log(f"auto-discovery under {state.bench_root}: watching {battery.name} (newest of "
             f"{len(scan.find_batteries(state.bench_root))})")
    else:
        _log(f"auto-discovery under {state.bench_root}: no battery dirs yet, waiting for one")

    poller = None
    if not args.no_pool:
        poller = _PoolPoller(state, args.coordinator)
        poller.start()

    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    _log(f"dashboard on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("stopping")
    finally:
        state_poller.stop()
        if poller is not None:
            poller.stop()
        replay.stop()
        server.server_close()
    return 0


def _load_api_env() -> None:
    """Loads secrets.env from the repo root, the same file every other module reads."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = REPO_ROOT / "secrets.env"
    if env_path.exists():
        load_dotenv(env_path)


def main(argv: list[str] | None = None) -> int:
    return serve(argv)


if __name__ == "__main__":
    raise SystemExit(main())
