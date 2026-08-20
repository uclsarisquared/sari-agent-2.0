"""The sandbox registry for Distributed Sari Bench.

Sims connect on ``/sandbox`` and advertise the command port they self-assigned; benchmark workers
connect on ``/bench`` and lease one at a time. The registry's whole job is to make sure a worker
never gets a sandbox that another worker is using, and never gets one that has not been reset since
its last attempt.

Two invariants drive the design:

* **Reset happens on release, not on acquire.** A sandbox handed back after an attempt is taken out
  of the pool, told to reset, and only re-pooled once *it* reports ready. So an acquire always
  yields a pristine environment, and a run that crashed mid-attempt still gets cleaned up - the
  cleanup does not depend on the agent process surviving to ask for it.
* **A lease outlives neither its worker nor its sandbox.** A worker that disconnects has its lease
  reaped; a sandbox that stops heartbeating has its lease holder told ``bench.sandbox_lost`` so the
  attempt can be requeued instead of hanging on a machine that is never coming back.
* **Dropping a sandbox always hangs up on it.** Eviction is a decision the sim has to hear about:
  it re-registers on connect, so a drop that leaves its socket open removes it from the pool for
  good. Closing the connection is what makes an eviction recoverable rather than terminal.

Leases remain in memory, but human aliases and quarantines are persisted. A coordinator restart is
recoverable without operator action: sims reconnect on their own backoff, workers reconnect per
attempt, and a known-bad sandbox remains ineligible until it is explicitly unquarantined.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sari_bench.protocol import (
    BENCH_ROUTE,
    DEFAULT_COORDINATOR_PORT,
    HEARTBEAT_TIMEOUT_SECONDS,
    SANDBOX_ROUTE,
    STATE_LEASED,
    STATE_READY,
    STATE_RESETTING,
    commands_uri,
    decode,
    encode,
    route_of,
)

# How often the reaper sweeps for dead sandboxes and expired leases.
REAP_INTERVAL_SECONDS = 2.0

# Resetting several Unity players on one machine at once can stall all their main threads in asset
# teardown/rebuild. Keep one reset in flight; queued sandboxes remain out of the leasable pool.
DEFAULT_MAX_CONCURRENT_RESETS = 1
DEFAULT_RESET_TIMEOUT_SECONDS = 180.0

# Backstop for a worker whose TCP connection is half-open, so its lease is never held forever.
# Generous by default: a legitimate attempt can legitimately run for tens of minutes.
DEFAULT_LEASE_TTL_SECONDS = 7200.0
COORDINATOR_STATE_ENV = "SARI_BENCH_COORDINATOR_STATE"
DEFAULT_COORDINATOR_STATE_PATH = Path(
    os.environ.get(
        COORDINATOR_STATE_ENV,
        Path.home() / ".local" / "state" / "sari_bench" / "coordinator.json",
    )
)


def _human_label(value: Any, fallback: str, *, max_chars: int = 80) -> str:
    """Return a compact log/UI label without allowing control characters in protocol output."""
    label = re.sub(r"[^A-Za-z0-9._/-]+", "-", str(value or "").strip()).strip("-./")
    return (label or fallback)[:max_chars]


@dataclass
class Sandbox:
    sandbox_id: str
    host: str
    port: int
    websocket: Any
    state: str
    alias: str
    store_loaded: bool = True
    v1_compatibility: bool = False
    unity_version: str = ""
    store_name: str = ""
    last_heartbeat: float = field(default_factory=time.monotonic)
    lease_id: str | None = None
    reset_started_at: float | None = None
    reset_reason: str = ""
    quarantined: bool = False
    quarantine_reason: str = ""
    quarantine_source: str = ""
    quarantined_at: str = ""
    lease_alias: str = ""

    @property
    def leasable(self) -> bool:
        """Ready, idle, and actually serving a store."""
        return (
            self.lease_id is None
            and self.state == STATE_READY
            and self.store_loaded
            and not self.quarantined
        )

    def describe(self) -> dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "sandbox_alias": self.alias,
            "host": self.host,
            "port": self.port,
            "state": self.state,
            "store_loaded": self.store_loaded,
            "v1_compatibility": self.v1_compatibility,
            "unity_version": self.unity_version,
            "store_name": self.store_name,
            "lease_id": self.lease_id,
            "lease_alias": self.lease_alias,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
            "quarantine_source": self.quarantine_source,
            "quarantined_at": self.quarantined_at,
            "reset_seconds": (
                round(time.monotonic() - self.reset_started_at, 1)
                if self.reset_started_at is not None
                else None
            ),
            "reset_reason": self.reset_reason,
            "commands_uri": commands_uri(self.host, self.port),
        }


@dataclass
class Lease:
    lease_id: str
    sandbox_id: str
    holder: Any
    alias: str
    created_at: float = field(default_factory=time.monotonic)


class Coordinator:
    """Sandbox pool plus lease bookkeeping. One instance per coordinator process."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = DEFAULT_COORDINATOR_PORT,
        heartbeat_timeout: float = HEARTBEAT_TIMEOUT_SECONDS,
        lease_ttl: float = DEFAULT_LEASE_TTL_SECONDS,
        max_concurrent_resets: int = DEFAULT_MAX_CONCURRENT_RESETS,
        reset_timeout: float = DEFAULT_RESET_TIMEOUT_SECONDS,
        state_path: Path | str | None = None,
        log: Any = None,
    ) -> None:
        self.host = host
        self.port = port
        self.heartbeat_timeout = heartbeat_timeout
        self.lease_ttl = lease_ttl
        self.max_concurrent_resets = max(1, max_concurrent_resets)
        self.reset_timeout = reset_timeout
        self.state_path = Path(state_path).expanduser() if state_path else None
        self.log = log or _print_log

        self._sandboxes: dict[str, Sandbox] = {}
        self._leases: dict[str, Lease] = {}
        self._sandbox_aliases: dict[str, str] = {}
        self._quarantines: dict[str, dict[str, str]] = {}
        self._next_sandbox_alias = 1
        self._next_lease_alias = 1
        self._load_state()
        # None is the process-lifetime default ("all"). A finite limit constrains only new claims;
        # lowering it below the current lease count drains naturally as those leases are released.
        self._lease_limit: int | None = None
        # Workers waiting for a free sandbox, oldest first, so leases are handed out fairly.
        self._waiters: list[asyncio.Future[Lease]] = []
        self._pending_resets: list[tuple[str, str, str]] = []
        self._active_resets: set[str] = set()
        self._server: Any = None
        self._reaper: asyncio.Task[None] | None = None
        # Strong refs to in-flight socket closes, so they are not garbage-collected mid-handshake.
        self._closers: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------ durable health state

    def _load_state(self) -> None:
        if self.state_path is None:
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as error:
            self.log(f"Could not read coordinator state {self.state_path}: {error!r}")
            return
        if not isinstance(payload, dict):
            return
        aliases = payload.get("sandbox_aliases")
        quarantines = payload.get("quarantines")
        if isinstance(aliases, dict):
            self._sandbox_aliases = {
                str(sandbox_id): str(alias)
                for sandbox_id, alias in aliases.items()
                if sandbox_id and alias
            }
        if isinstance(quarantines, dict):
            self._quarantines = {
                str(sandbox_id): {
                    str(key): str(value)
                    for key, value in record.items()
                    if value is not None
                }
                for sandbox_id, record in quarantines.items()
                if sandbox_id and isinstance(record, dict)
            }
        used_numbers = []
        for alias in self._sandbox_aliases.values():
            match = re.fullmatch(r"sandbox-(\d+)", alias)
            if match:
                used_numbers.append(int(match.group(1)))
        self._next_sandbox_alias = max(used_numbers, default=0) + 1

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        payload = {
            "version": 1,
            "sandbox_aliases": self._sandbox_aliases,
            "quarantines": self._quarantines,
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{self.state_path.name}.", suffix=".tmp", dir=self.state_path.parent
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                os.replace(temp_name, self.state_path)
            except BaseException:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temp_name)
                raise
        except OSError as error:
            self.log(f"Could not persist coordinator state {self.state_path}: {error!r}")

    def _sandbox_alias(self, sandbox_id: str) -> str:
        alias = self._sandbox_aliases.get(sandbox_id)
        if alias:
            return alias
        alias = f"sandbox-{self._next_sandbox_alias:02d}"
        self._next_sandbox_alias += 1
        self._sandbox_aliases[sandbox_id] = alias
        self._save_state()
        return alias

    def _resolve_sandbox_id(self, selector: str) -> str | None:
        if selector in self._sandboxes or selector in self._quarantines:
            return selector
        matches = [
            sandbox_id
            for sandbox_id, alias in self._sandbox_aliases.items()
            if selector == alias
        ]
        return matches[0] if len(matches) == 1 else None

    def _resolve_sandbox(self, selector: str) -> Sandbox | None:
        sandbox_id = self._resolve_sandbox_id(selector)
        return self._sandboxes.get(sandbox_id) if sandbox_id else None

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        import websockets

        self._server = await websockets.serve(self._handle_connection, self.host, self.port)
        self._reaper = asyncio.create_task(self._reap_forever())
        self.log(f"Coordinator listening on ws://{self.host}:{self.port}{SANDBOX_ROUTE} and {BENCH_ROUTE}")

    async def stop(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper
            self._reaper = None

        # A bench handler can be parked awaiting one of these futures. Wake it before waiting for
        # the websocket server's handlers to close, or shutdown itself deadlocks on an empty pool.
        for waiter in self._waiters:
            if not waiter.done():
                waiter.cancel()
        self._waiters.clear()

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        for closer in list(self._closers):
            closer.cancel()
        self._closers.clear()

    @property
    def bound_port(self) -> int:
        """The port actually bound, which differs from ``port`` when 0 was requested."""
        if self._server is None:
            return self.port
        for socket in getattr(self._server, "sockets", []) or []:
            return socket.getsockname()[1]
        return self.port

    # ------------------------------------------------------------------ routing

    async def _handle_connection(self, websocket: Any, path: str | None = None) -> None:
        route = route_of(websocket, path)
        if route == SANDBOX_ROUTE:
            await self._handle_sandbox(websocket)
        elif route == BENCH_ROUTE:
            await self._handle_bench(websocket)
        else:
            await websocket.close(code=1008, reason=f"Use {SANDBOX_ROUTE} or {BENCH_ROUTE}")

    # ------------------------------------------------------------------ sandbox side

    async def _handle_sandbox(self, websocket: Any) -> None:
        sandbox_id: str | None = None
        try:
            async for raw in websocket:
                message = decode(raw)
                if message is None:
                    continue

                message_type = message.get("type")
                if message_type == "sandbox.hello":
                    sandbox_id = await self._register(websocket, message)
                elif sandbox_id is None:
                    # Everything else is meaningless before we know who is speaking.
                    continue
                elif message_type == "sandbox.heartbeat":
                    sandbox = self._sandboxes.get(sandbox_id)
                    if sandbox is not None:
                        sandbox.last_heartbeat = time.monotonic()
                elif message_type == "sandbox.state":
                    await self._on_sandbox_state(sandbox_id, message)
        except Exception as error:  # noqa: BLE001 - a dropped sim must not kill the coordinator
            self.log(f"Sandbox connection error ({sandbox_id}): {error!r}")
        finally:
            if sandbox_id is not None:
                # The socket is already gone; closing it again would only stall this teardown.
                await self._drop_sandbox(sandbox_id, reason="disconnected", close_socket=False)

    async def _register(self, websocket: Any, message: dict[str, Any]) -> str | None:
        sandbox_id = message.get("sandbox_id")
        port = message.get("port")
        if not isinstance(sandbox_id, str) or not sandbox_id or not isinstance(port, int):
            await _send(websocket, encode("coord.error", reason="invalid_hello"))
            return None

        # The sim does not have to know its own reachable address: whatever address it reached us
        # from is, by construction, an address that works. advertised_host overrides that for the
        # cases where it does not - NAT, or a container with a published port.
        advertised = message.get("advertised_host") or ""
        host = advertised or _peer_host(websocket) or "127.0.0.1"

        # A sim that restarted re-registers under the same id; retire the stale entry (and any
        # lease pointing at it) rather than ending up with two rows for one machine. Only hang up on
        # the stale entry's socket if it is a different one - a sim that says hello twice on a single
        # connection must not have that connection closed underneath it.
        stale = self._sandboxes.get(sandbox_id)
        if stale is not None:
            await self._drop_sandbox(
                sandbox_id, reason="re-registered", close_socket=stale.websocket is not websocket
            )

        alias = self._sandbox_alias(sandbox_id)
        quarantine = self._quarantines.get(sandbox_id) or {}
        sandbox = Sandbox(
            sandbox_id=sandbox_id,
            host=host,
            port=port,
            websocket=websocket,
            state=str(message.get("state") or STATE_READY),
            alias=alias,
            store_loaded=bool(message.get("store_loaded", True)),
            v1_compatibility=bool(message.get("v1_compatibility", False)),
            unity_version=str(message.get("unity_version") or ""),
            store_name=str(message.get("store_name") or ""),
            quarantined=bool(quarantine),
            quarantine_reason=str(quarantine.get("reason") or ""),
            quarantine_source=str(quarantine.get("source") or ""),
            quarantined_at=str(quarantine.get("quarantined_at") or ""),
        )
        self._sandboxes[sandbox_id] = sandbox

        if not sandbox.store_loaded:
            self.log(
                f"Sandbox {sandbox_id} at {sandbox.host}:{sandbox.port} registered WITHOUT a store "
                "file; it will not be leased until it reports one."
            )
        else:
            suffix = " [QUARANTINED]" if sandbox.quarantined else ""
            self.log(
                f"Sandbox {sandbox.alias} registered at {sandbox.host}:{sandbox.port} "
                f"({sandbox.state}){suffix}"
            )

        await _send(
            websocket,
            encode("coord.welcome", sandbox_id=sandbox_id, sandbox_alias=sandbox.alias),
        )

        # The coordinator keeps leases only in memory. After it restarts, a Unity process may
        # reconnect still reporting Leased from the old coordinator, but no matching lease can
        # possibly exist here. Reset that orphaned attempt so the sim reports Ready and rejoins the
        # pool; otherwise it remains visibly "Leased" yet can never be acquired again.
        if sandbox.quarantined:
            # Reconnection never clears a quarantine. Keep Unity locally unleased while leaving the
            # control socket up so operators can still see and recover the instance.
            await _send(websocket, encode("coord.release", reason="quarantined"))
        elif sandbox.state == STATE_LEASED and sandbox.lease_id is None:
            await self._queue_reset(
                sandbox_id,
                lease_id="coordinator-recovery",
                reason="orphaned_lease",
            )

        self._fulfil_waiters()
        return sandbox_id

    async def _on_sandbox_state(self, sandbox_id: str, message: dict[str, Any]) -> None:
        sandbox = self._sandboxes.get(sandbox_id)
        if sandbox is None:
            return

        sandbox.state = str(message.get("state") or sandbox.state)
        sandbox.store_loaded = bool(message.get("store_loaded", sandbox.store_loaded))
        sandbox.last_heartbeat = time.monotonic()

        # A sandbox reporting Ready after a release is what actually returns it to the pool: the
        # reset it was told to run has finished.
        if sandbox.state == STATE_READY and sandbox.lease_id is None:
            self._active_resets.discard(sandbox_id)
            self._pending_resets = [
                pending for pending in self._pending_resets if pending[0] != sandbox_id
            ]
            sandbox.reset_started_at = None
            sandbox.reset_reason = ""
            await self._pump_resets()
            self._fulfil_waiters()

    async def _drop_sandbox(self, sandbox_id: str, *, reason: str, close_socket: bool = True) -> None:
        sandbox = self._sandboxes.pop(sandbox_id, None)
        if sandbox is None:
            return

        self.log(f"Sandbox {sandbox_id} removed ({reason})")
        self._active_resets.discard(sandbox_id)
        self._pending_resets = [
            pending for pending in self._pending_resets if pending[0] != sandbox_id
        ]
        await self._pump_resets()

        # Dropping a sandbox whose socket is still open MUST hang up on it. The sim only re-registers
        # from its socket's on-open, and only reconnects once it sees a close, so an eviction that
        # leaves the connection up takes that sandbox out of the pool permanently: it keeps
        # heartbeating happily into a registry that no longer has a row for it. That is not
        # hypothetical - it is how a fleet-wide reset hitch turned into every sim silently
        # disappearing and never coming back.
        if close_socket:
            self._close_socket_soon(sandbox.websocket, reason)

        if sandbox.lease_id is None:
            return

        # Tell the worker its machine vanished so it can requeue the attempt rather than sit on a
        # subprocess that will never talk to anything again.
        lease = self._leases.pop(sandbox.lease_id, None)
        if lease is not None:
            await _send(
                lease.holder,
                encode(
                    "bench.sandbox_lost",
                    lease_id=lease.lease_id,
                    sandbox_id=sandbox_id,
                    reason=reason,
                ),
            )

    # ------------------------------------------------------------------ bench side

    async def _handle_bench(self, websocket: Any) -> None:
        """One worker, at most one lease. Requests are handled in order on this connection."""
        try:
            async for raw in websocket:
                message = decode(raw)
                if message is None:
                    continue

                message_type = message.get("type")
                if message_type == "bench.acquire":
                    await self._handle_acquire(websocket, message)
                elif message_type == "bench.release":
                    await self._handle_release(websocket, message)
                elif message_type == "bench.status":
                    # `waiting` is the only place the parked acquires are visible from outside this
                    # process: a worker blocked in `bench.acquire` has nothing on disk and no lease,
                    # so without this count a full fleet and a queue behind it look identical.
                    await _send(websocket, encode("bench.pool", **self.pool_status()))
                elif message_type == "bench.capacity":
                    await self._handle_capacity(websocket, message)
                elif message_type == "bench.quarantine":
                    await self._handle_quarantine(websocket, message)
                elif message_type == "bench.unquarantine":
                    await self._handle_unquarantine(websocket, message)
                else:
                    await _send(
                        websocket,
                        encode("bench.error", reason="unknown_message", message_type=message_type),
                    )
        except Exception as error:  # noqa: BLE001 - one worker must not kill the coordinator
            self.log(f"Bench connection error: {error!r}")
        finally:
            await self._reap_leases_of(websocket, reason="worker_disconnected")
            self._drop_waiters_of(websocket)

    async def _handle_acquire(self, websocket: Any, message: dict[str, Any]) -> None:
        requested_alias = _human_label(message.get("lease_alias"), "", max_chars=120)
        sandbox = self._take_leasable()
        if sandbox is not None:
            lease = self._claim(sandbox, websocket, requested_alias)
        else:
            # No sandbox free: park until one is. The worker is blocked on our reply, which is
            # exactly the backpressure we want - it stops the fleet running ahead of itself.
            future: asyncio.Future[Lease] = asyncio.get_running_loop().create_future()
            setattr(future, "_bench_holder", websocket)
            setattr(future, "_lease_alias", requested_alias)
            self._waiters.append(future)
            closed_task: asyncio.Task[Any] | None = None
            try:
                wait_closed = getattr(websocket, "wait_closed", None)
                if callable(wait_closed):
                    closed_task = asyncio.create_task(wait_closed())
                    done, _pending = await asyncio.wait(
                        {future, closed_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if future not in done:
                        future.cancel()
                        self._waiters = [waiter for waiter in self._waiters if waiter is not future]
                        return
                lease = future.result()
            except asyncio.CancelledError:
                return
            finally:
                if closed_task is not None and not closed_task.done():
                    closed_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await closed_task
            sandbox = self._sandboxes.get(lease.sandbox_id)
            if sandbox is None:
                # The sandbox died between being handed to us and us resuming; _drop_sandbox has
                # already told this worker, so there is nothing left to reply.
                return

        await _send(
            sandbox.websocket,
            encode(
                "coord.lease",
                lease_id=lease.lease_id,
                lease_alias=lease.alias,
                sandbox_alias=sandbox.alias,
            ),
        )
        await _send(
            websocket,
            encode(
                "bench.lease",
                lease_id=lease.lease_id,
                lease_alias=lease.alias,
                sandbox_id=sandbox.sandbox_id,
                sandbox_alias=sandbox.alias,
                host=sandbox.host,
                port=sandbox.port,
                commands_uri=commands_uri(sandbox.host, sandbox.port),
            ),
        )
        self.log(f"Leased {sandbox.alias} ({sandbox.host}:{sandbox.port}) as {lease.alias}")

    async def _handle_quarantine(self, websocket: Any, message: dict[str, Any]) -> None:
        lease_id = message.get("lease_id")
        selector = str(message.get("sandbox") or message.get("sandbox_id") or "")
        sandbox: Sandbox | None = None

        if isinstance(lease_id, str) and lease_id:
            lease = self._leases.get(lease_id)
            if lease is None or lease.holder is not websocket:
                await _send(websocket, encode("bench.error", reason="unknown_lease"))
                return
            sandbox = self._sandboxes.get(lease.sandbox_id)
        elif selector:
            sandbox = self._resolve_sandbox(selector)

        if sandbox is None:
            await _send(websocket, encode("bench.error", reason="unknown_sandbox"))
            return

        reason = _human_label(message.get("reason"), "operator_quarantine", max_chars=240)
        source = _human_label(message.get("source"), "operator", max_chars=120)
        await self._quarantine_sandbox(sandbox, reason=reason, source=source)
        await _send(
            websocket,
            encode(
                "bench.quarantined",
                sandbox_id=sandbox.sandbox_id,
                sandbox_alias=sandbox.alias,
                reason=reason,
            ),
        )

    async def _handle_unquarantine(self, websocket: Any, message: dict[str, Any]) -> None:
        selector = str(message.get("sandbox") or message.get("sandbox_id") or "")
        sandbox_id = self._resolve_sandbox_id(selector) if selector else None
        if sandbox_id is None:
            await _send(websocket, encode("bench.error", reason="unknown_sandbox"))
            return

        self._quarantines.pop(sandbox_id, None)
        self._save_state()
        sandbox = self._sandboxes.get(sandbox_id)
        alias = self._sandbox_aliases.get(sandbox_id, sandbox_id)
        if sandbox is not None:
            sandbox.quarantined = False
            sandbox.quarantine_reason = ""
            sandbox.quarantine_source = ""
            sandbox.quarantined_at = ""
            # A recovered instance must finish a reset before it can be leased again.
            await self._queue_reset(sandbox_id, "operator", "unquarantined")
        await _send(
            websocket,
            encode(
                "bench.unquarantined",
                sandbox_id=sandbox_id,
                sandbox_alias=alias,
                connected=sandbox is not None,
            ),
        )
        self.log(f"Unquarantined {alias}")

    async def _quarantine_sandbox(
        self, sandbox: Sandbox, *, reason: str, source: str
    ) -> None:
        quarantined_at = datetime.now(timezone.utc).isoformat()
        self._quarantines[sandbox.sandbox_id] = {
            "reason": reason,
            "source": source,
            "quarantined_at": quarantined_at,
        }
        sandbox.quarantined = True
        sandbox.quarantine_reason = reason
        sandbox.quarantine_source = source
        sandbox.quarantined_at = quarantined_at
        self._save_state()

        self._active_resets.discard(sandbox.sandbox_id)
        self._pending_resets = [
            pending for pending in self._pending_resets if pending[0] != sandbox.sandbox_id
        ]
        lease = self._leases.pop(sandbox.lease_id, None) if sandbox.lease_id else None
        sandbox.lease_id = None
        sandbox.lease_alias = ""
        sandbox.reset_started_at = None
        sandbox.reset_reason = ""
        await self._pump_resets()

        # Release without reset: repeated resets of a protocol-broken player are the failure loop
        # quarantine is meant to stop.
        await _send(sandbox.websocket, encode("coord.release", reason="quarantined"))
        if lease is not None:
            await _send(
                lease.holder,
                encode(
                    "bench.sandbox_lost",
                    lease_id=lease.lease_id,
                    lease_alias=lease.alias,
                    sandbox_id=sandbox.sandbox_id,
                    sandbox_alias=sandbox.alias,
                    reason=f"quarantined:{reason}",
                ),
            )
        self.log(f"Quarantined {sandbox.alias} ({source}: {reason})")
        self._fulfil_waiters()

    async def _handle_capacity(self, websocket: Any, message: dict[str, Any]) -> None:
        """Update the in-memory active-lease ceiling and wake eligible queued work."""
        limit = message.get("limit")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
            await _send(websocket, encode("bench.error", reason="invalid_capacity"))
            return
        self._lease_limit = limit
        self._fulfil_waiters()
        await _send(websocket, encode("bench.capacity", **self.capacity_status()))
        label = "all" if limit is None else str(limit)
        self.log(f"Active lease cap set to {label}")

    async def _handle_release(self, websocket: Any, message: dict[str, Any]) -> None:
        lease_id = message.get("lease_id")
        lease = self._leases.pop(lease_id, None) if isinstance(lease_id, str) else None
        if lease is None:
            # Releasing an unknown lease is a no-op, not an error: a worker that already had its
            # lease reaped (sandbox died, TTL expired) still runs its release in a finally block.
            await _send(websocket, encode("bench.released", lease_id=lease_id, known=False))
            return

        outcome = message.get("outcome") or "unknown"
        await self._queue_reset(lease.sandbox_id, lease.lease_id, outcome)
        await _send(websocket, encode("bench.released", lease_id=lease.lease_id, known=True))

    async def _queue_reset(self, sandbox_id: str, lease_id: str, reason: str) -> None:
        sandbox = self._sandboxes.get(sandbox_id)
        if sandbox is None:
            return

        sandbox.lease_id = None
        sandbox.lease_alias = ""
        # Deliberately NOT marked ready here. The sandbox re-enters the pool only when it sends
        # sandbox.state{Ready}, i.e. once its reset has actually settled. Marking it ready now
        # would hand a mid-reset environment to the next attempt.
        sandbox.state = STATE_RESETTING
        sandbox.reset_reason = reason
        if sandbox_id not in self._active_resets and not any(
            pending[0] == sandbox_id for pending in self._pending_resets
        ):
            self._pending_resets.append((sandbox_id, lease_id, reason))
        await self._pump_resets()

    async def _pump_resets(self) -> None:
        """Starts queued resets up to the fleet-wide concurrency limit."""
        while self._pending_resets and len(self._active_resets) < self.max_concurrent_resets:
            sandbox_id, lease_id, reason = self._pending_resets.pop(0)
            sandbox = self._sandboxes.get(sandbox_id)
            if sandbox is None:
                continue
            self._active_resets.add(sandbox_id)
            sandbox.reset_started_at = time.monotonic()
            self.log(f"Resetting {sandbox.alias} (lease {lease_id}, reason={reason})")
            await _send(
                sandbox.websocket,
                encode("coord.reset", lease_id=lease_id, reason=reason),
            )

    def _close_socket_soon(self, websocket: Any, reason: str) -> None:
        """Hangs up in the background. The close handshake waits on a peer that may be wedged, and
        the reaper is the one calling this - it must not stall a sweep behind one bad connection."""

        async def close() -> None:
            with contextlib.suppress(Exception):
                # 1013 "try again later": the sim's reconnect backoff is the intended response.
                await websocket.close(code=1013, reason=reason[:120])

        task = asyncio.create_task(close())
        self._closers.add(task)
        task.add_done_callback(self._closers.discard)

    # ------------------------------------------------------------------ pool helpers

    def _take_leasable(self) -> Sandbox | None:
        if self._lease_limit is not None and len(self._leases) >= self._lease_limit:
            return None
        for sandbox in self._sandboxes.values():
            if sandbox.leasable:
                return sandbox
        return None

    def _claim(self, sandbox: Sandbox, holder: Any, requested_alias: str = "") -> Lease:
        """Marks a sandbox as taken. Must happen synchronously with picking it, so two workers
        cannot both be handed the same sandbox before either has resumed."""
        fallback = f"lease-{self._next_lease_alias:04d}"
        self._next_lease_alias += 1
        lease = Lease(
            lease_id=uuid.uuid4().hex,
            sandbox_id=sandbox.sandbox_id,
            holder=holder,
            alias=_human_label(requested_alias, fallback, max_chars=120),
        )
        self._leases[lease.lease_id] = lease
        sandbox.lease_id = lease.lease_id
        sandbox.lease_alias = lease.alias
        return lease

    def _fulfil_waiters(self) -> None:
        """Hands free sandboxes to parked workers, oldest waiter first."""
        while self._waiters:
            # Discard waiters whose worker went away while queued.
            if self._waiters[0].done():
                self._waiters.pop(0)
                continue

            sandbox = self._take_leasable()
            if sandbox is None:
                return

            future = self._waiters.pop(0)
            future.set_result(
                self._claim(
                    sandbox,
                    getattr(future, "_bench_holder", None),
                    getattr(future, "_lease_alias", ""),
                )
            )

    def _drop_waiters_of(self, websocket: Any) -> None:
        remaining: list[asyncio.Future[Lease]] = []
        for future in self._waiters:
            if getattr(future, "_bench_holder", None) is websocket and not future.done():
                future.cancel()
            else:
                remaining.append(future)
        self._waiters = remaining

    async def _reap_leases_of(self, websocket: Any, *, reason: str) -> None:
        for lease in [lease for lease in self._leases.values() if lease.holder is websocket]:
            self._leases.pop(lease.lease_id, None)
            self.log(f"Reaping lease {lease.lease_id} ({reason})")
            await self._queue_reset(lease.sandbox_id, lease.lease_id, reason)

    def pool_snapshot(self) -> list[dict[str, Any]]:
        return [sandbox.describe() for sandbox in self._sandboxes.values()]

    def capacity_status(self) -> dict[str, Any]:
        registered = len(self._sandboxes)
        eligible = sum(
            1
            for sandbox in self._sandboxes.values()
            if sandbox.store_loaded and not sandbox.quarantined
        )
        effective = eligible if self._lease_limit is None else min(self._lease_limit, eligible)
        return {
            "capacity_limit": self._lease_limit,
            "effective_capacity": effective,
            "active_leases": len(self._leases),
            "registered_sandboxes": registered,
            "eligible_sandboxes": eligible,
            "quarantined_sandboxes": sum(
                1 for sandbox in self._sandboxes.values() if sandbox.quarantined
            ),
        }

    def pool_status(self) -> dict[str, Any]:
        return {
            "sandboxes": self.pool_snapshot(),
            "waiting": len([waiter for waiter in self._waiters if not waiter.done()]),
            **self.capacity_status(),
        }

    # ------------------------------------------------------------------ reaper

    async def _reap_forever(self) -> None:
        while True:
            await asyncio.sleep(REAP_INTERVAL_SECONDS)
            try:
                await self._reap_once()
            except Exception as error:  # noqa: BLE001 - the reaper must never die
                self.log(f"Reaper error: {error!r}")

    async def _reap_once(self) -> None:
        now = time.monotonic()

        for sandbox_id, sandbox in list(self._sandboxes.items()):
            if now - sandbox.last_heartbeat > self.heartbeat_timeout:
                await self._drop_sandbox(sandbox_id, reason="heartbeat_timeout")

        for sandbox_id in list(self._active_resets):
            sandbox = self._sandboxes.get(sandbox_id)
            if (
                sandbox is not None
                and sandbox.reset_started_at is not None
                and now - sandbox.reset_started_at > self.reset_timeout
            ):
                self.log(
                    f"Sandbox {sandbox.alias} reset exceeded {self.reset_timeout:g}s; quarantining"
                )
                await self._quarantine_sandbox(
                    sandbox, reason="reset_timeout", source="coordinator_reaper"
                )

        # TTL only catches half-open worker connections; a clean disconnect is reaped immediately
        # in _handle_bench's finally block.
        for lease in list(self._leases.values()):
            if now - lease.created_at > self.lease_ttl:
                self._leases.pop(lease.lease_id, None)
                self.log(f"Lease {lease.lease_id} exceeded its TTL; reclaiming its sandbox")
                await self._queue_reset(lease.sandbox_id, lease.lease_id, "lease_ttl")

        self._fulfil_waiters()


async def _send(websocket: Any, payload: str) -> None:
    """Best-effort send: a peer that has gone away is handled by its own connection teardown."""
    try:
        await websocket.send(payload)
    except Exception:  # noqa: BLE001
        return


def _peer_host(websocket: Any) -> str | None:
    address = getattr(websocket, "remote_address", None)
    if isinstance(address, tuple) and address:
        return str(address[0])
    return None


def _print_log(message: str) -> None:
    print(f"[coordinator] {message}", flush=True)


async def async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Distributed Sari Bench sandbox registry.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: all interfaces).")
    parser.add_argument("--port", type=int, default=DEFAULT_COORDINATOR_PORT)
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=HEARTBEAT_TIMEOUT_SECONDS,
        help="Seconds without a heartbeat before a sandbox is dropped from the pool.",
    )
    parser.add_argument(
        "--lease-ttl",
        type=float,
        default=DEFAULT_LEASE_TTL_SECONDS,
        help="Seconds before an unreleased lease is reclaimed.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_COORDINATOR_STATE_PATH,
        help=(
            "Persistent aliases/quarantines JSON file "
            f"(default: {DEFAULT_COORDINATOR_STATE_PATH}; override with {COORDINATOR_STATE_ENV})."
        ),
    )
    args = parser.parse_args(argv)

    coordinator = Coordinator(
        host=args.host,
        port=args.port,
        heartbeat_timeout=args.heartbeat_timeout,
        lease_ttl=args.lease_ttl,
        state_path=args.state_file,
    )
    await coordinator.start()
    try:
        await asyncio.Future()  # run until interrupted
    except asyncio.CancelledError:
        pass
    finally:
        await coordinator.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
