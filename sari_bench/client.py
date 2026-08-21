"""Benchmark-worker side of the coordinator protocol.

One :class:`CoordinatorClient` per worker, one lease at a time. A background reader owns the socket
so the worker can notice ``bench.sandbox_lost`` *while* it is blocked waiting on an agent
subprocess - which is exactly when a sandbox is most likely to die, and the one moment a plain
request/response client would be deaf to it.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from sari_bench.protocol import BENCH_ROUTE, decode, encode

DEFAULT_ACQUIRE_TIMEOUT_SECONDS = 30.0


class SandboxLost(Exception):
    """Raised when the leased sandbox stops responding mid-attempt."""

    def __init__(self, sandbox_id: str, reason: str) -> None:
        super().__init__(f"sandbox {sandbox_id} lost: {reason}")
        self.sandbox_id = sandbox_id
        self.reason = reason


class Lease:
    """A leased sandbox and the address an agent should connect to."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.lease_id: str = payload["lease_id"]
        self.lease_alias: str = str(payload.get("lease_alias") or self.lease_id[:8])
        self.sandbox_id: str = payload["sandbox_id"]
        self.sandbox_alias: str = str(payload.get("sandbox_alias") or self.sandbox_id[:8])
        self.host: str = payload["host"]
        self.port: int = payload["port"]
        self.commands_uri: str = payload["commands_uri"]

    def __repr__(self) -> str:
        return f"Lease({self.sandbox_alias} @ {self.host}:{self.port})"


class CoordinatorClient:
    """Async client for one benchmark worker."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        if not self.url.endswith(BENCH_ROUTE):
            self.url = f"{self.url}{BENCH_ROUTE}"

        self._socket: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._replies: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._lost: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._closed = asyncio.Event()

    async def __aenter__(self) -> CoordinatorClient:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    async def connect(self) -> None:
        import websockets

        self._closed.clear()
        self._socket = await websockets.connect(self.url)
        self._reader = asyncio.create_task(self._read_forever())

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None

        if self._socket is not None:
            await self._socket.close()
            self._socket = None

    async def _read_forever(self) -> None:
        try:
            async for raw in self._socket:
                message = decode(raw)
                if message is None:
                    continue
                if message.get("type") == "bench.sandbox_lost":
                    await self._lost.put(message)
                else:
                    await self._replies.put(message)
        except Exception:  # noqa: BLE001 - surfaced to the worker as a closed connection
            pass
        finally:
            self._closed.set()

    async def _request(self, message: str, expect: str) -> dict[str, Any]:
        await self._socket.send(message)
        while True:
            reply_task = asyncio.create_task(self._replies.get())
            closed_task = asyncio.create_task(self._closed.wait())
            try:
                done, _pending = await asyncio.wait(
                    {reply_task, closed_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # Prefer a reply if it raced with the close notification: the peer may close
                # immediately after sending a complete response.
                if reply_task not in done:
                    raise ConnectionError(
                        f"Coordinator connection closed while waiting for {expect} ({self.url})"
                    )
                reply = reply_task.result()
            finally:
                for task in (reply_task, closed_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(reply_task, closed_task, return_exceptions=True)

            if reply.get("type") == expect:
                return reply
            if reply.get("type") == "bench.error":
                raise RuntimeError(f"Coordinator rejected request: {reply.get('reason')}")

    async def acquire(
        self,
        *,
        timeout: float | None = DEFAULT_ACQUIRE_TIMEOUT_SECONDS,
        lease_alias: str = "",
    ) -> Lease:
        """Lease a sandbox, optionally bounding how long the coordinator may park the request.

        A timed-out acquire makes this protocol connection unusable: the coordinator may still
        grant the request and send its reply later, which a subsequent request could mistake for
        its own response. Close the connection so the coordinator removes the waiter (and reaps a
        lease if the grant raced with the timeout). Callers that want to keep waiting must reconnect
        before trying again.
        """
        request = self._request(
            encode("bench.acquire", lease_alias=lease_alias), "bench.lease"
        )
        if timeout is None:
            return Lease(await request)
        if timeout <= 0:
            raise ValueError("acquire timeout must be positive")
        try:
            return Lease(await asyncio.wait_for(request, timeout=timeout))
        except asyncio.TimeoutError:
            await self.close()
            raise asyncio.TimeoutError(
                f"Timed out after {timeout:g}s waiting for bench.lease ({self.url})"
            ) from None

    async def release(self, lease: Lease, outcome: str) -> None:
        """Hands the sandbox back. The coordinator resets it before anyone else gets it."""
        await self.release_lease_id(lease.lease_id, outcome)

    async def release_lease_id(self, lease_id: str, outcome: str) -> bool:
        """Defensively release a persisted lease ID; unknown/reaped IDs are harmless."""
        reply = await self._request(
            encode("bench.release", lease_id=lease_id, outcome=outcome),
            "bench.released",
        )
        return bool(reply.get("known"))

    async def pool(self) -> list[dict[str, Any]]:
        reply = await self._request(encode("bench.status"), "bench.pool")
        return list(reply.get("sandboxes") or [])

    async def pool_status(self) -> tuple[list[dict[str, Any]], int]:
        """The pool plus how many acquires are parked behind it.

        Separate from `pool` because only the dashboard needs the second number, and a coordinator
        too old to report it must read as "no queue" rather than as an error.
        """
        reply = await self._request(encode("bench.status"), "bench.pool")
        try:
            waiting = max(0, int(reply.get("waiting") or 0))
        except (TypeError, ValueError):
            waiting = 0
        return list(reply.get("sandboxes") or []), waiting

    async def fleet_status(self) -> dict[str, Any]:
        """Return the pool and additive coordinator-capacity fields.

        Defaults preserve useful values when connected to a coordinator predating capacity caps.
        """
        reply = await self._request(encode("bench.status"), "bench.pool")
        sandboxes = list(reply.get("sandboxes") or [])
        active = sum(1 for sandbox in sandboxes if sandbox.get("lease_id"))
        eligible = sum(1 for sandbox in sandboxes if bool(sandbox.get("store_loaded", True)))

        def count(name: str, default: int) -> int:
            try:
                return max(0, int(reply.get(name, default)))
            except (TypeError, ValueError):
                return default

        return {
            "sandboxes": sandboxes,
            "waiting": count("waiting", 0),
            "capacity_limit": reply.get("capacity_limit"),
            "effective_capacity": count("effective_capacity", eligible),
            "active_leases": count("active_leases", active),
            "registered_sandboxes": count("registered_sandboxes", len(sandboxes)),
            "eligible_sandboxes": count("eligible_sandboxes", eligible),
            "quarantined_sandboxes": count(
                "quarantined_sandboxes",
                sum(1 for sandbox in sandboxes if sandbox.get("quarantined")),
            ),
        }

    async def quarantine(
        self, lease: Lease, *, reason: str, source: str = "runner"
    ) -> dict[str, Any]:
        """Quarantine the sandbox currently held by this connection."""
        return await self._request(
            encode(
                "bench.quarantine",
                lease_id=lease.lease_id,
                reason=reason,
                source=source,
            ),
            "bench.quarantined",
        )

    async def quarantine_sandbox(
        self, selector: str, *, reason: str, source: str = "operator"
    ) -> dict[str, Any]:
        """Quarantine a connected sandbox by its human alias or canonical ID."""
        return await self._request(
            encode(
                "bench.quarantine", sandbox=selector, reason=reason, source=source
            ),
            "bench.quarantined",
        )

    async def unquarantine(self, selector: str) -> dict[str, Any]:
        """Clear quarantine by human alias or canonical sandbox ID."""
        return await self._request(
            encode("bench.unquarantine", sandbox=selector),
            "bench.unquarantined",
        )

    async def set_capacity(self, limit: int | None) -> dict[str, Any]:
        """Set the coordinator-wide active lease cap (None means unlimited)."""
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
            raise ValueError("capacity limit must be a non-negative integer or None")
        reply = await self._request(
            encode("bench.capacity", limit=limit), "bench.capacity"
        )
        return {
            "capacity_limit": reply.get("capacity_limit"),
            "effective_capacity": int(reply.get("effective_capacity") or 0),
            "active_leases": int(reply.get("active_leases") or 0),
            "registered_sandboxes": int(reply.get("registered_sandboxes") or 0),
            "eligible_sandboxes": int(reply.get("eligible_sandboxes") or 0),
        }

    async def wait_for_sandbox_lost(self, lease: Lease) -> SandboxLost:
        """Resolves only if *this* lease's sandbox dies. Race it against the attempt."""
        while True:
            message = await self._lost.get()
            if message.get("lease_id") == lease.lease_id:
                return SandboxLost(
                    str(message.get("sandbox_id") or lease.sandbox_id),
                    str(message.get("reason") or "unknown"),
                )
