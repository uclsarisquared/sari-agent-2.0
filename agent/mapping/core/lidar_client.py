import asyncio
import json
import math
import struct

import websockets

from sim.sandbox_fault import signal_fault
from sim.env import (
    SandboxCommandTimeout,
    sandbox_command_timeout,
    sandbox_infrastructure_error,
)

MAGIC = b"LDR1"
HEADER_FORMAT = "<4sHHffffId"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def _signal_protocol_fault(payload) -> None:
    preview = payload[:500] if isinstance(payload, str) else repr(payload)[:500]
    signal_fault(
        "lidar_protocol_nonbinary",
        "RequestLidarScan returned a text frame; expected binary LiDAR data",
        payload_type=type(payload).__name__,
        response_preview=preview,
    )


async def _request_scan_async(uri):
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"command": "RequestLidarScan"}))
        return await ws.recv()


def parse_scan(payload: bytes) -> dict:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        _signal_protocol_fault(payload)
        raise TypeError(
            "RequestLidarScan returned a text frame; expected binary LiDAR data "
            f"(got {type(payload).__name__})"
        )
    if len(payload) < HEADER_SIZE:
        raise ValueError(f"Payload too short for LiDAR header: {len(payload)} bytes")

    (
        magic,
        channels,
        azimuth_samples,
        min_range,
        max_range,
        azimuth_start_deg,
        azimuth_step_deg,
        sequence,
        timestamp_seconds,
    ) = struct.unpack_from(HEADER_FORMAT, payload, 0)

    if magic != MAGIC:
        raise ValueError(f"Unexpected LiDAR magic: {magic!r}")

    offset = HEADER_SIZE
    vertical_angles = list(struct.unpack_from(f"<{channels}f", payload, offset))
    offset += channels * 4
    range_count = channels * azimuth_samples
    ranges = list(struct.unpack_from(f"<{range_count}f", payload, offset))

    return {
        "channels": channels,
        "azimuth_samples": azimuth_samples,
        "min_range": min_range,
        "max_range": max_range,
        "azimuth_start_deg": azimuth_start_deg,
        "azimuth_step_deg": azimuth_step_deg,
        "sequence": sequence,
        "timestamp_seconds": timestamp_seconds,
        "vertical_angles_deg": vertical_angles,
        "ranges": ranges,
    }


def RequestLidarScan(uri: str = "ws://localhost:8080/commands", timeout: float = None) -> dict:
    budget = sandbox_command_timeout() if timeout is None else timeout
    try:
        budget = float(budget)
    except (TypeError, ValueError) as error:
        raise ValueError(f"timeout must be a positive finite number, got {budget!r}") from error
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError(f"timeout must be a positive finite number, got {budget!r}")
    try:
        payload = asyncio.get_event_loop().run_until_complete(
            asyncio.wait_for(_request_scan_async(uri), timeout=budget)
        )
    except asyncio.TimeoutError as error:
        message = (
            f"RequestLidarScan: websocket round trip to {uri} did not complete within "
            f"{budget}s - the sandbox is presumed wedged"
        )
        signal_fault(
            "sandbox_command_timeout",
            message,
            command="RequestLidarScan",
            uri=uri,
            timeout=budget,
        )
        raise SandboxCommandTimeout(message) from error

    server_error = sandbox_infrastructure_error(payload)
    if server_error is not None:
        message = (
            f"RequestLidarScan: sandbox at {uri} reported an infrastructure failure: "
            f"{server_error}"
        )
        signal_fault(
            "sandbox_command_timeout",
            message,
            command="RequestLidarScan",
            uri=uri,
            timeout=budget,
            server_response=server_error,
        )
        raise SandboxCommandTimeout(message)

    return parse_scan(payload)
