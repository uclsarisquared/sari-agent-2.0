import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import struct
import tempfile

import websockets

MAGIC = b"LDR1"
HEADER_FORMAT = "<4sHHffffId"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
SANDBOX_FAULT_PATH_ENV = "SARI_SANDBOX_FAULT_PATH"


def _signal_protocol_fault(payload) -> None:
    path_text = os.environ.get(SANDBOX_FAULT_PATH_ENV, "").strip()
    if not path_text:
        return
    path = Path(path_text)
    preview = payload[:500] if isinstance(payload, str) else repr(payload)[:500]
    body = {
        "code": "lidar_protocol_nonbinary",
        "message": "RequestLidarScan returned a text frame; expected binary LiDAR data",
        "payload_type": type(payload).__name__,
        "response_preview": preview,
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(body, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
    except OSError:
        # Reporting must never hide the original protocol violation.
        return


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


def RequestLidarScan(uri: str = "ws://localhost:8080/commands") -> dict:
    payload = asyncio.get_event_loop().run_until_complete(_request_scan_async(uri))
    return parse_scan(payload)
