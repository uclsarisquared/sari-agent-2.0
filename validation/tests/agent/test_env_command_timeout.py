import asyncio
import json
import time

import pytest

from sim import env
from mapping.core import lidar_client


def test_send_command_times_out_when_sandbox_never_replies(monkeypatch):
    """MEASURED 2026-08-21: two live bench attempts sat parked in `await websocket.recv()` for
    over an hour because nothing below `SendCommand` ever bounded that wait. This is the
    regression test for the fix."""
    async def hangs_forever(_command, _uri):
        await asyncio.sleep(3600)

    monkeypatch.setattr(env, "_send_command_once", hangs_forever)

    with pytest.raises(env.SandboxCommandTimeout):
        asyncio.get_event_loop().run_until_complete(
            env.SendCommand({"command": "RequestScreenshot"}, "ws://sandbox/commands", timeout=0.01)
        )


def test_send_command_timeout_signals_the_fault_file(monkeypatch, tmp_path):
    async def hangs_forever(_command, _uri):
        await asyncio.sleep(3600)

    monkeypatch.setattr(env, "_send_command_once", hangs_forever)
    fault_path = tmp_path / "sandbox_fault.json"
    monkeypatch.setenv("SARI_SANDBOX_FAULT_PATH", str(fault_path))

    with pytest.raises(env.SandboxCommandTimeout):
        asyncio.get_event_loop().run_until_complete(
            env.SendCommand({"command": "TransformAgent"}, "ws://sandbox/commands", timeout=0.01)
        )

    payload = json.loads(fault_path.read_text())
    assert payload["code"] == "sandbox_command_timeout"
    assert payload["command"] == "TransformAgent"


def test_send_command_returns_normally_within_budget(monkeypatch):
    async def replies(_command, _uri):
        return "ok"

    monkeypatch.setattr(env, "_send_command_once", replies)

    result = asyncio.get_event_loop().run_until_complete(
        env.SendCommand({"command": "SetCrouch"}, "ws://sandbox/commands", timeout=5.0)
    )
    assert result == "ok"


@pytest.mark.parametrize(
    "server_error",
    [
        "Error: command 'RequestScreenshot' timed out after 8.00s in the coroutine queue.",
        "Error: command 'TransformAgent' was rejected because the coroutine queue is full "
        "(64 items).",
        "Error: LiDAR center GPU readback timed out after 6s",
    ],
)
def test_server_watchdog_errors_signal_infrastructure_fault(
    monkeypatch, tmp_path, server_error
):
    async def replies(_command, _uri):
        return server_error

    monkeypatch.setattr(env, "_send_command_once", replies)
    fault_path = tmp_path / "sandbox_fault.json"
    monkeypatch.setenv("SARI_SANDBOX_FAULT_PATH", str(fault_path))

    with pytest.raises(env.SandboxCommandTimeout, match="infrastructure failure"):
        asyncio.get_event_loop().run_until_complete(
            env.SendCommand(
                {"command": "RequestScreenshot"},
                "ws://sandbox/commands",
                timeout=10.0,
            )
        )

    payload = json.loads(fault_path.read_text())
    assert payload["code"] == "sandbox_command_timeout"
    assert payload["server_response"] == server_error


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_send_command_rejects_non_positive_and_non_finite_budgets(monkeypatch, timeout):
    async def should_not_send(_command, _uri):
        raise AssertionError("invalid timeout reached the websocket")

    monkeypatch.setattr(env, "_send_command_once", should_not_send)
    with pytest.raises(ValueError, match="positive finite"):
        asyncio.get_event_loop().run_until_complete(
            env.SendCommand({"command": "SetCrouch"}, "ws://sandbox/commands", timeout=timeout)
        )


@pytest.mark.parametrize("raw", ["not-a-number", "0", "-1", "nan", "inf"])
def test_environment_timeout_rejects_invalid_values(monkeypatch, raw):
    monkeypatch.setenv(env.SANDBOX_COMMAND_TIMEOUT_ENV, raw)
    with pytest.raises(ValueError, match="positive finite"):
        env.sandbox_command_timeout()


def test_local_response_processing_is_outside_the_websocket_budget(monkeypatch):
    async def replies(_command, _uri):
        return b"image"

    def slow_local_processing(_command, response):
        time.sleep(0.03)
        return {"image": response}

    monkeypatch.setattr(env, "_send_command_once", replies)
    monkeypatch.setattr(env, "_process_command_response", slow_local_processing)

    result = asyncio.get_event_loop().run_until_complete(
        env.SendCommand({"command": "RequestScreenshot"}, "ws://sandbox/commands", timeout=0.01)
    )
    assert result == {"image": b"image"}


def test_wait_for_ready_is_not_bounded_by_the_per_command_timeout(monkeypatch):
    """WaitUntilReady is deliberately held open by the sim until it is actually ready - it must
    keep using its own caller-owned deadline (`remaining`), not the short per-command default."""
    async def slow_but_ready(_command, _uri):
        await asyncio.sleep(0.05)
        return "Ready"

    monkeypatch.setattr(env, "_send_command_once", slow_but_ready)
    monkeypatch.setattr(env, "DEFAULT_SANDBOX_COMMAND_TIMEOUT", 0.01)

    assert env.wait_for_ready("ws://sandbox/commands", timeout=1.0) is True


def test_lidar_scan_uses_the_same_timeout_fault_path(monkeypatch, tmp_path):
    async def hangs_forever(_uri):
        await asyncio.sleep(3600)

    monkeypatch.setattr(lidar_client, "_request_scan_async", hangs_forever)
    fault_path = tmp_path / "sandbox_fault.json"
    monkeypatch.setenv("SARI_SANDBOX_FAULT_PATH", str(fault_path))

    with pytest.raises(env.SandboxCommandTimeout):
        lidar_client.RequestLidarScan("ws://sandbox/commands", timeout=0.01)

    payload = json.loads(fault_path.read_text())
    assert payload["code"] == "sandbox_command_timeout"
    assert payload["command"] == "RequestLidarScan"


def test_lidar_scan_promotes_server_watchdog_error(monkeypatch, tmp_path):
    server_error = "Error: LiDAR GPU readback timed out after 6s"

    async def replies(_uri):
        return server_error

    monkeypatch.setattr(lidar_client, "_request_scan_async", replies)
    fault_path = tmp_path / "sandbox_fault.json"
    monkeypatch.setenv("SARI_SANDBOX_FAULT_PATH", str(fault_path))

    with pytest.raises(env.SandboxCommandTimeout, match="infrastructure failure"):
        lidar_client.RequestLidarScan("ws://sandbox/commands", timeout=10.0)

    payload = json.loads(fault_path.read_text())
    assert payload["code"] == "sandbox_command_timeout"
    assert payload["server_response"] == server_error
