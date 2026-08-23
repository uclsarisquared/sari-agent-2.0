"""Runner tests: the full lease -> spawn -> release cycle against a stub agent.

The stub stands in for run_agent.py (importing the real one pulls the whole model
stack). It writes the same summary.json the orchestrator does, so the result-folding path is
exercised for real. What is being pinned down:

  1. prompts x tries all run, and each attempt lands in its own run dir;
  2. automatic concurrency fills the pool and grows when another sandbox joins;
  3. explicit --concurrency remains a hard cap;
  4. an attempt that overruns its time limit is killed and recorded, not left hanging;
  5. a crashed agent still releases its sandbox, so the battery finishes.

    python sari_bench/tests/test_runner.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import socket
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sari_bench.coordinator import Coordinator
from sari_bench.client import CoordinatorClient
from sari_bench import capture
from sari_bench.runner import (
    BenchmarkRunner,
    OcrPreflightError,
    ORCHESTRATOR_ENTRY,
    Prompt,
    ResumeError,
    load_prompts,
)
from sari_bench.tests.test_coordinator import FakeSandbox
from sari_bench.watch.notify import Discord
from sari_bench.watch.server import WatchState
from sari_bench.watch import scan

# Records its own liveness window so the test can prove two agents overlapped (or did not), and
# honours the run-dir contract the runner reads results back out of.
STUB_AGENT = '''
import json, os, sys, time

run_dir = sys.argv[sys.argv.index("--run-dir") + 1]
task = sys.argv[sys.argv.index("--task") + 1]
os.makedirs(run_dir, exist_ok=True)

with open(os.path.join(run_dir, "liveness.json"), "w") as f:
    json.dump({"start": time.time(), "pid": os.getpid(),
               "uri": os.environ.get("SARI_WS_URI", ""),
               "ocr_url": os.environ.get("SARI_OCR_URL", "")}, f)

# The real agent's token_meter rewrites this every few seconds, so it exists even for an attempt
# that never reaches summary.json.
with open(os.path.join(run_dir, "tokens.json"), "w") as f:
    json.dump({"tokens_in": 1200, "tokens_out": 300, "calls": 4, "api_calls": 5,
               "by_role": {"actor": {"tokens_in": 800, "tokens_out": 200, "calls": 2,
                                      "api_calls": 3},
                           "guard": {"tokens_in": 400, "tokens_out": 100, "calls": 2,
                                      "api_calls": 2}}}, f)

mode = os.environ.get("STUB_MODE", "ok")
if mode == "crash":
    sys.exit(3)
if mode == "hang":
    time.sleep(600)
if mode == "api_retry_exhausted":
    with open(os.environ["SARI_API_RETRY_EXHAUSTED_PATH"], "w") as f:
        json.dump({"attempts": 10, "error_type": "TimeoutError",
                   "error": "server stayed down", "call_name": "semantic_reasoning",
                   "failure_kind": "timeout"}, f)
    time.sleep(600)
if mode == "sandbox_fault" and "51001" in os.environ.get("SARI_WS_URI", ""):
    with open(os.environ["SARI_SANDBOX_FAULT_PATH"], "w") as f:
        json.dump({"code": "lidar_protocol_nonbinary",
                   "message": "RequestLidarScan returned a text frame"}, f)
    time.sleep(600)
if mode == "cancel_siblings" and task == "winner prompt" and not run_dir.endswith("try01"):
    time.sleep(600)

time.sleep(float(os.environ.get("STUB_SLEEP", "0.3")))
with open(os.path.join(run_dir, "summary.json"), "w") as f:
    json.dump({"task": task, "success": True, "legs_planned": 1, "legs_completed": 1,
               "response": "Done — the requested task was completed.",
               "response_source": "model",
               "llm_calls": 6, "api_calls": 8,
               "tokens_in": 2000, "tokens_out": 500,
               "tokens": {"tokens_in": 2000, "tokens_out": 500, "calls": 6,
                          "api_calls": 8,
                          "by_role": {"actor": {"tokens_in": 1400, "tokens_out": 350,
                                                  "calls": 4, "api_calls": 5},
                                      "guard": {"tokens_in": 600, "tokens_out": 150,
                                                  "calls": 2, "api_calls": 3}}},
               "legs": [{"end_reason": "halt_granted", "success": True,
                         "tokens_in": 2000, "tokens_out": 500, "api_calls": 8}]}, f)
with open(os.path.join(run_dir, "liveness.json"), "r+") as f:
    data = json.load(f); data["end"] = time.time()
    f.seek(0); json.dump(data, f); f.truncate()
'''


def test_default_agent_entry_uses_the_public_launcher() -> None:
    assert ORCHESTRATOR_ENTRY == "run_agent.py"


async def _start_coordinator() -> tuple[Coordinator, str]:
    coordinator = Coordinator(host="127.0.0.1", port=0, log=lambda _m: None)
    await coordinator.start()
    return coordinator, f"ws://127.0.0.1:{coordinator.bound_port}"


def _runner(url: str, prompts: list[Prompt], workspace: Path, **kwargs) -> BenchmarkRunner:
    stub = workspace / "stub_agent.py"
    stub.write_text(STUB_AGENT, encoding="utf-8")
    options = dict(
        tries=1,
        time_limit_minutes=1.0,
        concurrency=None,
        max_steps=10,
        arm="graph",
        map_dir=None,
        leg_retries=0,
        timeout_grace=1.0,
        capture_interval=0.0,
        capacity_poll_interval=0.05,
        ocr_health_check=lambda _url: {
            "ready": True,
            "api_version": "v1",
            "model": "fake-ocr",
        },
    )
    options.update(kwargs)
    return BenchmarkRunner(
        prompts=prompts,
        coordinator_url=url,
        output_dir=workspace / "runs",
        agent_entry=str(stub),
        agent_cwd=workspace,
        **options,
    )


def test_load_prompts_accepts_the_battery_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "battery.json"
        path.write_text(
            json.dumps(
                {
                    "prompts": [
                        {"id": "p1", "family": "pickup", "prompt": "Pick up soy sauce",
                         "looking_for": "a bottle"},
                        {"id": "p2", "prompt": "Go to the checkout"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        prompts = load_prompts(path)
        assert [p.id for p in prompts] == ["p1", "p2"]
        assert prompts[0].family == "pickup"

        # A bare list, and bare strings, are both accepted.
        path.write_text(json.dumps(["do a thing", {"prompt": "do another"}]), encoding="utf-8")
        assert [p.id for p in load_prompts(path)] == ["prompt_00", "prompt_01"]

        path.write_text(json.dumps([{"id": "dupe", "prompt": "a"}, {"id": "dupe", "prompt": "b"}]),
                        encoding="utf-8")
        try:
            load_prompts(path)
        except ValueError as error:
            assert "duplicate" in str(error)
        else:
            raise AssertionError("duplicate prompt ids were accepted")
    print("ok  prompt batteries load in every documented shape")


def test_automatic_retries_outrank_fresh_work_fifo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        runner = _runner(
            "ws://127.0.0.1:1",
            [Prompt(id=name, prompt=name) for name in ("api", "lost", "fresh-a", "fresh-b")],
            workspace,
        )
        runner._enqueue_work("fresh-a", 1, priority=1)
        runner._enqueue_work("fresh-b", 1, priority=1)
        for prompt_id, reason in (("api", "api_retry_exhausted"), ("lost", "sandbox_lost")):
            run_dir = runner.output_dir / prompt_id / "try01"
            run_dir.mkdir(parents=True)
            (run_dir / "attempt.json").write_text("{}", encoding="utf-8")
            runner._schedule_requeue(
                run_dir,
                reason=reason,
                item=(prompt_id, 1, int(prompt_id == "api"), int(prompt_id == "lost")),
                wall_seconds=3.0,
            )

        ordered = [runner._queue.get_nowait().prompt_id for _ in range(4)]
        assert ordered == ["api", "lost", "fresh-a", "fresh-b"], ordered
    print("ok  API and sandbox-loss retries outrank fresh work and remain FIFO")


def test_completion_guard_is_threaded_into_agent_command_and_battery_config() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        runner = _runner(
            "ws://127.0.0.1:1",
            [Prompt(id="p", prompt="pick it up")],
            workspace,
            completion_guard="vlm",
            context_policy="a5",
            api_max_attempts=6,
            max_api_requeues=2,
            sandbox_command_timeout=17.0,
        )
        lease = type("LeaseStub", (), {"commands_uri": "ws://127.0.0.1:51001/commands"})()
        command = runner._agent_command(
            runner.prompts["p"], lease, workspace / "runs" / "p" / "try01")
        index = command.index("--completion-guard")
        assert command[index + 1] == "vlm", command
        assert runner._semantic_config()["completion_guard"] == "vlm"
        retry_index = command.index("--api-max-attempts")
        assert command[retry_index + 1] == "6"
        assert runner._semantic_config()["api_max_attempts"] == 6
        assert runner._semantic_config()["max_api_requeues"] == 2
        policy_index = command.index("--context-policy")
        assert command[policy_index + 1] == "a5"
        assert runner._semantic_config()["context_policy"] == "a5"
        ocr_index = command.index("--ocr-url")
        assert command[ocr_index + 1] == "http://127.0.0.1:9100"
        assert runner._semantic_config()["ocr_url"] == "http://127.0.0.1:9100"
        assert runner._semantic_config()["sandbox_command_timeout_seconds"] == 17.0
        runner.output_dir.mkdir(parents=True)
        runner._write_battery_manifest(1)
        battery = json.loads((runner.output_dir / "battery.json").read_text())
        assert battery["completion_guard"] == "vlm"
        assert battery["context_policy"] == "a5"
        assert battery["api_max_attempts"] == 6
        assert battery["max_api_requeues"] == 2
        assert battery["ocr_url"] == "http://127.0.0.1:9100"
        assert battery["sandbox_command_timeout_seconds"] == 17.0

        changed_timeout = runner._semantic_config()
        changed_timeout["sandbox_command_timeout_seconds"] = 18.0
        try:
            runner._validate_resume_config(changed_timeout)
        except ResumeError:
            pass
        else:
            raise AssertionError("a battery resumed with a different sandbox command timeout")

        disabled = _runner(
            "ws://127.0.0.1:1",
            [Prompt(id="p", prompt="pick it up")],
            workspace,
            completion_guard="none",
        )
        disabled_command = disabled._agent_command(
            disabled.prompts["p"], lease, workspace / "runs" / "p" / "try02")
        disabled_index = disabled_command.index("--completion-guard")
        assert disabled_command[disabled_index + 1] == "none"
        assert disabled._semantic_config()["completion_guard"] == "none"

        # A legacy battery without this field means deterministic, never an implicit VLM switch.
        legacy = runner._semantic_config()
        legacy.pop("completion_guard")
        legacy["context_policy"] = "a5"
        try:
            runner._validate_resume_config(legacy)
        except ResumeError:
            pass
        else:
            raise AssertionError("a legacy deterministic battery resumed with the VLM guard")

        legacy_workspace = workspace / "legacy"
        legacy_workspace.mkdir()
        deterministic = _runner(
            "ws://127.0.0.1:1",
            [Prompt(id="p", prompt="pick it up")],
            legacy_workspace,
        )
        legacy = deterministic._semantic_config()
        legacy.pop("completion_guard")
        legacy.pop("context_policy")
        legacy.pop("sandbox_command_timeout_seconds")
        deterministic._validate_resume_config(legacy)
    print("ok  completion guard reaches the agent command and durable battery config")


def test_refusal_cap_action_is_threaded_and_resume_checked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        runner = _runner(
            "ws://127.0.0.1:1",
            [Prompt(id="p", prompt="pick it up")],
            workspace,
            refusal_cap_action="halt",
        )
        lease = type("LeaseStub", (), {"commands_uri": "ws://127.0.0.1:51001/commands"})()
        command = runner._agent_command(
            runner.prompts["p"], lease, workspace / "runs" / "p" / "try01")
        index = command.index("--refusal-cap-action")
        assert command[index + 1] == "halt", command
        assert runner._semantic_config()["refusal_cap_action"] == "halt"

        # The default is continue, and a battery predating the option ran with halt semantics: a
        # resume must refuse rather than silently switch what a refusal cap does to the attempt.
        second_workspace = workspace / "continuing"
        second_workspace.mkdir()
        continuing = _runner(
            "ws://127.0.0.1:1",
            [Prompt(id="p", prompt="pick it up")],
            second_workspace,
            refusal_cap_action="continue",
        )
        legacy = continuing._semantic_config()
        legacy.pop("refusal_cap_action")
        try:
            continuing._validate_resume_config(legacy)
        except ResumeError:
            pass
        else:
            raise AssertionError("a legacy halting battery resumed as continue")

        bad_workspace = workspace / "bad"
        bad_workspace.mkdir()
        try:
            _runner("ws://127.0.0.1:1", [Prompt(id="p", prompt="x")],
                    bad_workspace, refusal_cap_action="skip")
        except ValueError:
            pass
        else:
            raise AssertionError("an unsupported refusal cap action was accepted")
    print("ok  refusal cap action reaches the agent command and is resume-checked")


def test_adaptive_replanning_is_threaded_and_resume_checked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        runner = _runner(
            "ws://127.0.0.1:1", [Prompt(id="p", prompt="pick it up")], workspace,
            adaptive_leg_replanning=True,
        )
        command = runner._agent_command(
            runner.prompts["p"],
            SimpleNamespace(commands_uri="ws://sandbox/commands"),
            workspace / "attempt",
        )
        assert "--adaptive-leg-replanning" in command
        assert runner._semantic_config()["adaptive_leg_replanning"] is True
        runner.output_dir.mkdir(parents=True, exist_ok=True)
        runner._write_battery_manifest(1)
        battery = json.loads((runner.output_dir / "battery.json").read_text())
        assert battery["adaptive_leg_replanning"] is True

        disabled = _runner(
            "ws://127.0.0.1:1", [Prompt(id="p", prompt="pick it up")],
            workspace, adaptive_leg_replanning=False,
        )
        try:
            disabled._validate_resume_config(battery)
        except ResumeError:
            pass
        else:
            raise AssertionError("adaptive battery resumed with replanning disabled")

        legacy = dict(battery)
        legacy.pop("adaptive_leg_replanning")
        disabled._validate_resume_config(legacy)
    print("ok  adaptive replanning reaches subprocesses and is resume-checked")


async def test_ocr_preflight_fails_before_coordinator_or_lease() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        reached_coordinator = False

        def fail(_url):
            from agent.vision.ocr_client import OcrUnavailable

            raise OcrUnavailable("OCR is down")

        runner = _runner(
            "ws://127.0.0.1:1",
            [Prompt(id="p", prompt="pick it up")],
            workspace,
            ocr_health_check=fail,
        )

        async def coordinator_probe():
            nonlocal reached_coordinator
            reached_coordinator = True

        runner._wait_for_registered_sandbox = coordinator_probe
        try:
            await runner.run()
        except OcrPreflightError as error:
            assert "OCR is down" in str(error)
        else:
            raise AssertionError("battery proceeded without OCR")
        assert reached_coordinator is False
    print("ok  OCR preflight fails before coordinator contact or sandbox leasing")


async def test_battery_runs_every_prompt_and_attempt() -> None:
    coordinator, url = await _start_coordinator()
    sandboxes = [FakeSandbox("sandbox-a", 51001), FakeSandbox("sandbox-b", 51002)]
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            for sandbox in sandboxes:
                await sandbox.connect(url)

            prompts = [Prompt(id="p1", prompt="task one"), Prompt(id="p2", prompt="task two")]
            summary = await _runner(url, prompts, workspace, tries=2).run()

            assert summary["total_attempts"] == 4, summary["total_attempts"]
            assert summary["total_successes"] == 4, summary["total_successes"]
            assert {row["prompt_id"] for row in summary["prompts"]} == {"p1", "p2"}
            assert all(row["success_rate"] == 1.0 for row in summary["prompts"])
            assert all(row["end_reason"] == "halt_granted" for row in summary["attempts"])

            # Every attempt got its own directory, and the agent was pointed at a real sandbox.
            for prompt_id in ("p1", "p2"):
                for attempt in (1, 2):
                    liveness = workspace / "runs" / prompt_id / f"try{attempt:02d}" / "liveness.json"
                    assert liveness.exists(), liveness
                    live = json.loads(liveness.read_text())
                    assert live["uri"].endswith("/commands")
                    assert live["ocr_url"] == "http://127.0.0.1:9100"
                    manifest = json.loads((liveness.parent / "attempt.json").read_text())
                    assert manifest["ocr_url"] == "http://127.0.0.1:9100"

            # Reset-per-attempt: four attempts over two sandboxes means four resets.
            assert sum(s.reset_count for s in sandboxes) == 4

            # The incremental log is a complete record on its own.
            lines = (workspace / "runs" / "attempts.jsonl").read_text().strip().splitlines()
            assert len(lines) == 4
        finally:
            for sandbox in sandboxes:
                await sandbox.close()
            await coordinator.stop()
    print("ok  every prompt x attempt runs, in its own dir, with a reset between each")


async def test_pool_size_bounds_concurrency() -> None:
    """One sandbox must mean one agent at a time, however many workers are configured."""
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            await sandbox.connect(url)
            prompts = [Prompt(id=f"p{i}", prompt=f"task {i}") for i in range(3)]
            os.environ["STUB_SLEEP"] = "0.5"
            try:
                await _runner(url, prompts, workspace, concurrency=3).run()
            finally:
                os.environ.pop("STUB_SLEEP", None)

            windows = sorted(
                (json.loads(path.read_text())["start"], json.loads(path.read_text())["end"])
                for path in (workspace / "runs").rglob("liveness.json")
            )
            assert len(windows) == 3
            for (_, earlier_end), (later_start, _) in zip(windows, windows[1:]):
                assert later_start >= earlier_end, "two agents shared one sandbox"
        finally:
            await sandbox.close()
            await coordinator.stop()
    print("ok  a one-sandbox pool serialises attempts even at concurrency 3")


async def test_automatic_concurrency_fills_and_grows_with_the_fleet() -> None:
    """An empty-start run begins when a sandbox joins, then expands while attempts are active."""
    coordinator, url = await _start_coordinator()
    first = FakeSandbox("sandbox-a", 51001)
    second = FakeSandbox("sandbox-b", 51002)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        prompts = [Prompt(id=f"p{i}", prompt=f"task {i}") for i in range(3)]
        os.environ["STUB_SLEEP"] = "1.0"
        runner_task = asyncio.create_task(_runner(url, prompts, workspace).run())
        try:
            await asyncio.sleep(0.15)
            assert not list((workspace / "runs").rglob("liveness.json"))

            await first.connect(url)
            first_liveness = workspace / "runs" / "p0" / "try01" / "liveness.json"
            for _ in range(100):
                if first_liveness.exists():
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("automatic runner did not use the first joined sandbox")

            await second.connect(url)
            for _ in range(100):
                liveness = list((workspace / "runs").rglob("liveness.json"))
                if len(liveness) >= 2:
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("automatic runner did not grow for the second sandbox")

            windows = [json.loads(path.read_text()) for path in liveness]
            assert any("end" not in window for window in windows), windows
            summary = await asyncio.wait_for(runner_task, timeout=30)
            assert summary["concurrency"] == "auto"
            assert summary["concurrency_mode"] == "auto"
            assert summary["concurrency_limit"] is None
            assert summary["peak_workers"] == 2
            battery = json.loads((workspace / "runs" / "battery.json").read_text())
            assert battery["peak_workers"] == 2
        finally:
            os.environ.pop("STUB_SLEEP", None)
            if not runner_task.done():
                runner_task.cancel()
                await asyncio.gather(runner_task, return_exceptions=True)
            await first.close()
            await second.close()
            await coordinator.stop()
    print("ok  automatic concurrency fills an empty fleet and grows when a sandbox joins")


async def test_explicit_concurrency_remains_a_hard_cap() -> None:
    coordinator, url = await _start_coordinator()
    sandboxes = [FakeSandbox("sandbox-a", 51001), FakeSandbox("sandbox-b", 51002)]
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            for sandbox in sandboxes:
                await sandbox.connect(url)
            os.environ["STUB_SLEEP"] = "0.4"
            try:
                summary = await _runner(
                    url,
                    [Prompt(id=f"p{i}", prompt=f"task {i}") for i in range(3)],
                    workspace,
                    concurrency=1,
                ).run()
            finally:
                os.environ.pop("STUB_SLEEP", None)

            windows = sorted(
                (json.loads(path.read_text())["start"], json.loads(path.read_text())["end"])
                for path in (workspace / "runs").rglob("liveness.json")
            )
            for (_, earlier_end), (later_start, _) in zip(windows, windows[1:]):
                assert later_start >= earlier_end, "explicit concurrency=1 was exceeded"
            assert summary["concurrency"] == 1
            assert summary["concurrency_mode"] == "fixed"
            assert summary["concurrency_limit"] == 1
            assert summary["peak_workers"] == 1
        finally:
            for sandbox in sandboxes:
                await sandbox.close()
            await coordinator.stop()
    print("ok  explicit concurrency remains a hard cap")


async def test_startup_timeout_explains_an_empty_pool() -> None:
    coordinator, url = await _start_coordinator()
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            runner = _runner(
                url,
                [Prompt(id="p1", prompt="cannot start")],
                workspace,
                sandbox_startup_timeout=0.2,
            )
            try:
                await runner.run()
            except RuntimeError as error:
                message = str(error)
                assert "No usable sandbox registered" in message, message
                assert "SARI_BENCH_COORDINATOR" in message, message
            else:
                raise AssertionError("an empty pool left the runner waiting forever")
            assert not (workspace / "runs" / "battery.json").exists()
        finally:
            await coordinator.stop()
    print("ok  an empty pool fails with sandbox startup instructions")


async def test_overrunning_attempt_is_killed_and_recorded() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            await sandbox.connect(url)
            os.environ["STUB_MODE"] = "hang"
            try:
                runner = _runner(
                    url,
                    [Prompt(id="p1", prompt="never finishes")],
                    workspace,
                    time_limit_minutes=1.0 / 60.0,  # one second
                    timeout_grace=0.5,
                )
                summary = await asyncio.wait_for(runner.run(), timeout=60)
            finally:
                os.environ.pop("STUB_MODE", None)

            assert summary["total_attempts"] == 1
            assert summary["attempts"][0]["outcome"] == "harness_timeout"
            assert summary["total_successes"] == 0
            # Killed or not, the sandbox has to come back to the pool clean.
            assert sandbox.reset_count == 1
        finally:
            await sandbox.close()
            await coordinator.stop()
    print("ok  an attempt past its time limit is killed, recorded, and its sandbox reset")


async def test_crashed_agent_still_releases_its_sandbox() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            await sandbox.connect(url)
            os.environ["STUB_MODE"] = "crash"
            try:
                prompts = [Prompt(id="p1", prompt="crashes"), Prompt(id="p2", prompt="also crashes")]
                summary = await asyncio.wait_for(
                    _runner(url, prompts, workspace, concurrency=1).run(), timeout=60
                )
            finally:
                os.environ.pop("STUB_MODE", None)

            assert summary["total_attempts"] == 2
            assert all(row["outcome"] == "agent_error" for row in summary["attempts"])
            assert all(row["exit_code"] == 3 for row in summary["attempts"])
            assert summary["total_successes"] == 0
            assert all(row["success"] is False for row in summary["attempts"])
            manifests = [
                json.loads((workspace / "runs" / prompt / "try01" / "attempt.json").read_text())
                for prompt in ("p1", "p2")
            ]
            assert all(manifest["success"] is False for manifest in manifests)
            # The second attempt only ran because the first released despite crashing.
            assert sandbox.reset_count == 2
        finally:
            await sandbox.close()
            await coordinator.stop()
    print("ok  a crashed agent still releases, so the battery finishes")


async def test_api_retry_exhaustion_requeues_the_logical_attempt() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            await sandbox.connect(url)
            os.environ["STUB_MODE"] = "api_retry_exhausted"
            try:
                summary = await asyncio.wait_for(
                    _runner(
                        url,
                        [Prompt(id="p1", prompt="wait for the endpoint")],
                        workspace,
                        concurrency=1,
                    ).run(),
                    timeout=60,
                )
            finally:
                os.environ.pop("STUB_MODE", None)

            attempt = summary["attempts"][0]
            assert attempt["outcome"] == "api_retry_exhausted", attempt
            assert attempt["requeues"] == 3, attempt
            assert attempt["success"] is False
            run_parent = workspace / "runs" / "p1"
            assert all(
                (run_parent / f"try01.requeue{index:02d}").is_dir()
                for index in range(3)
            )
            assert sandbox.reset_count == 4
        finally:
            await sandbox.close()
            await coordinator.stop()
    print("ok  exhausted API retries requeue the logical attempt with a finite budget")


async def test_sandbox_protocol_fault_quarantines_and_requeues() -> None:
    coordinator, url = await _start_coordinator()
    faulty = FakeSandbox("sandbox-faulty", 51001)
    replacement = FakeSandbox("sandbox-healthy", 51002)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            await faulty.connect(url)
            await replacement.connect(url)
            os.environ["STUB_MODE"] = "sandbox_fault"
            try:
                summary = await asyncio.wait_for(
                    _runner(
                        url,
                        [Prompt(id="p1", prompt="survive bad lidar")],
                        workspace,
                        concurrency=1,
                    ).run(),
                    timeout=20,
                )
            finally:
                os.environ.pop("STUB_MODE", None)

            assert summary["total_successes"] == 1
            assert summary["attempts"][0]["sandbox_id"] == "sandbox-healthy"
            pool = {row["sandbox_id"]: row for row in coordinator.pool_snapshot()}
            assert pool["sandbox-faulty"]["quarantined"] is True
            assert pool["sandbox-faulty"]["quarantine_reason"].startswith(
                "lidar_protocol_nonbinary"
            )
            battery = json.loads((workspace / "runs" / "battery.json").read_text())
            assert "sandbox-faulty" in battery["quarantined_sandboxes"]
        finally:
            await faulty.close()
            await replacement.close()
            await coordinator.stop()
    print("ok  a sandbox protocol fault quarantines the player and requeues on a healthy one")


async def test_command_timeout_recovery_resets_and_probes_the_failed_lane() -> None:
    seen: list[str] = []

    async def round_trip(_uri: str, payload: dict, _timeout: float):
        command = str(payload.get("command") or "")
        seen.append(command)
        if command == "ResetEnvironment":
            return "Environment reset"
        if command == "RequestScreenshot":
            return b"\x89PNG\r\n\x1a\nvalid-test-png"
        return f"Error: unexpected command {command}"

    with tempfile.TemporaryDirectory() as tmp:
        runner = _runner(
            "ws://127.0.0.1:1",
            [Prompt(id="p1", prompt="recover")],
            Path(tmp),
        )
        runner._sandbox_round_trip = round_trip
        recovered, detail = await runner._recover_timed_out_sandbox(
            "ws://sandbox/commands",
            "RequestScreenshot",
        )
        assert recovered is True, detail
        assert seen == ["ResetEnvironment", "RequestScreenshot"]
    print("ok  command timeout recovery resets before probing the failed serialized lane")


async def test_command_timeout_recovery_rejects_a_failed_probe() -> None:
    async def round_trip(_uri: str, payload: dict, _timeout: float):
        if payload.get("command") == "ResetEnvironment":
            return "Environment reset"
        return "Error: probe still wedged"

    with tempfile.TemporaryDirectory() as tmp:
        runner = _runner(
            "ws://127.0.0.1:1",
            [Prompt(id="p1", prompt="quarantine")],
            Path(tmp),
        )
        runner._sandbox_round_trip = round_trip
        recovered, detail = await runner._recover_timed_out_sandbox(
            "ws://sandbox/commands",
            "TransformAgent",
        )
        assert recovered is False
        assert "probe still wedged" in detail
    print("ok  a failed post-reset lane probe falls back to quarantine")


async def test_operator_release_clears_battery_local_quarantine() -> None:
    """A coordinator release must override the compatibility denylist without a lease loop."""
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-recovered", 51003)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        runner = _runner(
            url,
            [Prompt(id="p1", prompt="use the recovered sandbox")],
            workspace,
            concurrency=1,
        )
        runner.output_dir.mkdir(parents=True)
        runner._write_battery_manifest(1)
        acquired_client = None
        acquired_lease = None
        try:
            await sandbox.connect(url)
            async with CoordinatorClient(url) as operator:
                original = await asyncio.wait_for(operator.acquire(), timeout=2)
                runner._record_local_quarantine(
                    original, reason="protocol_fault", source="test"
                )
                await operator.quarantine(
                    original, reason="protocol_fault", source="test"
                )
                await operator.unquarantine(original.sandbox_alias)

            acquired_client, acquired_lease = await asyncio.wait_for(
                runner._acquire_sandbox(
                    0,
                    "p1",
                    1,
                    runner.output_dir / "p1" / "try01",
                    is_retry=False,
                ),
                timeout=3,
            )
            assert acquired_lease.sandbox_id == sandbox.sandbox_id
            assert sandbox.reset_count == 1, "local rejection triggered another reset"
            assert sandbox.sandbox_id not in runner._local_quarantines
            battery = json.loads((runner.output_dir / "battery.json").read_text())
            assert sandbox.sandbox_id not in battery["quarantined_sandboxes"]
        finally:
            if acquired_client is not None and acquired_lease is not None:
                await acquired_client.release(acquired_lease, outcome="done")
                await acquired_client.close()
            await sandbox.close()
            await coordinator.stop()
    print("ok  operator release clears the battery-local sandbox quarantine")


async def test_api_requeue_is_pending_until_redispatch() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001, auto_ready=False)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        run_task: asyncio.Task[dict] | None = None
        try:
            await sandbox.connect(url)
            os.environ["STUB_MODE"] = "api_retry_exhausted"
            runner = _runner(
                url,
                [Prompt(id="p1", prompt="wait for the endpoint")],
                workspace,
                concurrency=1,
                max_api_requeues=1,
                lease_acquire_timeout=0.05,
            )
            run_task = asyncio.create_task(runner.run())

            await asyncio.wait_for(sandbox.reset_requested.wait(), timeout=10)
            run_dir = workspace / "runs" / "p1" / "try01"
            pending = json.loads((run_dir / "attempt.json").read_text())
            assert pending["state"] == "requeued" and pending["outcome"] == "requeued"
            assert pending["pending_retry"] is True
            assert pending["requeue_reason"] == "api_retry_exhausted"
            assert pending["retry_queued_at"]
            assert pending["wall_seconds"] >= 0
            view = scan.scan_battery(workspace / "runs", time.time()).as_dict()
            assert view["attempts"][0]["pending_retry"] is True
            assert view["counts"]["pending_retry"] == 1
            assert view["counts"].get("requeued", 0) == 0
            elapsed = view["attempts"][0]["elapsed_seconds"]
            later = scan.scan_attempt(run_dir, workspace / "runs", time.time() + 600).as_dict()
            assert later["elapsed_seconds"] == elapsed
            assert later["remaining_seconds"] is None

            # The reset is deliberately held. The retry must time out, disconnect its parked
            # acquire, re-check fleet health, and keep trying instead of hanging in one RPC.
            for _ in range(100):
                pending = json.loads((run_dir / "attempt.json").read_text())
                if int(pending.get("retry_acquire_attempts") or 0) >= 2:
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("retry acquire did not time out and re-check the fleet")
            assert "ready=0" in pending["retry_wait_reason"]
            assert pending["retry_last_checked_at"]

            os.environ["STUB_MODE"] = "ok"
            await sandbox.report_ready()
            await asyncio.wait_for(run_task, timeout=20)

            archived = json.loads(
                (workspace / "runs" / "p1" / "try01.requeue00" / "attempt.json").read_text()
            )
            assert archived["pending_retry"] is False
            assert archived["state"] == "requeued" and archived["outcome"] == "requeued"
            replacement = json.loads((run_dir / "attempt.json").read_text())
            assert "pending_retry" not in replacement
            assert replacement["state"] == "finished"
        finally:
            os.environ.pop("STUB_MODE", None)
            if run_task is not None and not run_task.done():
                run_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await run_task
            await sandbox.close()
            await coordinator.stop()
    print("ok  API requeue stays pending until redispatch rotates the failed execution")


async def test_sandbox_loss_uses_the_pending_requeue_lifecycle() -> None:
    coordinator, url = await _start_coordinator()
    first = FakeSandbox("sandbox-a", 51001)
    replacement = FakeSandbox("sandbox-b", 51002)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        run_task: asyncio.Task[dict] | None = None
        first_closed = False
        try:
            await first.connect(url)
            os.environ["STUB_MODE"] = "hang"
            runner = _runner(
                url,
                [Prompt(id="p1", prompt="survive a lost sandbox")],
                workspace,
                concurrency=1,
            )
            run_task = asyncio.create_task(runner.run())
            manifest_path = workspace / "runs" / "p1" / "try01" / "attempt.json"
            for _ in range(100):
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    if manifest.get("state") == "running":
                        break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError("first sandbox attempt did not start")

            await first.close()
            first_closed = True
            for _ in range(100):
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("pending_retry"):
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError("sandbox loss was not published as a pending retry")
            assert manifest["requeue_reason"] == "sandbox_lost"
            assert manifest["state"] == "requeued" and manifest["outcome"] == "requeued"

            os.environ["STUB_MODE"] = "ok"
            await replacement.connect(url)
            await asyncio.wait_for(run_task, timeout=20)

            archived = json.loads(
                (workspace / "runs" / "p1" / "try01.requeue00" / "attempt.json").read_text()
            )
            assert archived["requeue_reason"] == "sandbox_lost"
            assert archived["pending_retry"] is False
        finally:
            os.environ.pop("STUB_MODE", None)
            if run_task is not None and not run_task.done():
                run_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await run_task
            if not first_closed:
                await first.close()
            await replacement.close()
            await coordinator.stop()
    print("ok  sandbox loss shares pending-retry bookkeeping and preserves its reason")


async def test_zero_api_requeues_records_first_exhaustion() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            await sandbox.connect(url)
            os.environ["STUB_MODE"] = "api_retry_exhausted"
            try:
                summary = await asyncio.wait_for(
                    _runner(
                        url,
                        [Prompt(id="p1", prompt="do not requeue")],
                        workspace,
                        concurrency=1,
                        api_max_attempts=4,
                        max_api_requeues=0,
                    ).run(),
                    timeout=60,
                )
            finally:
                os.environ.pop("STUB_MODE", None)

            attempt = summary["attempts"][0]
            assert attempt["outcome"] == "api_retry_exhausted", attempt
            assert attempt["requeues"] == 0 and attempt["api_requeues"] == 0
            assert attempt["api_max_attempts"] == 4
            assert attempt["max_api_requeues"] == 0
            assert sandbox.reset_count == 1
            manifest = json.loads(
                (workspace / "runs" / "p1" / "try01" / "attempt.json").read_text()
            )
            assert manifest["api_max_attempts"] == 4
            assert manifest["max_api_requeues"] == 0
        finally:
            await sandbox.close()
            await coordinator.stop()
    print("ok  zero API requeues records the first exhausted process")


async def test_capture_lifecycle_follows_the_agent_process() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    original = capture.record_previews
    called: list[str] = []
    stopped = asyncio.Event()

    async def fake_recorder(
        _run_dir: Path,
        commands_uri: str,
        _interval: float,
        *,
        stats: capture.CaptureStats,
        **_kwargs,
    ) -> capture.CaptureStats:
        called.append(commands_uri)
        stats.frames += 1
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return stats

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            await sandbox.connect(url)
            capture.record_previews = fake_recorder
            summary = await _runner(
                url,
                [Prompt(id="p1", prompt="record me")],
                workspace,
                capture_interval=0.05,
            ).run()
            assert summary["total_successes"] == 1
            assert stopped.is_set(), "recorder survived the agent process"
            assert called == ["ws://127.0.0.1:51001/commands"]
            manifest = json.loads(
                (workspace / "runs" / "p1" / "try01" / "attempt.json").read_text(encoding="utf-8")
            )
            assert manifest["capture_frames"] == 1
            assert manifest["capture_failures"] == 0
            assert sandbox.reset_count == 1
        finally:
            capture.record_previews = original
            await sandbox.close()
            await coordinator.stop()
    print("ok  supplementary capture stops before the sandbox is released")


async def test_cancelling_runner_kills_agent_and_releases_sandbox() -> None:
    """Ctrl+C cancels runner.run(); detached agent sessions must not survive that cancellation."""
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            await sandbox.connect(url)
            os.environ["STUB_MODE"] = "hang"
            runner_task = asyncio.create_task(
                _runner(url, [Prompt(id="p1", prompt="interrupt me")], workspace).run()
            )
            liveness_path = workspace / "runs" / "p1" / "try01" / "liveness.json"
            try:
                for _ in range(100):
                    if liveness_path.exists():
                        break
                    await asyncio.sleep(0.05)
                else:
                    raise AssertionError("agent did not start")

                agent_pid = json.loads(liveness_path.read_text(encoding="utf-8"))["pid"]
                os.kill(agent_pid, 0)  # It is alive before cancellation.
                runner_task.cancel()
                try:
                    await asyncio.wait_for(runner_task, timeout=30)
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError("runner cancellation did not propagate")
            finally:
                os.environ.pop("STUB_MODE", None)

            try:
                os.kill(agent_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError(f"agent process {agent_pid} survived runner cancellation")

            assert sandbox.reset_count == 1
            manifest = json.loads(
                (workspace / "runs" / "p1" / "try01" / "attempt.json").read_text(encoding="utf-8")
            )
            assert manifest["state"] == "finished", manifest
            assert manifest["outcome"] == "interrupted", manifest
        finally:
            await sandbox.close()
            await coordinator.stop()
    print("ok  cancelling the runner kills its detached agent and releases the sandbox")


async def test_human_success_stops_running_and_skips_queued_siblings() -> None:
    from sari_bench.watch.notify import Discord
    from sari_bench.watch.server import WatchState

    coordinator, url = await _start_coordinator()
    sandboxes = [FakeSandbox("sandbox-a", 51001), FakeSandbox("sandbox-b", 51002)]
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            for sandbox in sandboxes:
                await sandbox.connect(url)
            os.environ["STUB_MODE"] = "cancel_siblings"
            prompts = [
                Prompt(id="p", prompt="winner prompt", family="shared"),
                Prompt(id="next", prompt="next prompt", family="shared"),
            ]
            runner_task = asyncio.create_task(
                _runner(url, prompts, workspace, tries=4, concurrency=2).run()
            )
            winner_manifest = workspace / "runs" / "p" / "try01" / "attempt.json"
            running_manifest = workspace / "runs" / "p" / "try02" / "attempt.json"
            second_running_manifest = workspace / "runs" / "p" / "try03" / "attempt.json"
            for _ in range(300):
                winner = (
                    json.loads(winner_manifest.read_text())
                    if winner_manifest.exists()
                    else {}
                )
                running = (
                    json.loads(running_manifest.read_text())
                    if running_manifest.exists()
                    else {}
                )
                second_running = (
                    json.loads(second_running_manifest.read_text())
                    if second_running_manifest.exists()
                    else {}
                )
                if (
                    winner.get("state") == "finished"
                    and running.get("pid")
                    and second_running.get("pid")
                ):
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("winner did not finish while a sibling was running")

            state = WatchState(
                bench_root=workspace,
                fixed_battery=workspace / "runs",
                discord=Discord(enabled=False),
                min_interval=0.0,
            )
            response = state.verdict("p/try01", "pass", by="tester")
            assert response["ok"] is True, response
            assert response["siblings_stopped"] == 2, response

            summary = await asyncio.wait_for(runner_task, timeout=30)
        finally:
            os.environ.pop("STUB_MODE", None)
            for sandbox in sandboxes:
                await sandbox.close()
            await coordinator.stop()

        assert summary["total_attempts"] == 8, summary
        rows = {(row["prompt_id"], row["attempt"]): row for row in summary["attempts"]}
        assert rows[("p", 1)]["outcome"] == "completed"
        assert rows[("p", 2)]["outcome"] == "operator_kill", rows[("p", 2)]
        assert rows[("p", 2)]["end_reason"] == "already_successful"
        assert rows[("p", 3)]["outcome"] == "operator_kill", rows[("p", 3)]
        assert rows[("p", 3)]["end_reason"] == "already_successful"
        assert rows[("p", 4)]["outcome"] == "skipped", rows[("p", 4)]
        assert rows[("p", 4)]["end_reason"] == "already_successful"
        assert rows[("p", 4)]["wall_seconds"] == 0.0
        assert rows[("p", 4)]["winning_attempt_key"] == "p/try01"
        assert all(rows[("next", n)]["outcome"] == "completed" for n in (1, 2, 3, 4))

        skipped_manifest = json.loads(
            (workspace / "runs" / "p" / "try04" / "attempt.json").read_text()
        )
        assert skipped_manifest["state"] == "finished"
        assert skipped_manifest["outcome"] == "skipped"
        assert skipped_manifest["winning_attempt_key"] == "p/try01"
        lines = (workspace / "runs" / "attempts.jsonl").read_text().strip().splitlines()
        assert len(lines) == 8, "planned-attempt denominator was not preserved in the result spine"
        from sari_bench.report import collect
        report_rows = {
            (row["prompt_id"], row["attempt"]): row
            for row in collect(workspace / "runs")[0]
        }
        assert report_rows[("p", 2)]["end_reason"] == "already_successful"
        assert report_rows[("p", 4)]["outcome"] == "skipped"
        assert report_rows[("p", 4)]["winning_attempt_key"] == "p/try01"
    print("ok  verified success kills a running sibling, skips queued work, and advances the battery")


async def test_runner_closes_lease_and_pid_publication_races() -> None:
    """Force the winner to appear at each checkpoint around subprocess publication."""

    class RacingRunner(BenchmarkRunner):
        winner_on_call: int
        winner_calls: int

        def _human_verified_winner(self, _prompt_id: str) -> dict | None:
            self.winner_calls += 1
            if self.winner_calls < self.winner_on_call:
                return None
            return {
                "winning_attempt_key": "p/try00",
                "stop_requested_at": "2026-07-27T12:00:00",
                "stop_requested_by": "tester",
            }

    for winner_on_call, expected_outcome in ((2, "skipped"), (3, "operator_kill")):
        coordinator, url = await _start_coordinator()
        sandbox = FakeSandbox(f"sandbox-{winner_on_call}", 51000 + winner_on_call)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            try:
                await sandbox.connect(url)
                base = _runner(
                    url,
                    [Prompt(id="p", prompt="race prompt")],
                    workspace,
                    concurrency=1,
                )
                runner = RacingRunner(
                    prompts=list(base.prompts.values()),
                    coordinator_url=base.coordinator_url,
                    output_dir=base.output_dir,
                    tries=base.tries,
                    time_limit_minutes=base.time_limit_minutes,
                    concurrency=base.concurrency,
                    max_steps=base.max_steps,
                    arm=base.arm,
                    map_dir=base.map_dir,
                    leg_retries=base.leg_retries,
                    timeout_grace=base.timeout_grace,
                    capture_interval=0.0,
                    agent_entry=base.agent_entry,
                    agent_cwd=base.agent_cwd,
                    ocr_url=base.ocr_url,
                    ocr_health_check=base.ocr_health_check,
                )
                runner.winner_on_call = winner_on_call
                runner.winner_calls = 0
                summary = await runner.run()
            finally:
                await sandbox.close()
                await coordinator.stop()

            row = summary["attempts"][0]
            assert row["outcome"] == expected_outcome, row
            assert row["end_reason"] == "already_successful", row
            assert row["winning_attempt_key"] == "p/try00", row
            assert sandbox.reset_count == 1
            manifest = json.loads(
                (workspace / "runs" / "p" / "try01" / "attempt.json").read_text()
            )
            assert bool(manifest.get("pid")) is (winner_on_call == 3), manifest
    print("ok  runner observes winners after lease acquisition and immediately after PID publication")


async def test_token_usage_is_recorded_per_attempt() -> None:
    """Tokens in/out are the point of this test twice over: from the agent's summary.json when it
    exits cleanly, and from its periodic tokens.json when the harness kills it first."""
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            await sandbox.connect(url)
            summary = await asyncio.wait_for(
                _runner(url, [Prompt(id="p1", prompt="task one")], workspace).run(), timeout=60
            )

            attempt = summary["attempts"][0]
            # summary.json wins over tokens.json: it is the agent's final word.
            assert (attempt["tokens_in"], attempt["tokens_out"]) == (2000, 500), attempt
            assert attempt["llm_calls"] == 6, attempt
            assert attempt["api_calls"] == 8, attempt
            assert summary["tokens_in"] == 2000 and summary["tokens_out"] == 500, summary
            assert summary["tokens_total"] == 2500, summary
            assert summary["api_calls"] == 8, summary
            assert summary["api_calls_coverage"] == {"known": 1, "total": 1}, summary
            row = summary["prompts"][0]
            assert (row["tokens_in"], row["tokens_out"]) == (2000, 500), row
            assert (row["tokens_in_avg"], row["tokens_out_avg"]) == (2000, 500), row
            assert row["api_calls"] == 8, row

            # Per-reasoner attribution follows the same authority chain as the totals, and the role
            # rows must re-total to them - a breakdown that does not add up to the number beside it
            # is worse than no breakdown.
            assert attempt["tokens_by_role"] == {
                "actor": {"tokens_in": 1400, "tokens_out": 350, "calls": 4,
                          "api_calls": 5},
                "guard": {"tokens_in": 600, "tokens_out": 150, "calls": 2,
                          "api_calls": 3},
            }, attempt
            assert summary["tokens_by_role"]["actor"]["tokens_in"] == 1400, summary
            assert summary["tokens_by_role"]["guard"]["api_calls"] == 3, summary
            assert summary["tokens_by_role"]["guard"]["api_calls_coverage"] == {
                "known": 1, "total": 1}, summary
            assert sum(r["tokens_in"] for r in summary["tokens_by_role"].values()) == 2000, summary
            # Pipeline order, so the battery summary reads as the agent's own pipeline.
            assert list(summary["tokens_by_role"]) == ["actor", "guard"], summary

            # The finished manifest carries them too, which is what the watcher reads.
            manifest = json.loads(
                (workspace / "runs" / "p1" / "try01" / "attempt.json").read_text(encoding="utf-8"))
            assert (manifest["tokens_in"], manifest["tokens_out"]) == (2000, 500), manifest
            assert manifest["api_calls"] == 8, manifest
            assert manifest["tokens_by_role"]["guard"]["calls"] == 2, manifest
        finally:
            await sandbox.close()
            await coordinator.stop()

    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-b", 51002)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            await sandbox.connect(url)
            os.environ["STUB_MODE"] = "hang"
            try:
                runner = _runner(
                    url,
                    [Prompt(id="p1", prompt="never finishes")],
                    workspace,
                    time_limit_minutes=1.0 / 60.0,
                    timeout_grace=0.5,
                )
                summary = await asyncio.wait_for(runner.run(), timeout=60)
            finally:
                os.environ.pop("STUB_MODE", None)

            attempt = summary["attempts"][0]
            assert attempt["outcome"] == "harness_timeout", attempt
            # No summary.json was ever written; the tokens still have to be accounted for - and so
            # does which reasoner spent them, which is the case an ablation most wants: the attempts
            # that ran to the wall are the expensive ones.
            assert (attempt["tokens_in"], attempt["tokens_out"]) == (1200, 300), attempt
            assert attempt["api_calls"] == 5, attempt
            assert attempt["tokens_by_role"] == {
                "actor": {"tokens_in": 800, "tokens_out": 200, "calls": 2,
                          "api_calls": 3},
                "guard": {"tokens_in": 400, "tokens_out": 100, "calls": 2,
                          "api_calls": 2},
            }, attempt
        finally:
            await sandbox.close()
            await coordinator.stop()
    print("ok  token usage is recorded per attempt, killed attempts included")


async def test_watcher_retry_replaces_a_finished_try_after_runner_exit() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            await sandbox.connect(url)
            runner = _runner(url, [Prompt(id="p1", prompt="retry me")], workspace)
            await asyncio.wait_for(runner.run(), timeout=60)

            battery = workspace / "runs"
            run_dir = battery / "p1" / "try01"
            old_manifest = json.loads((run_dir / "attempt.json").read_text())
            archive = battery / "p1" / "try01.requeue00"
            archive.mkdir()
            (archive / "agent.log").write_text("old sandbox-loss log")

            state = WatchState(
                bench_root=workspace,
                fixed_battery=battery,
                discord=Discord(enabled=False),
                min_interval=0.0,
                coordinator_url=url,
                retry_agent_entry=str(workspace / "stub_agent.py"),
                retry_agent_cwd=workspace,
                retry_ocr_health_check=runner.ocr_health_check,
            )
            assert state.verdict("p1/try01", "pass", by="tester")["ok"] is True
            accepted = state.retry("p1/try01")
            assert accepted["ok"] is True, accepted
            assert state.retry("p1/try01")["ok"] is False, "duplicate retry was accepted"

            for _ in range(1000):
                with state._lock:
                    pending = bool(state._retry_jobs)
                if not pending:
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("watcher retry did not finish")

            replacement = json.loads((run_dir / "attempt.json").read_text())
            assert replacement["run_id"] != old_manifest["run_id"]
            assert replacement["state"] == "finished"
            assert archive.exists(), "existing sandbox-loss history was discarded"
            preserved = battery / "p1" / "try01.requeue01" / "attempt.json"
            assert json.loads(preserved.read_text())["run_id"] == old_manifest["run_id"]
            rows = [
                json.loads(line)
                for line in (battery / "attempts.jsonl").read_text().splitlines()
                if line.strip()
            ]
            assert len(rows) == 1 and rows[0]["prompt_id"] == "p1" and rows[0]["attempt"] == 1
            summary = json.loads((battery / "summary.json").read_text())
            assert summary["total_attempts"] == 1, summary
            plan = json.loads((battery / "battery.json").read_text())
            assert "p1" not in (plan.get("human_verified_winners") or {})
            assert "verified_success" not in replacement
            assert sandbox.reset_count == 2, "replacement did not lease and reset a fresh sandbox"
        finally:
            await sandbox.close()
            await coordinator.stop()
    print("ok  watcher retry archives history and replaces a finished try after runner exit")


async def test_watcher_retry_stops_and_replaces_a_live_try() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        runner_task: asyncio.Task | None = None
        try:
            await sandbox.connect(url)
            os.environ["STUB_MODE"] = "hang"
            runner_task = asyncio.create_task(
                _runner(url, [Prompt(id="p1", prompt="interrupt me")], workspace).run()
            )
            manifest_path = workspace / "runs" / "p1" / "try01" / "attempt.json"
            for _ in range(500):
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    if manifest.get("state") == "running":
                        break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("original attempt never became live")
            old_run_id = manifest["run_id"]
            os.environ.pop("STUB_MODE", None)

            state = WatchState(
                bench_root=workspace,
                fixed_battery=workspace / "runs",
                discord=Discord(enabled=False),
                min_interval=0.0,
                coordinator_url=url,
                retry_agent_entry=str(workspace / "stub_agent.py"),
                retry_agent_cwd=workspace,
                retry_ocr_health_check=lambda _url: {
                    "ready": True, "api_version": "v1", "model": "fake-ocr"
                },
            )
            assert state.retry("p1/try01")["ok"] is True
            await asyncio.wait_for(runner_task, timeout=60)
            for _ in range(1000):
                with state._lock:
                    pending = bool(state._retry_jobs)
                if not pending:
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("live retry did not finish")

            manifest = json.loads(manifest_path.read_text())
            assert manifest["run_id"] != old_run_id
            assert manifest["outcome"] == "completed"
            rows = [
                json.loads(line)
                for line in (workspace / "runs" / "attempts.jsonl").read_text().splitlines()
            ]
            assert len(rows) == 1 and rows[0]["outcome"] == "completed", rows
            assert sandbox.reset_count == 2
        finally:
            os.environ.pop("STUB_MODE", None)
            if runner_task is not None and not runner_task.done():
                runner_task.cancel()
                await asyncio.gather(runner_task, return_exceptions=True)
            await sandbox.close()
            await coordinator.stop()
    print("ok  watcher retry stops, cleans, and replaces a live try")


async def test_nonempty_output_requires_explicit_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        output = workspace / "runs"
        output.mkdir()
        (output / "unrelated.txt").write_text("do not overwrite", encoding="utf-8")
        runner = _runner(
            "ws://127.0.0.1:1",
            [Prompt(id="p", prompt="never start")],
            workspace,
            sandbox_startup_timeout=0.01,
        )
        try:
            await runner.run()
        except ResumeError as error:
            assert "--resume" in str(error), error
        else:
            raise AssertionError("non-empty output directory was reused without --resume")
        assert (output / "unrelated.txt").read_text() == "do not overwrite"
        assert not (output / "battery.json").exists()
    print("ok  a non-empty output directory requires explicit resume")


async def test_completed_resume_repairs_results_without_a_sandbox() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        prompts = [Prompt(id="p", prompt="resume me")]
        try:
            await sandbox.connect(url)
            await _runner(url, prompts, workspace, tries=2, concurrency=1).run()
            original_resets = sandbox.reset_count
        finally:
            await sandbox.close()
            await coordinator.stop()

        battery_dir = workspace / "runs"
        attempts_path = battery_dir / "attempts.jsonl"
        rows = [json.loads(line) for line in attempts_path.read_text().splitlines()]
        duplicate = dict(rows[0])
        duplicate["error"] = "latest duplicate retained"
        # Simulate both legacy duplication and a crash after closing a manifest but before
        # publishing its aggregate result row.
        attempts_path.write_text(
            json.dumps(rows[0]) + "\n" + json.dumps(duplicate) + "\n",
            encoding="utf-8",
        )
        try01_manifest_path = battery_dir / "p" / "try01" / "attempt.json"
        stale_manifest = json.loads(try01_manifest_path.read_text())
        stale_manifest["state"] = "running"
        try01_manifest_path.write_text(json.dumps(stale_manifest), encoding="utf-8")
        battery = json.loads((battery_dir / "battery.json").read_text())
        battery["human_verified_winners"] = {
            "p": {
                "winning_attempt_key": "p/try01",
                "stop_requested_at": "2026-07-27T12:00:00",
                "stop_requested_by": "tester",
            }
        }
        (battery_dir / "battery.json").write_text(json.dumps(battery), encoding="utf-8")
        try01_start = json.loads(
            (battery_dir / "p" / "try01" / "liveness.json").read_text()
        )["start"]

        # Operational settings may change, and a fully complete resume must not need a coordinator.
        resumed = _runner(
            "ws://127.0.0.1:1",
            prompts,
            workspace,
            tries=2,
            concurrency=None,
            resume=True,
        )
        summary = await resumed.run()
        repaired = [json.loads(line) for line in attempts_path.read_text().splitlines()]
        assert len(repaired) == 2, repaired
        assert repaired[0]["error"] == "latest duplicate retained", repaired
        assert summary["total_attempts"] == 2
        assert json.loads(
            (battery_dir / "p" / "try01" / "liveness.json").read_text()
        )["start"] == try01_start
        assert json.loads(try01_manifest_path.read_text())["state"] == "finished"
        durable = json.loads((battery_dir / "battery.json").read_text())
        assert durable["human_verified_winners"]["p"]["winning_attempt_key"] == "p/try01"
        assert durable["resume_count"] == 1
        assert original_resets == 2

        incompatible = _runner(
            "ws://127.0.0.1:1",
            prompts,
            workspace,
            tries=3,
            resume=True,
        )
        try:
            await incompatible.run()
        except ResumeError as error:
            assert "tries" in str(error), error
        else:
            raise AssertionError("semantic resume mismatch was accepted")
    print("ok  completed resume compacts/backfills results, preserves winners, and needs no sandbox")


async def test_resume_deletes_and_reruns_only_an_interrupted_attempt() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        prompts = [Prompt(id="p", prompt="partial battery")]
        try:
            await sandbox.connect(url)
            await _runner(url, prompts, workspace, tries=2, concurrency=1).run()
            battery = workspace / "runs"
            first_start = json.loads(
                (battery / "p" / "try01" / "liveness.json").read_text()
            )["start"]
            interrupted_dir = battery / "p" / "try02"
            interrupted = json.loads((interrupted_dir / "attempt.json").read_text())
            interrupted.update(
                {
                    "state": "running",
                    "outcome": "",
                    "pid": 999_999_999,
                    "lease_id": "already-reaped-test-lease",
                }
            )
            (interrupted_dir / "attempt.json").write_text(json.dumps(interrupted))
            (interrupted_dir / "partial-only.txt").write_text("discard me")
            rows = [
                json.loads(line)
                for line in (battery / "attempts.jsonl").read_text().splitlines()
            ]
            (battery / "attempts.jsonl").write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in rows
                    if not (row["prompt_id"] == "p" and row["attempt"] == 2)
                )
            )

            resumed = _runner(
                url,
                prompts,
                workspace,
                tries=2,
                concurrency=1,
                resume=True,
            )
            summary = await resumed.run()
            assert summary["total_attempts"] == 2
            assert sandbox.reset_count == 3, "a finished attempt was rerun"
            assert json.loads(
                (battery / "p" / "try01" / "liveness.json").read_text()
            )["start"] == first_start
            assert not (interrupted_dir / "partial-only.txt").exists()
            assert not (battery / "p" / "try02.requeue00").exists()
            result_rows = [
                json.loads(line)
                for line in (battery / "attempts.jsonl").read_text().splitlines()
            ]
            assert len(result_rows) == 2
        finally:
            await sandbox.close()
            await coordinator.stop()
    print("ok  resume deletes and cleanly reruns only the interrupted logical attempt")


async def test_resume_stops_an_orphan_before_pid_publication() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    orphan: asyncio.subprocess.Process | None = None
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        prompts = [Prompt(id="p", prompt="orphaned attempt")]
        try:
            await sandbox.connect(url)
            base = _runner(url, prompts, workspace, concurrency=1)
            base.output_dir.mkdir(parents=True)
            base._write_battery_manifest(1)
            run_dir = base.output_dir / "p" / "try01"
            run_dir.mkdir(parents=True)
            env = dict(os.environ)
            env["STUB_MODE"] = "hang"
            orphan = await asyncio.create_subprocess_exec(
                sys.executable,
                base.agent_entry,
                "--task",
                "orphaned attempt",
                "--run-dir",
                str(run_dir),
                env=env,
                start_new_session=True,
            )
            for _ in range(100):
                if (run_dir / "liveness.json").exists():
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("orphan fixture did not start")

            # Model SIGKILL landing between create_subprocess_exec() and publication of its PID.
            (run_dir / "attempt.json").write_text(
                json.dumps(
                    {
                        "run_id": "crash-window",
                        "prompt_id": "p",
                        "prompt": "orphaned attempt",
                        "attempt": 1,
                        "state": "starting",
                        "pid": None,
                        "runner_host": socket.gethostname(),
                        "runner_boot_id": base._boot_id(),
                        "run_dir": str(run_dir),
                        "command": [
                            sys.executable,
                            base.agent_entry,
                            "--task",
                            "orphaned attempt",
                            "--run-dir",
                            str(run_dir),
                        ],
                    }
                ),
                encoding="utf-8",
            )

            resumed = _runner(url, prompts, workspace, concurrency=1, resume=True)
            summary = await resumed.run()
            await asyncio.wait_for(orphan.wait(), timeout=5)
            assert orphan.returncode != 0, "the old agent was not terminated"
            assert summary["total_attempts"] == 1
            replacement = json.loads((run_dir / "attempt.json").read_text())
            assert replacement["run_id"] != "crash-window"
            assert replacement["state"] == "finished"
        finally:
            if orphan is not None and orphan.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(orphan.pid), signal.SIGKILL)
                await orphan.wait()
            await sandbox.close()
            await coordinator.stop()
    print("ok  resume finds and stops an orphan from the pre-PID publication crash window")


async def main() -> int:
    test_load_prompts_accepts_the_battery_schema()
    test_automatic_retries_outrank_fresh_work_fifo()
    test_completion_guard_is_threaded_into_agent_command_and_battery_config()
    test_refusal_cap_action_is_threaded_and_resume_checked()
    test_adaptive_replanning_is_threaded_and_resume_checked()
    for test in (
        test_ocr_preflight_fails_before_coordinator_or_lease,
        test_battery_runs_every_prompt_and_attempt,
        test_pool_size_bounds_concurrency,
        test_automatic_concurrency_fills_and_grows_with_the_fleet,
        test_explicit_concurrency_remains_a_hard_cap,
        test_startup_timeout_explains_an_empty_pool,
        test_overrunning_attempt_is_killed_and_recorded,
        test_crashed_agent_still_releases_its_sandbox,
        test_api_retry_exhaustion_requeues_the_logical_attempt,
        test_sandbox_protocol_fault_quarantines_and_requeues,
        test_command_timeout_recovery_resets_and_probes_the_failed_lane,
        test_command_timeout_recovery_rejects_a_failed_probe,
        test_operator_release_clears_battery_local_quarantine,
        test_api_requeue_is_pending_until_redispatch,
        test_sandbox_loss_uses_the_pending_requeue_lifecycle,
        test_zero_api_requeues_records_first_exhaustion,
        test_capture_lifecycle_follows_the_agent_process,
        test_cancelling_runner_kills_agent_and_releases_sandbox,
        test_human_success_stops_running_and_skips_queued_siblings,
        test_runner_closes_lease_and_pid_publication_races,
        test_token_usage_is_recorded_per_attempt,
        test_watcher_retry_replaces_a_finished_try_after_runner_exit,
        test_watcher_retry_stops_and_replaces_a_live_try,
        test_nonempty_output_requires_explicit_resume,
        test_completed_resume_repairs_results_without_a_sandbox,
        test_resume_deletes_and_reruns_only_an_interrupted_attempt,
        test_resume_stops_an_orphan_before_pid_publication,
    ):
        await test()
    print("\nAll runner tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
