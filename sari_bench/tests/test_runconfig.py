from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import tomli

from sari_bench.runner import BenchmarkRunner, async_main
from sari_runconfig import RunConfigError, load_run_config


def test_context_ablation_configs_are_five_try_and_runner_is_complete() -> None:
    root = Path(__file__).resolve().parents[2]
    config_dir = root / "configs" / "context-ablation"
    names = [
        "baseline", "a1", "a2c", "a3", "a4", "a5", "a6-2", "a6-4",
        "a7-no-stop-guard", "hard-baseline",
    ]
    configs = {}
    for name in names:
        with (config_dir / f"{name}.toml").open("rb") as handle:
            configs[name] = tomli.load(handle)["bench"]
        assert configs[name]["tries"] == 5, name

    a7 = dict(configs["a7-no-stop-guard"])
    baseline = dict(configs["baseline"])
    assert a7["context_policy"] == "baseline-2img"
    assert a7["completion_guard"] == "none"
    assert baseline["completion_guard"] == "vlm"
    baseline["context_policy"] = "baseline-2img"
    a7.pop("name")
    baseline.pop("name")
    a7.pop("completion_guard")
    baseline.pop("completion_guard")
    assert a7 == baseline, "A7 must differ from the fresh baseline only in name and guard"

    from agent.agent_core.context_policy import CONTEXT_POLICIES
    assert "a7-no-stop-guard" not in CONTEXT_POLICIES
    expected_policies = {
        "baseline": "baseline",
        "a1": "a1-2img",
        "a2c": "a2c-2img",
        "a3": "a3-2img",
        "a4": "a4-2img",
        "a5": "a5-2img",
        "a6-2": "a6-2",
        "a6-4": "a6-4",
        "a7-no-stop-guard": "baseline-2img",
        "hard-baseline": "baseline-2img",
    }
    assert {name: config["context_policy"] for name, config in configs.items()} == expected_policies

    runner = config_dir / "run_all.sh"
    text = runner.read_text(encoding="utf-8")
    match = re.search(r"policies=\(([^)]*)\)", text)
    assert match is not None
    scheduled = match.group(1).split()
    assert scheduled == names
    assert len(scheduled) == len(set(scheduled))
    assert os.access(runner, os.X_OK), "the ablation runner lost its executable bit"


def test_comprehensive_context_ablation_runner_covers_every_distinct_arm() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "configs" / "context-ablation" / "run_all_comprehensive.sh"
    text = runner.read_text(encoding="utf-8")

    match = re.search(r"policies=\(([^)]*)\)", text)
    assert match is not None
    assert match.group(1).split() == [
        "baseline", "a1", "a2c", "a3", "a4", "a5", "a6-2", "a6-4",
        "a7-no-stop-guard",
    ]
    for battery in ("easy_prompts.json", "medium_prompts.json", "hard_prompts.json"):
        assert battery in text
    assert "--prompts \"$prompts\"" in text
    assert "--tries 3" in text
    assert os.access(runner, os.X_OK), "the comprehensive runner lost its executable bit"


def test_loader_resolves_paths_from_the_config_and_rejects_typos(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "run.toml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
[agent]
navigation_strategy = "graph-advised"
completion_guard = "none"

[api_retry]
max_attempts = 6

[environment]
map_dir = "../maps/frozen"

[bench]
prompts = "../prompts/battery.json"
tries = 2
max_api_requeues = 0

[experimental]
adaptive_leg_replanning = true
""",
        encoding="utf-8",
    )

    config = load_run_config(config_path)
    assert config.get("agent", "navigation_strategy") == "graph-advised"
    assert config.get("agent", "completion_guard") == "none"
    assert config.get("api_retry", "max_attempts") == 6
    assert config.get("bench", "max_api_requeues") == 0
    assert config.get("experimental", "adaptive_leg_replanning", False) is True
    assert config.get("environment", "map_dir") == str(tmp_path / "maps" / "frozen")
    assert config.get("bench", "prompts") == str(tmp_path / "prompts" / "battery.json")

    config_path.write_text("[agent]\ncompletion_gard = \"vlm\"\n", encoding="utf-8")
    with pytest.raises(RunConfigError, match=r"agent\.completion_gard"):
        load_run_config(config_path)


def test_deprecated_resolver_backend_alias_loads_and_warns(tmp_path: Path, capsys) -> None:
    """The pre-2026-08-05 spelling still runs: it is rewritten to the current value, so nothing
    downstream ever sees 'qwen', and the deprecation is announced rather than silent."""
    config_path = tmp_path / "run.toml"
    config_path.write_text('[agent]\nresolver_backend = "qwen"\n', encoding="utf-8")

    config = load_run_config(config_path)

    assert config.get("agent", "resolver_backend") == "endpoint"
    warning = capsys.readouterr().err
    assert "DEPRECATED" in warning and "'endpoint'" in warning


def test_deprecated_agent_arm_key_loads_and_warns(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text('[agent]\narm = "graph-advised"\n', encoding="utf-8")

    config = load_run_config(config_path)

    assert config.get("agent", "navigation_strategy") == "graph-advised"
    assert config.get("agent", "arm") is None
    warning = capsys.readouterr().err
    assert "DEPRECATED" in warning and "agent.navigation_strategy" in warning


@pytest.mark.parametrize(
    "body, message",
    [
        ('[agent]\nnavigation_strategy = "magic"\n',
         "agent.navigation_strategy must be one of"),
        ('[agent]\nresolver_backend = "gpt"\n', "agent.resolver_backend must be one of"),
        ("[limits]\nmax_steps = true\n", "limits.max_steps must be an integer"),
        ("[bench]\ntries = 0\n", "bench.tries must be at least 1"),
        ("[api_retry]\nmax_attempts = 0\n", "api_retry.max_attempts must be at least 1"),
        ("[bench]\nmax_api_requeues = -1\n", "bench.max_api_requeues cannot be negative"),
        ("[bench]\nlease_acquire_timeout = 0\n",
         "bench.lease_acquire_timeout must be a positive finite number"),
        ("[bench]\nsandbox_command_timeout = 0\n",
         "bench.sandbox_command_timeout must be a positive finite number"),
        ("[bench]\nsandbox_command_timeout = inf\n",
         "bench.sandbox_command_timeout must be a positive finite number"),
        ("[environment]\nsandbox_command_timeout = nan\n",
         "environment.sandbox_command_timeout must be a positive finite number"),
        ("[experimental]\nadaptive_leg_replanning = 1\n",
         "experimental.adaptive_leg_replanning must be a bool"),
        ("[mystery]\nvalue = 1\n", r"unknown section\(s\)"),
    ],
)
def test_loader_rejects_invalid_values(tmp_path: Path, body: str, message: str) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(body, encoding="utf-8")
    with pytest.raises(RunConfigError, match=message):
        load_run_config(config_path)


def test_bench_uses_config_and_explicit_cli_flags_win(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{"id": "p1", "prompt": "Pick it up"}]), encoding="utf-8")
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[bench]
prompts = "prompts.json"
tries = 2
time_limit = 90.0
per_leg_minutes = 12.0
max_steps = 77
arm = "vlm"
context_policy = "a5"
completion_guard = "vlm"
name = "from-config"
max_api_requeues = 1
lease_acquire_timeout = 11.0
sandbox_command_timeout = 18.0

[experimental]
adaptive_leg_replanning = true

[api_retry]
max_attempts = 7
""",
        encoding="utf-8",
    )

    seen: dict[str, object] = {}

    async def fake_run(runner: BenchmarkRunner) -> dict:
        seen.update(
            tries=runner.tries,
            time_limit=runner.time_limit_minutes,
            per_leg_minutes=runner.per_leg_minutes,
            max_steps=runner.max_steps,
            arm=runner.arm,
            context_policy=runner.context_policy,
            completion_guard=runner.completion_guard,
            api_max_attempts=runner.api_max_attempts,
            max_api_requeues=runner.max_api_requeues,
            lease_acquire_timeout=runner.lease_acquire_timeout,
            sandbox_command_timeout=runner.sandbox_command_timeout,
            adaptive_leg_replanning=runner.adaptive_leg_replanning,
        )
        return {}

    with patch.object(BenchmarkRunner, "run", fake_run):
        result = asyncio.run(
            async_main(
                [
                    "--config",
                    str(config_path),
                    "--tries",
                    "4",
                    "--arm",
                    "graph",
                    "--context-policy",
                    "a6-2",
                    "--api-max-attempts",
                    "5",
                    "--max-api-requeues",
                    "2",
                    "--lease-acquire-timeout",
                    "4.5",
                    "--sandbox-command-timeout",
                    "7.5",
                    "--no-adaptive-leg-replanning",
                ]
            )
        )

    assert result == 0
    assert seen == {
        "tries": 4,
        "time_limit": 90.0,
        "per_leg_minutes": 12.0,
        "max_steps": 77,
        "arm": "graph",
        "context_policy": "a6-2",
        "completion_guard": "vlm",
        "api_max_attempts": 5,
        "max_api_requeues": 2,
        "lease_acquire_timeout": 4.5,
        "sandbox_command_timeout": 7.5,
        "adaptive_leg_replanning": False,
    }


def test_standalone_agent_uses_config_and_explicit_cli_flags_win(tmp_path: Path) -> None:
    from agent.orchestrator import subtask_agents
    from orchestrator import cli as orchestrator_cli

    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[agent]
navigation_strategy = "vlm"
context_policy = "a5"
resolver_backend = "claude-cli"
completion_guard = "none"
leg_retries = 3

[api_retry]
max_attempts = 7

[bench]
max_api_requeues = 2

[limits]
max_steps = 21
max_minutes = 6.5

[environment]
map_dir = "map"
reset_start = true
sandbox_command_timeout = 19.0

[output]
run_dir = "runs/one"
summary = "runs/one/result.json"

[experimental]
adaptive_leg_replanning = true
""",
        encoding="utf-8",
    )

    seen: dict[str, object] = {}

    def fake_orchestrate(config):
        seen.update(vars(config))
        seen["sandbox_command_timeout"] = os.environ.get("SARI_SANDBOX_COMMAND_TIMEOUT")

    with (
        patch.object(orchestrator_cli, "orchestrate", fake_orchestrate),
        patch.dict(os.environ, {}, clear=False),
    ):
        subtask_agents.main(
            [
                "--config",
                str(config_path),
                "--task",
                "configured run",
                "--max-steps",
                "42",
                "--context-policy",
                "a6-4",
                "--no-reset-start",
                "--api-max-attempts",
                "4",
                "--adaptive-leg-replanning",
            ]
        )

    assert seen["task"] == "configured run"
    assert seen["arm"] == "vlm"
    assert seen["context_policy"] == "a6-4"
    assert seen["caps"] == (42, 6.5)
    assert seen["resolver_backend"] == "claude-cli"
    assert seen["completion_guard"] == "none"
    assert seen["leg_retries"] == 3
    assert seen["api_max_attempts"] == 4
    assert seen["max_api_requeues"] == 2
    assert seen["output_dir"] == str(tmp_path / "map")
    assert seen["run_dir"] == str(tmp_path / "runs" / "one")
    assert seen["out"] == str(tmp_path / "runs" / "one" / "result.json")
    assert seen["reset_start"] is False
    assert seen["sandbox_command_timeout"] == "19.0"
    assert seen["adaptive_leg_replanning"] is True
