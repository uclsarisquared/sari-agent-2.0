# Distributed Sari Bench

Run a prompt battery across a fleet of Sari Sandbox instances. Each attempt leases one sandbox,
runs the current agent, records its artifacts, and resets the sandbox before returning it to the
pool.

## Quick start

Configure `[bench]` in the root `runconfig.toml`, then start these processes in separate terminals:

```bash
# 1. Shared OCR service (auto-selects DirectML on the configured Windows/WSL host)
uv run poe ocr-server

# 2. Coordinator reachable by the runner and every sandbox host
uv run poe dbench-coordinator

# 3. One or more sandbox players (repeat on every Unity host)
SARI_BENCH_COORDINATOR=ws://<coordinator-host>:9000/sandbox ./SariSandbox

# 4. Battery runner
uv run poe dbench-run

# 5. Optional live dashboard at http://127.0.0.1:8900
uv run python -m sari_bench watch
```

Build the player from Unity with **Build > Distributed Sari Bench Player**. For editor testing, set
`SARI_BENCH_COORDINATOR` before starting Play mode. The sim must have
`sariSandboxV1CompatibilityLayer` enabled.

Check registered sandboxes with:

```bash
python -m sari_bench status --coordinator ws://<coordinator-host>:9000
python -m sari_bench status --json
python -m sari_bench quarantine sandbox-03 --reason lidar_protocol_nonbinary
python -m sari_bench unquarantine sandbox-03
```

Canonical UUIDs remain in JSON for joins and diagnostics, while the CLI and dashboard display the
coordinator's persistent `sandbox-NN` alias. Lease aliases use `<prompt-id>/tryNN`. Quarantines and
sandbox aliases survive coordinator restarts; unquarantine always resets the player before it can
be leased again. The watch server exposes the same backend at `GET /api/fleet/status`, with
`POST /api/fleet/quarantine` and `POST /api/fleet/unquarantine` accepting
`{"sandbox":"sandbox-03"}` (and an optional quarantine `reason`).

The runner fills all ready sandboxes by default. Set `concurrency` in `runconfig.toml` to cap it.
Command-line flags override the config file.

## Results

Runs are written to `bench_runs/<timestamp>[_<name>]/`. The main files are:

- `summary.json` — battery and per-prompt results
- `attempts.jsonl` — one row per attempt
- `<prompt_id>/try<NN>/` — logs, JPEG step evidence, tokens, response, and attempt details
- `<prompt_id>/try<NN>/capture/latest.jpg` — atomically refreshed live preview
- `<prompt_id>/try<NN>/replay.mp4` and `replay.vtt` — continuous replay and selectable captions

Useful follow-up commands:

```bash
python -m sari_bench report                       # export CSV summaries
python -m sari_bench video --battery <run-dir>    # render attempt replays
python -m sari_bench watch --run-dir <run-dir>    # inspect a specific run
python -m sari_bench cleanup-captures --bench-root bench_runs  # dry-run legacy cleanup
# Add --apply only after reviewing the per-attempt deletion summary.
```

To continue an interrupted battery, keep the same prompt and attempt settings, set its existing
directory as `output_dir`, and enable `resume` in `runconfig.toml`. The harness restarts incomplete
attempts; it does not resume them mid-step.

OpenAI-compatible failures are retried according to `[api_retry].max_attempts` (default 10), with
structured-response validation inside the same budget. If one call exhausts it, a distributed run
stops the current process and re-queues the same logical attempt according to
`[bench].max_api_requeues` (default 3). The equivalent CLI overrides are `--api-max-attempts` and
`--max-api-requeues`. Each discarded process is retained beside the active try as
`tryNN.requeueNN`; after the re-queue budget is exhausted, the final result is recorded as
`api_retry_exhausted`.

Lease requests time out and reconnect after `[bench].lease_acquire_timeout` seconds (default 30),
with capped exponential backoff. Logical retries also refresh fleet health before each acquisition;
the dashboard shows that recovery state and its latest fleet diagnosis.

Once leased, each ordinary sandbox websocket round trip is bounded by
`[bench].sandbox_command_timeout` seconds (default 10). Exceeding that positive, finite deadline
stops the now-uncertain agent process. The harness then resets the sandbox and probes the same
serialized command lane: success returns the sandbox to the pool, while a failed reset/probe
quarantines it. Either way, the logical attempt restarts from a clean episode and the discarded
execution is treated as infrastructure rather than agent failure. The effective timeout is stored
with the battery and preserved by resumes and dashboard retries.
`WaitUntilReady` retains its separate startup deadline because the simulator intentionally holds
that request open while a reset settles.

The dashboard can grade, kill, and retry attempts. It has no authentication, so keep its default
localhost binding unless the network is trusted.
