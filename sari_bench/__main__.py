"""``python -m sari_bench <command>``.

    coordinator   run the sandbox registry
    run           run a prompt battery across the fleet
    status        print the current sandbox pool
    capacity      set the coordinator-wide active lease cap
    quarantine    remove a faulty sandbox from lease eligibility
    unquarantine  reset and restore a quarantined sandbox
    watch         live dashboard for a running battery (run it beside the runner)
    report        flatten a battery into attempts.csv / legs.csv
    video         render an attempt's screenshots into a replay video
    cleanup-captures  safely remove legacy numbered captures after replay validation
    convert-step-pngs  re-encode legacy per-step PNG debug frames as JPEG
    optimize-artifacts  compact safe closed benchmark artifacts (dry-run by default)
    ocr-server    run the shared runner-local PaddleOCR daemon
"""

from __future__ import annotations

import asyncio
import json
import sys

from sari_bench.protocol import DEFAULT_COORDINATOR_PORT

USAGE = __doc__


def _status(argv: list[str]) -> int:
    import argparse

    from sari_bench.client import CoordinatorClient

    parser = argparse.ArgumentParser(prog="sari_bench status")
    parser.add_argument("--coordinator", default=f"ws://localhost:{DEFAULT_COORDINATOR_PORT}")
    parser.add_argument("--json", action="store_true", help="Print parseable fleet-status JSON.")
    args = parser.parse_args(argv)

    async def fetch() -> dict[str, object]:
        async with CoordinatorClient(args.coordinator) as client:
            return await client.fleet_status()

    status = asyncio.run(fetch())
    pool = list(status.get("sandboxes") or [])
    if args.json:
        print(json.dumps(status, indent=2))
        return 0

    if not pool:
        print("No sandboxes registered.")
        print("Total connected sandboxes: 0")
        cap = status.get("capacity_limit")
        print(
            f"Lease capacity: {'all' if cap is None else cap}; "
            f"effective={status.get('effective_capacity', 0)}, "
            f"active={status.get('active_leases', 0)}, "
            f"eligible={status.get('eligible_sandboxes', 0)}"
        )
        return 0

    print(f"{'SANDBOX':<14} {'ADDRESS':<24} {'STATE':<12} {'LEASE':<28} DETAIL")
    for sandbox in pool:
        address = f"{sandbox['host']}:{sandbox['port']}"
        sandbox_label = sandbox.get("sandbox_alias") or str(sandbox["sandbox_id"])[:8]
        lease = sandbox.get("lease_alias") or (
            str(sandbox.get("lease_id") or "")[:8] or "-"
        )
        state = "Quarantined" if sandbox.get("quarantined") else sandbox["state"]
        reset = "-"
        if sandbox.get("quarantined"):
            reset = sandbox.get("quarantine_reason") or "quarantined"
        elif sandbox.get("state") == "Resetting":
            seconds = sandbox.get("reset_seconds")
            reset = (
                f"{seconds}s ({sandbox.get('reset_reason') or 'unknown'})"
                if seconds is not None
                else "quarantined/untracked"
            )
        print(
            f"{sandbox_label:<14} {address:<24} {state:<12} {lease:<28} {reset}"
        )
    print(f"Total connected sandboxes: {len(pool)}")
    cap = status.get("capacity_limit")
    print(
        f"Lease capacity: {'all' if cap is None else cap}; "
        f"effective={status.get('effective_capacity', 0)}, "
        f"active={status.get('active_leases', 0)}, "
        f"eligible={status.get('eligible_sandboxes', 0)}"
        f", quarantined={status.get('quarantined_sandboxes', 0)}"
    )
    return 0


def _capacity(argv: list[str]) -> int:
    import argparse

    from sari_bench.client import CoordinatorClient

    parser = argparse.ArgumentParser(prog="sari_bench capacity")
    parser.add_argument("limit", help="'all' or a non-negative integer")
    parser.add_argument("--coordinator", default=f"ws://localhost:{DEFAULT_COORDINATOR_PORT}")
    args = parser.parse_args(argv)
    if args.limit.lower() == "all":
        limit = None
    else:
        try:
            limit = int(args.limit)
        except ValueError:
            parser.error("limit must be 'all' or a non-negative integer")
        if limit < 0:
            parser.error("limit must be 'all' or a non-negative integer")

    async def update() -> dict[str, object]:
        async with CoordinatorClient(args.coordinator) as client:
            return await client.set_capacity(limit)

    status = asyncio.run(update())
    print(json.dumps(status, indent=2))
    return 0


def _quarantine(argv: list[str], *, clear: bool = False) -> int:
    import argparse

    from sari_bench.client import CoordinatorClient

    command = "unquarantine" if clear else "quarantine"
    parser = argparse.ArgumentParser(prog=f"sari_bench {command}")
    parser.add_argument("sandbox", help="Human alias (preferred) or canonical sandbox ID")
    parser.add_argument("--coordinator", default=f"ws://localhost:{DEFAULT_COORDINATOR_PORT}")
    if not clear:
        parser.add_argument("--reason", default="operator_quarantine")
        parser.add_argument("--source", default="status_cli")
    args = parser.parse_args(argv)

    async def update() -> dict[str, object]:
        async with CoordinatorClient(args.coordinator) as client:
            if clear:
                return await client.unquarantine(args.sandbox)
            return await client.quarantine_sandbox(
                args.sandbox, reason=args.reason, source=args.source
            )

    try:
        result = asyncio.run(update())
    except Exception as error:  # noqa: BLE001 - CLI should explain coordinator-version failures
        print(
            f"{command} failed: {error}. If the coordinator predates quarantine support, "
            "restart it with the updated code.",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    command, rest = argv[0], argv[1:]
    if command == "coordinator":
        from sari_bench.coordinator import main as coordinator_main

        return coordinator_main(rest)
    if command == "run":
        from sari_bench.runner import main as runner_main

        return runner_main(rest)
    if command == "status":
        return _status(rest)
    if command == "capacity":
        return _capacity(rest)
    if command == "quarantine":
        return _quarantine(rest)
    if command == "unquarantine":
        return _quarantine(rest, clear=True)
    if command == "watch":
        from sari_bench.watch.server import main as watch_main

        return watch_main(rest)
    if command == "report":
        from sari_bench.report import main as report_main

        return report_main(rest)
    if command == "video":
        from sari_bench.video import main as video_main

        return video_main(rest)
    if command == "cleanup-captures":
        from sari_bench.cleanup_captures import main as cleanup_main

        return cleanup_main(rest)
    if command == "convert-step-pngs":
        from sari_bench.convert_step_pngs import main as convert_main

        return convert_main(rest)
    if command == "optimize-artifacts":
        from sari_bench.optimize_artifacts import main as optimize_main

        return optimize_main(rest)
    if command == "ocr-server":
        from agent.vision.ocr_server import main as ocr_server_main

        return ocr_server_main(rest)

    print(f"Unknown command: {command}\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
