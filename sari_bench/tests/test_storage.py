"""Process-safe JSON persistence: unique temp files and locked read-modify-write."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from sari_bench import storage


def test_concurrent_writers_never_publish_a_torn_file(tmp_path: Path) -> None:
    path = tmp_path / "attempt.json"
    errors: list[BaseException] = []

    def write(index: int) -> None:
        try:
            for _ in range(50):
                storage.write_json_atomic(path, {"writer": index, "pad": "x" * 4096})
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=write, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert json.loads(path.read_text())["pad"] == "x" * 4096
    assert not list(tmp_path.glob("*.tmp"))


def test_patch_json_never_loses_a_concurrent_update(tmp_path: Path) -> None:
    path = tmp_path / "attempt.json"
    storage.write_json_atomic(path, {"state": "running"})

    def patch(index: int) -> None:
        for step in range(20):
            storage.patch_json(path, {f"w{index}_{step}": True})

    threads = [threading.Thread(target=patch, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    payload = json.loads(path.read_text())
    assert payload["state"] == "running"
    assert sum(1 for key in payload if key.startswith("w")) == 6 * 20


def test_patch_json_drops_fields(tmp_path: Path) -> None:
    path = tmp_path / "attempt.json"
    storage.write_json_atomic(path, {"verified_success": True, "keep": 1})
    merged = storage.patch_json(path, {"verified_verdict": "invalid"}, drop=("verified_success",))
    assert merged == {"keep": 1, "verified_verdict": "invalid"}
    assert json.loads(path.read_text()) == merged
