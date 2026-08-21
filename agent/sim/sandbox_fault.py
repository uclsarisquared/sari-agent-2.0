"""Shared attempt-local fault signalling for a broken/wedged sandbox.

Distributed Sari Bench (`sari_bench/runner.py`) points `SARI_SANDBOX_FAULT_PATH` at an
attempt-local file and polls for it; a file landing there is what turns "this sandbox is broken"
into an immediate kill + quarantine + requeue instead of the attempt running out its full
wall-clock budget before anyone notices. Standalone runs (`SARI_SANDBOX_FAULT_PATH` unset) are a
no-op here - the caller's own raised exception is the only signal in that case.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

FAULT_PATH_ENV = "SARI_SANDBOX_FAULT_PATH"


def signal_fault(code: str, message: str, **extra) -> None:
    """Best-effort write of one fault record. Never raises - reporting a fault must never hide
    the original error the caller is about to raise itself."""
    path_text = os.environ.get(FAULT_PATH_ENV, "").strip()
    if not path_text:
        return
    path = Path(path_text)
    body = {
        "code": code,
        "message": message,
        "detected_at": datetime.now(timezone.utc).isoformat(),
        **extra,
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
        return
