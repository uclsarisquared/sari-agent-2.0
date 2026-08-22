#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
powershell_exe="/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe"
requested_backend="${SARI_OCR_BACKEND:-auto}"

if [[ "$requested_backend" == "directml" ]]; then
    exec "$script_dir/start_ocr_server_directml.sh"
fi

if [[ "$requested_backend" == "auto" && -x "$powershell_exe" ]]; then
    if "$powershell_exe" -NoProfile -Command '
        $roots = @(
            (Join-Path $env:LOCALAPPDATA "sari-ocr-directml\.venv\Scripts\python.exe"),
            (Join-Path $env:LOCALAPPDATA "sari-directml-bench\.venv\Scripts\python.exe")
        )
        if ($roots | Where-Object { Test-Path $_ }) { exit 0 }
        exit 1
    ' >/dev/null; then
        exec "$script_dir/start_ocr_server_directml.sh"
    fi
fi

exec python -m sari_bench ocr-server --backend "$requested_backend"
