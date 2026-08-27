#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
powershell_exe="/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe"
requested_backend="${SARI_OCR_BACKEND:-$(
    cd "$repo_root"
    python -c 'from sari_runconfig import load_run_config; print(load_run_config("runconfig.toml").get("ocr", "backend", "auto"))'
)}"

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

cd "$repo_root"
exec python -m sari_bench ocr-server --config runconfig.toml --backend "$requested_backend"
