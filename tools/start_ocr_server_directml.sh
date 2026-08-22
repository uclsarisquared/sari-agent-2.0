#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
powershell_exe="/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe"

if [[ ! -x "$powershell_exe" ]]; then
    echo "DirectML OCR requires WSL with Windows PowerShell available." >&2
    exit 1
fi

windows_script="$(wslpath -w "$script_dir/start_ocr_server_directml.ps1")"
exec "$powershell_exe" -NoProfile -ExecutionPolicy Bypass -File "$windows_script" "$@"
