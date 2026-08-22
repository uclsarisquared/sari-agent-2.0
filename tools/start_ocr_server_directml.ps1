param(
    [string]$PythonPath = "",
    [string]$ListenAddress = "127.0.0.1",
    [int]$Port = 9100,
    [int]$DeviceId = 0,
    [string]$ModelCache = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = $env:SARI_OCR_DIRECTML_PYTHON
}
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "sari-ocr-directml\.venv\Scripts\python.exe"),
        (Join-Path $env:LOCALAPPDATA "sari-directml-bench\.venv\Scripts\python.exe")
    )
    $PythonPath = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if ([string]::IsNullOrWhiteSpace($PythonPath) -or -not (Test-Path $PythonPath)) {
    throw "DirectML OCR Python was not found. Set SARI_OCR_DIRECTML_PYTHON to a Windows Python environment containing paddleocr and onnxruntime-directml."
}

if ([string]::IsNullOrWhiteSpace($ModelCache)) {
    $benchmarkCache = Join-Path $env:LOCALAPPDATA "sari-directml-bench\models"
    $ModelCache = if (Test-Path $benchmarkCache) {
        $benchmarkCache
    } else {
        Join-Path $env:LOCALAPPDATA "sari-ocr-directml\models"
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PADDLE_PDX_CACHE_HOME = $ModelCache
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"

Push-Location $repoRoot
try {
    & $PythonPath -m sari_bench ocr-server `
        --backend directml `
        --directml-device-id $DeviceId `
        --host $ListenAddress `
        --port $Port
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
