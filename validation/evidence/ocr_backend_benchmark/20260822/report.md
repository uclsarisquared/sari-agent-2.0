# PaddleOCR ONNX CPU vs DirectML benchmark

Date: 2026-08-22 (Asia/Manila)

## Result

DirectML was 4.47x faster overall and 4.95x faster on full-resolution frames than ONNX Runtime's CPU provider. The recognized text strings matched exactly on all 24 inputs in both passes.

| Warm pass | ONNX CPU | DirectML | DirectML speedup |
| --- | ---: | ---: | ---: |
| All 24 inputs, total | 64.839 s | 14.511 s | 4.47x |
| All inputs, mean | 2.702 s | 0.605 s | 4.47x |
| Full frames, mean | 3.904 s | 0.788 s | 4.95x |
| Center crops, mean | 1.499 s | 0.421 s | 3.56x |
| All-input throughput | 0.370 input/s | 1.654 input/s | 4.47x |

The eight consecutive 1920x1080 run frames took 34.784 seconds on CPU and 7.026 seconds with DirectML, also a 4.95x speedup.

| Consecutive full frame | ONNX CPU | DirectML | Speedup |
| --- | ---: | ---: | ---: |
| `run_step20.png` | 3.042 s | 0.549 s | 5.54x |
| `run_step21.png` | 3.087 s | 0.552 s | 5.59x |
| `run_step22.png` | 4.184 s | 0.818 s | 5.11x |
| `run_step23.png` | 4.877 s | 0.946 s | 5.15x |
| `run_step24.png` | 4.605 s | 0.964 s | 4.78x |
| `run_step25.png` | 4.112 s | 0.832 s | 4.94x |
| `run_step26.png` | 4.733 s | 1.009 s | 4.69x |
| `run_step27.png` | 6.145 s | 1.355 s | 4.54x |

## Test setup

- GPU: AMD Radeon RX 6650 XT, Windows driver 32.0.21045.1000
- CPU: AMD Ryzen 7 5700X, 8 cores / 16 threads
- OS: Windows 11 Home, build 26200
- Python 3.10.21
- PaddleOCR 3.7.0 / PaddleX 3.7.2
- ONNX Runtime DirectML 1.23.0
- Models: PP-LCNet x1.0 text-line orientation, PP-OCRv6 medium detector, and PP-OCRv6 medium recognizer
- Providers compared: `CPUExecutionProvider` and `DmlExecutionProvider` device 0
- DirectML requirements applied: sequential execution and memory patterns disabled

The dataset contained 12 real application screenshots: eight consecutive frames from one benchmark run, three existing evidence frames, and one live frame captured from the current application. Each full frame and its center-half crop were processed twice. The second pass is reported to remove one-time graph and shader warm-up effects.

All three ONNX Runtime sessions were explicitly checked after initialization. The CPU run listed `CPUExecutionProvider`; the DirectML run listed `DmlExecutionProvider` first with CPU fallback available. ONNX Runtime assigned a small number of shape-related operations to CPU, which is expected for the DirectML provider.

Engine initialization is excluded from the comparison because the first CPU initialization also downloaded the models. The raw JSON files contain both passes, timings, provider verification, and recognized strings.
