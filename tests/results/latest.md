# Benchmark 2026-11-05 17:33 (whisper-vs-parakeet)
Machine: arm64, 36 GB RAM. ▲ better / ▼ worse than the previous run (2026-11-05 17:17), changes under 5% hidden.

## stt-full

| config | recording | seconds | x_realtime | wer | repeat_run | peak_ram_gb | gpu_avg_pct |
|---|---|---|---|---|---|---|---|
| whisper:mlx-community/whisper-large-v3-turbo | monologue-3min | 11.5 | 18.4 | 0.0 | 1 | 20.8 | 80 |
| whisper:mlx-community/whisper-large-v3-turbo | product-demo-1min | 2.0 | 21.9 | 0.0 | 1 | 21.0 | 94 |
| parakeet:mlx-community/parakeet-tdt-0.6b-v2 | monologue-3min | 4.8 | 44.7 | 0.039 | 1 | 23.4 | 89 |
| parakeet:mlx-community/parakeet-tdt-0.6b-v2 | product-demo-1min | 0.7 | 61.9 | 0.015 | 1 | 22.3 | 93 |

## stt-live

| config | recording | seconds | x_realtime | wer | repeat_run | final_rtf | peak_ram_gb | gpu_avg_pct |
|---|---|---|---|---|---|---|---|---|
| whisper:mlx-community/whisper-large-v3-turbo | monologue-3min | 29.2 | 7.3 | 0.035 | 1 | 0.09 | 20.9 | 59 |
| whisper:mlx-community/whisper-large-v3-turbo | product-demo-1min | 6.0 | 7.3 | 0.03 | 1 | 0.09 | 21.0 | 33 |
| parakeet:mlx-community/parakeet-tdt-0.6b-v2 | monologue-3min | 16.9 | 12.6 | 0.051 | 1 | 0.03 | 22.7 | 53 |
| parakeet:mlx-community/parakeet-tdt-0.6b-v2 | product-demo-1min | 3.3 | 13.3 | 0.03 | 1 | 0.03 | 22.7 | 72 |
