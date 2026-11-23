# Metal benchmark

| K | Prefill tok/s | Decode tok/s | TTFT ms | Peak MLX GiB |
|---:|---:|---:|---:|---:|
| 1 | 2284.3 | 69.1 | 28.0 | 1.54 |
| 2 | 1874.9 | 57.2 | 34.1 | 1.55 |
| 3 | 1579.4 | 47.0 | 40.5 | 1.56 |

| GDN training backend | Checkpoint layers | Forward ms | Value + gradient ms | Peak MLX GiB |
|---|---|---:|---:|---:|
| reference | True | 140.4 | 545.5 | 2.95 |
| native | True | 42.4 | 146.3 | 1.82 |
| native | False | 44.8 | 129.1 | 2.10 |
