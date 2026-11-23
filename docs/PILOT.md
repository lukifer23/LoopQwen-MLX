# pilot comparison

| Model | Answer NLL | Prose NLL | Procedural exact | GSM8K exact | Numeric MATH slice | Actual layer-token executions¹ |
|---|---:|---:|---:|---:|---:|---:|
| base | 1.4288 | 3.5173 | 28/110 | 0/16 | 0/8 | 79,992 |
| untrained_loop | 1.7909 | 3.7315 | 14/110 | 0/16 | 0/8 | 101,370 |
| trained_dense | 0.5613 | 3.5377 | 61/110 | 0/16 | 0/8 | 62,112 |
| trained_loop | 0.6392 | 3.7361 | 50/110 | 0/16 | 0/8 | 77,610 |
| loop_at_1 | 0.6704 | 3.5252 | 51/110 | 0/16 | 0/8 | 62,040 |
| loop_at_3 | 0.8066 | 4.1256 | 43/110 | 0/16 | 0/8 | 92,952 |
| dense_budget | 0.5613 | 3.5377 | 61/110 | 0/16 | 0/8 | 62,112 |

¹ Procedural generation only. Includes prefill. Each generated token after the first requires one decode forward. EOS can leave budget unused.

| Training | Updates | Input tokens | Scored tokens | Final update NLL | Seconds/update | Input tok/s | Peak MLX GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 200 | 41833 | 8540 | 0.2535 | 0.654 | 319.7 | 1.83 |
| loop | 200 | 41833 | 8540 | 0.4601 | 0.870 | 240.4 | 1.83 |

The dense budget arm receives a longer per-prompt output allowance at the same nominal decoder-layer budget. The vocabulary head, different layer types and adapter compute make this an approximate compute control.
