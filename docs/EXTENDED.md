# extended comparison

| Model | Answer NLL | Prose NLL | Procedural exact | GSM8K exact | Numeric MATH slice | Actual layer-token executions¹ |
|---|---:|---:|---:|---:|---:|---:|
| base | 1.4288 | 3.5173 | 28/110 | 0/16 | 0/8 | 79,992 |
| untrained_loop | 1.7909 | 3.7315 | 14/110 | 0/16 | 0/8 | 101,370 |
| trained_dense | 0.4983 | 3.5218 | 65/110 | 0/16 | 0/8 | 62,448 |
| trained_loop | 0.5348 | 3.7817 | 59/110 | 0/16 | 1/8 | 78,090 |
| loop_at_1 | 0.6157 | 3.5335 | 55/110 | 0/16 | 0/8 | 62,400 |
| loop_at_3 | 0.7093 | 4.1668 | 41/110 | 0/16 | 1/8 | 93,636 |
| dense_budget | 0.4983 | 3.5218 | 65/110 | 0/16 | 0/8 | 62,448 |

¹ Procedural generation only. Includes prefill. Each generated token after the first requires one decode forward. EOS can leave budget unused.

| Training | Updates | Input tokens | Scored tokens | Final update NLL | Seconds/update | Input tok/s | Peak MLX GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 1000 | 103637 | 21289 | 0.7357 | 0.233 | 444.5 | 2.70 |
| loop | 1000 | 103637 | 21289 | 0.8299 | 0.329 | 315.4 | 2.69 |

The dense budget arm receives a longer per-prompt output allowance at the same nominal decoder-layer budget. The vocabulary head, different layer types and adapter compute make this an approximate compute control.
