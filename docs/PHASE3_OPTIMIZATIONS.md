# Phase 3 Metal measurements

The 100K study used the native MLX 0.32.4 GDN backward, MLX-LM 0.32.2, rank-8 adapters and BF16 on a 14-core M3 Pro with 18 GiB of unified memory. All three arms processed identical examples and batch boundaries.

| Arm | Updates | Targets | Optimizer work (s) | Targets/s | Peak MLX GiB |
|---|---:|---:|---:|---:|---:|
| Dense K=1 | 629 | 100,068 | 298.6 | 335.1 | 6.53 |
| Direct K=2 | 629 | 100,068 | 416.1 | 240.5 | 7.13 |
| Learned K=2 | 629 | 100,068 | 455.6 | 219.6 | 7.13 |

After 325 gated updates, transient macOS warnings prompted a controlled checkpoint at 55,470 targets. The run retained its existing AdamW moments, cursor and sample plan. Per-layer checkpointing on the remaining 304 updates brought peak allocation for that segment to 4.83 GiB. Both promoted arms use per-layer checkpointing from 100K to 250K. The execution-policy registration reproduces the transition.
