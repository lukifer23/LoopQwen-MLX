# Phase 4 native performance

MLX 0.32.4 / MLX-LM 0.32.2, M3 Pro 14-core GPU, 18 GiB unified memory. The configuration screen used the same ten seed-109 logical updates (first two warmup). Sequence cap 256, batch four, BF16, gated K=2 with block rank eight. Timers include evaluated optimizer updates. The workload and the measured memory envelope selected 360 padded positions without activation recomputation.

| Cap | Checkpointing | Targets/s | Useful inputs/s | Positions/s | Median update s | Range s | Peak GiB |
|---|---|---|---|---|---|---|---|
| None | off | 298.8 | 581.9 | 604.4 | 0.904 | 0.340–1.280 | 6.85 |
| 256 | off | 283.1 | 551.4 | 556.3 | 0.983 | 0.376–1.272 | 3.76 |
| 256 | block | 254.3 | 495.3 | 499.7 | 1.073 | 0.385–1.501 | 3.31 |
| 256 | all | 233.0 | 453.8 | 457.9 | 1.205 | 0.434–1.563 | 2.67 |
| 360 | off | 305.0 | 593.9 | 600.9 | 0.889 | 0.345–1.230 | 4.63 |
| 360 | block | 266.4 | 518.7 | 524.9 | 0.987 | 0.427–1.409 | 4.01 |
| 360 | all | 255.4 | 497.3 | 503.2 | 1.028 | 0.396–1.560 | 3.15 |
| 512 | off | 300.5 | 585.2 | 597.3 | 0.906 | 0.331–1.304 | 5.61 |
| 512 | block | 264.6 | 515.2 | 525.8 | 1.027 | 0.419–1.391 | 4.77 |
| 512 | all | 242.8 | 472.9 | 482.6 | 1.099 | 0.462–1.454 | 3.72 |
| 768 | off | 310.2 | 604.0 | 622.4 | 0.855 | 0.337–1.276 | 6.62 |
| 768 | block | 279.4 | 544.0 | 560.5 | 0.947 | 0.375–1.428 | 5.17 |
| 768 | all | 244.5 | 476.1 | 490.5 | 1.069 | 0.407–1.792 | 4.21 |

## Sustained production runs

| Arm / milestone | Updates | Targets | Unique | Update seconds | Targets/s | Padding | Peak GiB | Validation s | Checkpoint s | Full invocation wall s¹ |
|---|---|---|---|---|---|---|---|---|---|---|
| dense-250000-k1 | 1610 | 250211 | 6440 | 757.5 | 330.3 | 2.6% | 4.42 | 5.3 | 0.61 | 1049.3 |
| gated-250000-k1 | 1610 | 250211 | 6440 | 1027.8 | 243.4 | 2.6% | 4.75 | 6.6 | 0.65 | 1046.6 |
| time-250000-k1 | 2160 | 334012 | 8640 | 1028.0 | 324.9 | 2.6% | 4.42 | 7.1 | 1.06 | 1049.3 |
| dense-100000-k1 | 619 | 100229 | 2476 | 292.2 | 343.0 | 2.5% | 4.42 | 3.5 | 0.19 | 1049.3 |
| gated-100000-k1 | 619 | 100229 | 2476 | 409.1 | 245.0 | 2.5% | 4.75 | 4.4 | 0.2 | 1046.6 |
| joint-100000-k1 | 619 | 100229 | 2476 | 564.7 | 177.5 | 2.5% | 4.67 | 4.4 | 0.25 | 576.2 |

¹ Every milestone on a trajectory has the same full invocation wall time. The cumulative update, validation and checkpoint columns stop at the named milestone. Small additional costs include parsing, startup, per-update telemetry, serialization of the separate output summary, and finalization.

[Source-resolved configurations, work counters and checkpoint measurements](measurements/phase4/training.json)

The uncapped profile measured 298.8 targets/s at 6.85 GiB. The selected 360-position cap measured 305.0 targets/s at 4.63 GiB on the same examples. A 32% reduction in peak allocation was the strongest measured engineering benefit. Eight timed updates provide limited precision for the small throughput difference. The 250K gated arm sustained 243.4 targets/s across its complete, more varied schedule. Minimum system memory available during those updates was 3.41 GiB.

With real FP32 model weights for the gradient audit and the native GDN backward path, the 0.25/0.75 shared-exit objective agreed with two independent forwards to 1.38×10⁻⁶ relative gradient RMS. Splitting that logical batch gave 7.16×10⁻⁶ relative gradient RMS and 2.14×10⁻⁶ post-update logit RMS. That audit used boundary adapters to cover the shared suffix gradient. The registered language runs used the approximate BF16 backbone with FP32 LoRA and gate parameters. The extended BF16 diagnostic and stock/merged deployment checks remain deferred at closeout. [Numerical audit](measurements/phase4/gradient-audit.json)
