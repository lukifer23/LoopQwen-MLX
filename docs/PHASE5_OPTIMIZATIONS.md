# Native categorical training on the M3 Pro

The hardware screen used 27 real permutation workloads with the pinned BF16 Qwen backbone, FP32 block adapters and eight-way head. Each cell has one warmup followed by two measured, synchronized AdamW updates. Workloads use a 384-position physical cap and the complete prompt length. The logical size and requested depth stay fixed across matched checkpointing policies.

| Checkpoint policy | Logical maps | Passes | Median update s | Range s | Maps/s | Labels/s | Positions/s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 4 | 1 | 0.120 | 0.120–0.120 | 33.3 | 33.3 | 1397 | 2.33 |
| off | 4 | 2 | 0.182 | 0.175–0.189 | 21.9 | 43.9 | 921 | 2.50 |
| off | 4 | 3 | 0.252 | 0.247–0.257 | 15.9 | 47.6 | 667 | 2.66 |
| off | 8 | 1 | 0.198 | 0.198–0.199 | 40.3 | 40.3 | 1693 | 3.16 |
| off | 8 | 2 | 0.331 | 0.328–0.334 | 24.2 | 48.3 | 1015 | 3.60 |
| off | 8 | 3 | 0.460 | 0.459–0.461 | 17.4 | 52.2 | 730 | 3.91 |
| off | 16 | 1 | 0.378 | 0.375–0.380 | 42.4 | 42.4 | 1780 | 3.37 |
| off | 16 | 2 | 0.645 | 0.643–0.647 | 24.8 | 49.6 | 1042 | 3.89 |
| off | 16 | 3 | 0.913 | 0.906–0.920 | 17.5 | 52.6 | 736 | 4.23 |
| block | 4 | 1 | 0.127 | 0.127–0.127 | 31.4 | 31.4 | 1320 | 2.26 |
| block | 4 | 2 | 0.220 | 0.219–0.222 | 18.1 | 36.3 | 762 | 2.27 |
| block | 4 | 3 | 0.308 | 0.308–0.309 | 13.0 | 38.9 | 545 | 2.39 |
| block | 8 | 1 | 0.249 | 0.231–0.267 | 32.1 | 32.1 | 1349 | 2.74 |
| block | 8 | 2 | 0.417 | 0.416–0.418 | 19.2 | 38.4 | 806 | 3.14 |
| block | 8 | 3 | 0.561 | 0.561–0.562 | 14.2 | 42.7 | 598 | 3.35 |
| block | 16 | 1 | 0.444 | 0.440–0.447 | 36.1 | 36.1 | 1515 | 2.91 |
| block | 16 | 2 | 0.772 | 0.770–0.773 | 20.7 | 41.5 | 871 | 3.36 |
| block | 16 | 3 | 1.125 | 1.106–1.145 | 14.2 | 42.7 | 597 | 3.60 |
| all | 4 | 1 | 0.134 | 0.134–0.134 | 29.9 | 29.9 | 1254 | 2.26 |
| all | 4 | 2 | 0.218 | 0.216–0.221 | 18.3 | 36.7 | 770 | 2.27 |
| all | 4 | 3 | 0.309 | 0.307–0.310 | 13.0 | 38.9 | 544 | 2.39 |
| all | 8 | 1 | 0.229 | 0.228–0.230 | 34.9 | 34.9 | 1467 | 2.74 |
| all | 8 | 2 | 0.395 | 0.393–0.397 | 20.3 | 40.5 | 851 | 3.14 |
| all | 8 | 3 | 0.563 | 0.563–0.564 | 14.2 | 42.6 | 597 | 3.35 |
| all | 16 | 1 | 0.447 | 0.442–0.453 | 35.8 | 35.8 | 1502 | 2.91 |
| all | 16 | 2 | 0.777 | 0.777–0.777 | 20.6 | 41.2 | 865 | 3.36 |
| all | 16 | 3 | 1.099 | 1.098–1.099 | 14.6 | 43.7 | 612 | 3.60 |

Batch eight without checkpointing had the lowest sum of per-depth median update times under the registered safety limits. The small timing samples support this local policy choice. Sustained runs use the same physical cap, sample membership and memory policy. The Phase 4 language-model targets/s metric counts next-token predictions through a much larger vocabulary head and measures a different workload.
