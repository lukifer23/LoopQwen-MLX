# Canonical categorical pointer adapter, seed 109

Qwen/Qwen3.5-0.8B-Base at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` with 1,344,512 trainable weights. Rank-eight LoRA adapts the shared layers 10–15. A shared 8,192-weight head reads the final prompt position. The artifact requires the base weights (Apache 2.0) and this repository's native runtime (MLX 0.32.4 and MLX-LM 0.32.2).

This stage applied 480 AdamW updates, 3,840 map exposures and 7,488 state labels under the all objective. The training renderer was canonical. Parent adapter SHA-256: `None`. The data manifest fingerprint is `d3f53a5e6bebc5fcf1e2a69287765a443b9b103635ade5b5afe8ea17e260c8b3`. The adapter and categorical head SHA-256 is `c577e9a9503172799c7df2704021d3808d19158f49ecc14c317a72038396e1e1`.

Development accuracies on eight shuffled orders of 384 unseen permutation maps at hops 1–3 were 100.0%, 65.1%, 46.9%. Median three-hop interactive latency was 38.3 ms. Depths 4–6 are uncached extrapolation. These measurements cover eight-state permutation maps and one training seed. For language generation and retention see the separate research report.

```sh
scripts/run_phase6.sh train --arms canonical --seed 109
.venv/bin/python scripts/phase6_predict.py --arm canonical --verify
.venv/bin/python scripts/phase6_predict.py --arm canonical --map 5 4 7 6 0 2 1 3 --start 0 --depth 3
```
