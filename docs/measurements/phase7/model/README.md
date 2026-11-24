# R624 categorical mechanism

Experimental shared-depth adapter for Qwen/Qwen3.5-0.8B-Base at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` (Apache 2.0). Use native MLX 0.32.4, MLX-LM 0.32.2 and the unmerged checkpoint. The 1,344,512 trained weights comprise 1,336,320 block LoRA weights and 8,192 shared readout weights. The original tied vocabulary head stays frozen. Use K=3 for three-hop tasks. Longer categorical paths degrade, and general-purpose chat improvement remains unestablished.

Trainable weights SHA-256 `a63ba803bb96ea8ecbb411b6ffb2a94a7c350c93fbec31e05e9dfc694d2e9b79`. Data identity `d3f53a5e6bebc5fcf1e2a69287765a443b9b103635ade5b5afe8ea17e260c8b3`.

Local training stage: `scripts/run_phase7.sh exposure`. The exact training parent and AdamW policy are recorded in the stage registration and recovery files.

```sh
.venv/bin/python scripts/phase7_predict.py --map 5 4 7 6 0 2 1 3 --start 0 --depth 3
```

See [Phase 7 results](../../../PHASE7_RESULTS.md) for actual training exposure, retained accuracy, inference cost and limitations.
