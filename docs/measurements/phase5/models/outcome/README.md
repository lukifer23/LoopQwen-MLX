# Seed 109 categorical outcome checkpoint

Terminal-only continuation from the installed recurrent checkpoint. The artifact contains 1,344,512 trainable parameters, including the shared eight-class state readout. It requires the official Qwen3.5-0.8B-Base at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. The study code and generated maps use the repository MIT license. The official base weights carry Apache License 2.0.

After update 560, the checkpoint scored 384, 384, 384 correct at one, two and three hops on 384 canonical development maps. Artifact SHA-256 `d9f686c1320074a53d8f388e87da8b4a26ee9d8739e1105e72db6ddbc262fb2e`. Display order, number of hops and rendering can affect these scores. Run `.venv/bin/python scripts/phase5_predict.py --arm outcome --verify` for a fresh native probability comparison. [Research report](../../../../PHASE5_RESULTS.md)
