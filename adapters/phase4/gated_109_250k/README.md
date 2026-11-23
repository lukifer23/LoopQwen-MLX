# gated seed 109

Experimental unmerged LoRA for Qwen/Qwen3.5-0.8B-Base, revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. See the Qwen license in the parent adapters tree. Seed 109 saw 250,211 supervised V2 tokens over 1610 updates. Training depth K=2, first-exit objective coefficient 0 and 1,336,321 trainable parameters. The scalar gate uses sigmoid interpolation whenever active. K=1 deployment executes all 24 original layers once. V3 development K=1 solved 66/150 ID and 48/150 OOD. Single-seed development results constrain claims of generalization.

SHA-256 `fe472c4fb69db0a0b8663ac36bbf9b1edc9b2062d8368f85503a300c9e5a9cbf`. Use the locked native environment and `python -m loopqwen generate --checkpoint adapters/phase4/gated_109_250k --passes 1 --prompt 'Evaluate 2 + 3.' --max-tokens 32`. Reproduce with `scripts/run_phase4.sh replicate --seed 109` after the efficiency stage. Full evaluation and latency measurements are in the [Phase 4 report](../../../docs/PHASE4_RESULTS.md).
