# joint seed 109

Experimental unmerged LoRA for Qwen/Qwen3.5-0.8B-Base, revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. See the Qwen license in the parent adapters tree. Seed 109 saw 100,229 supervised V2 tokens over 619 updates. Training depth K=2, first-exit objective coefficient 0.25 and 1,336,321 trainable parameters. The scalar gate uses sigmoid interpolation whenever active. K=1 deployment executes all 24 original layers once. V3 development K=1 solved 53/150 ID and 45/150 OOD. Single-seed development results constrain claims of generalization.

SHA-256 `a45e16e6d2df51b3513709190b6c41dbe058285773b233782d55f3f3dab29444`. Use the locked native environment and `python -m loopqwen generate --checkpoint adapters/phase4/joint_109_100k --passes 1 --prompt 'Evaluate 2 + 3.' --max-tokens 32`. Reproduce with `scripts/run_phase4.sh dual-exit --seed 109` after the efficiency stage. Full evaluation and latency measurements are in the [Phase 4 report](../../../docs/PHASE4_RESULTS.md).
