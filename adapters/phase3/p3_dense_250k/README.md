# LoopQwen p3_dense

Experimental unmerged adapter on Qwen/Qwen3.5-0.8B-Base revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. See the bundled Qwen license in the adapters directory. Training uses the V2 mixed procedural corpus and seed 73. The six adapted layers 10–15 use rank-eight shared LoRA. The adapter has 1,336,320 trainable parameters including any learned gate. Execution uses 24 layer calls, mode direct and effective alpha None.

This development checkpoint saw 250,031 targets in 1605 updates. It scored 35/60 final answers with 32 fully verified certificates on the registered V2 development slice. A single training seed limits inference about training variability. Retention NLL was 3.6394 on the V2 prose sample.

Use native MLX 0.32.4 and MLX-LM 0.32.2. Reproduce with `scripts/run_phase3.sh main`, then `scripts/run_phase3.sh promote`. Generate with `python -m loopqwen generate --checkpoint adapters/phase3/p3_dense_250k --prompt 'Evaluate ((8 + 4) * 3 - 5).' --max-tokens 96`. The prompt is raw Base-model input. For the study protocol append the traced prompt suffix from `data/v2`.
