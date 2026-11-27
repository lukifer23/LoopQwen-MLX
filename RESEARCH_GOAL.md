# LoopQwen goal and handoff

**Native matched-depth study in progress.** Build one genuinely looped checkpoint that improves correct, supported answers and real tool use over untouched Qwen3.5-0.8B-Base, retains ordinary language ability, and answers novel questions using at least 131,072 input tokens. Measure quality, elapsed time and memory together. The release target remains open. [Acceptance criteria and proposed margins](docs/RELEASE_EVIDENCE.md)

## Where we stopped

| Checkpoint or study | Verified outcome |
|---|---|
| Latest conversational loop versus Base | 130/144 versus 105/144 requests finished in 110 versus 661 seconds. Supported delivery across the two observed domains was **24.3% versus 45.8%**. The release gate failed. [Paired results](research/results/source-copy-transfer-01/decision.json) · [Quality analysis](docs/measurements/source-copy-supported-delivery.json) |
| Long-context pointer fit | 640 updates on two complete Deno releases. Held source-location loss improved by 4.78 nats. The parent and trained two-read model both solved 18/24 paired token decisions. One trained read solved 20/24. The gate failed. [Figure](paper/long-copy-held.svg) · [Weights and results](docs/measurements/long-copy-fit) |
| Financial repair | 320 human questions on 160 reports have verified calculations. Both fits reached the registered macOS memory-pressure stop during feature extraction. The resumed fit took 29.82 seconds and completed **zero optimizer updates**. The streamed implementation passed its causal gradient test. [Pause receipt](docs/measurements/receipt-repair-streamed-pause.json) · [Training protocol](research/diagnostics/receipt-repair-fit-streamed.json) |

## Resume here

The financial branch is closed. The verifier found no completed weights and the 16-report pilot prerequisite failed. The three attempts consumed 134.83 seconds including the original interruption and owner-requested pause. [Terminal receipt and charges](docs/measurements/receipt-repair-fit-streamed/closure.json)

The middle-span study repeats layers 12–15 and retains the eight-layer suffix. Base scored 3/8 supported completions on the frozen donor screen and the untouched instruction sibling scored 0/8. Both trainable arms therefore start from the same certified Base text tensors. The two-pass serving exit uses 28 independent hybrid histories. Each arm trains its own suffix adapters. [Prospective comparison](research/diagnostics/middle-depth/registration.md) · [Donor decision](docs/measurements/middle-depth/donor-decision.json) · [Matched fit exposure](research/diagnostics/middle-depth/fit.json)

The 128-token output condition yielded 5/27 supported short answers for each trained model at update 32. Initial two-pass and untouched Base each exhausted this allowance on the 167,014-token complete novel. Output headroom and input length are registered separately. At 4,096 generated tokens per turn the untouched instruction sibling completed the invoice in 9.34 seconds and both follow-ups within eight seconds. Both fitted models exhausted the opening allowance. On paired short and medium novel excerpts, early loop and ordinary depth each scored 0/8 and Base scored 1/8. [Source-level decisions](paper/middle-depth-sources.csv) · [Output budgets](research/diagnostics/middle-depth/output-headroom.md)

On the native checkout, `.venv/bin/python -m scripts.middle_depth_wave fit` restores each arm's checksummed AdamW checkpoint and alternates epochs through update 576. `.venv/bin/python -m scripts.middle_depth_wave short-terminal` reuses finished trajectories. The exclusive worker ledger records every stopped attempt. Both update-32 policy adapters and their installed wheel are stored together in `exports/middle-depth`. Verify the relocated directory with `.venv/bin/python -m scripts.middle_depth_archive verify --directory exports/middle-depth --stage early`.

[Nine-page research paper draft](paper/paper-draft.pdf) · [Evidence overview](paper/evidence-overview.pdf) · [Current paper](paper/middle-depth.pdf) · [Comprehensive draft](paper/loopqwen-repair-query-study.pdf) · [Research model and offline replay](docs/SOURCE_COPY_MODEL.md)
