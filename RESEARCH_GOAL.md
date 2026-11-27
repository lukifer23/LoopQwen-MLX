# LoopQwen goal and handoff

**Paused on November 27, 2026.** Build one genuinely looped checkpoint that improves correct, supported answers and real tool use over untouched Qwen3.5-0.8B-Base, retains ordinary language ability, and answers novel questions using at least 131,072 input tokens. Measure quality, elapsed time and memory together. The release target remains open. [Acceptance criteria and proposed margins](docs/RELEASE_EVIDENCE.md)

## Where we stopped

| Checkpoint or study | Verified outcome |
|---|---|
| Latest conversational loop versus Base | 130/144 versus 105/144 requests finished in 110 versus 661 seconds. Supported delivery across the two observed domains was **24.3% versus 45.8%**. The release gate failed. [Paired results](research/results/source-copy-transfer-01/decision.json) · [Quality analysis](docs/measurements/source-copy-supported-delivery.json) |
| Long-context pointer fit | 640 updates on two complete Deno releases. Held source-location loss improved by 4.78 nats. The parent and trained two-read model both solved 18/24 paired token decisions. One trained read solved 20/24. The gate failed. [Figure](paper/long-copy-held.svg) · [Weights and results](docs/measurements/long-copy-fit) |
| Financial repair | 320 human questions on 160 reports have verified calculations. The first fit stopped under memory pressure. The revised fit was paused during feature extraction with **zero optimizer updates**. The streamed implementation passed its causal gradient test. [Pause receipt](docs/measurements/receipt-repair-streamed-pause.json) · [Training protocol](research/diagnostics/receipt-repair-fit-streamed.json) |

## Resume here

1. On the Apple Silicon training host, from the repository root, run `.venv/bin/python -m scripts.resume_receipt_repair check`. On an explicit resume, run the same command with `fit`. It restarts feature extraction from the verified conversational parent and carries forward time already spent. The optimizer never started. The frozen labels, order, memory ceiling and stop rule still apply.
2. Archive the terminal fit receipt and any weights on `main`. Run the cold verifier on completed weights. A passing fit and cold check open the registered 16-report free-generation pilot. A missed prerequisite closes downstream evaluation for those weights. [Commands and artifact locations](docs/TRAINING.md#resume-the-paused-receipt-repair-fit)
3. Only a qualifying candidate goes to one fresh direct Base comparison. Freeze the same input IDs, tools, generation rules, source groups, scorer and budget. Report ordinary reasoning, unseen tools, retained skills, full-document answers and costs for the **same checkpoint**. A qualifying comparison earns the matched ordinary-depth control and second seed.

[Six-page overview](paper/evidence-overview.pdf) · [Current paper](paper/repair-query.pdf) · [Comprehensive draft](paper/loopqwen-repair-query-study.pdf) · [Research model and offline replay](docs/SOURCE_COPY_MODEL.md)
