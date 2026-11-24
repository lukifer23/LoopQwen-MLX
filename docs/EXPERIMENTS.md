# Experiment protocol

For the completed Phase 5 pointer study, the registry lists the training, evaluation and archival commands. The original language-model protocols follow below. [Study registry](EXPERIMENT_REGISTRY.md#phase-5-pointer-installation) · [Phase 5 results](PHASE5_RESULTS.md)

The primary question is whether the additional application of layers 10–15 improves held-out performance after training, relative to giving the same adapter capacity and data to the original decoder. K=1 uses the stock layer order. K=2 repeats the span. K=3 applies it three times. All indices are zero-based.

## Controls and scoring

The two untrained controls use the same official weights. The trained twins use the same seeded LoRA initialization, selected block and complete projection set, AdamW schedule, split files and ordered examples. The pilot uses 200 optimizer updates of eight single-document microbatches. Each has 1,336,320 trainable parameters. Full-model and vision parameters stay outside the optimizer. The smoke pair uses 12 updates and the recurrent smoke run was restarted from step six.

The procedural test has 220 examples. Scoring takes the first ten instances of each of eleven categories from the fixed shuffled test split, yielding 110 questions for every model. Fixed short prose documents measure retention over 87 next-token targets. A separate answer-loss metric includes numeric response tokens and EOS. Each evaluation record stores its generated text, prompt, gold, decoded token IDs, correctness, seed, token budget and latency. Group results include category, prompt length and annotated step count.

Generation uses greedy decoding with a 16-token output allowance. The scorer reads the first answer line, accepts decimal and rational equivalents, handles a trailing period, and rejects ambiguous alternatives. This short-answer protocol measures both following the format and solving the problem. The additional GSM8K slice contains 16 examples from the official test split. The MATH probe uses the first eight numeric-answer, text-only problems in the pinned original 500-problem split. These slices have low statistical power. [Original MATH-500 split](https://github.com/openai/prm800k/tree/7ecc794703b2877f63226f2477a49b34f9b25163/prm800k/math_splits)

The dense compute control increases the output allowance on each question. For a prompt of length P and loop output cap N, the loop ceiling is 30 × (P + N − 1) layer-token executions. The dense cap is floor(30 × (P + N − 1) / 24) − P + 1. Prefill produces the first output token. Each arm can stop early at EOS. The report records actual executed layer-tokens and wall time. This is a ceiling on nominal decoder work, with different costs for attention, GDN, the head and generated length.

The variable-depth evaluation reuses a single trained recurrent checkpoint at K=1,2,3. K=3 is outside the direct K=2 pilot's training distribution. The placement calibration uses only the validation split and measures untrained six-layer spans starting at 6, 8, 10, 12 and 16. The training span and scoring protocol remain fixed across the reported comparisons.

## Extended pair

The measured 64-token K=2 microbatch of four used 2.68 GiB and averaged 109.9 ms per document for forward plus gradient, versus 141.6 ms for a single document. The extended pair therefore uses four independent right-padded documents per update, without gradient accumulation. Both models start again from the official base and complete 1,000 updates at the same seed. This increases exposure from 1,600 to 4,000 documents per model. Batch grouping and the number of updates both change from the 200-update pilot. Comparisons within each pair control these factors.
