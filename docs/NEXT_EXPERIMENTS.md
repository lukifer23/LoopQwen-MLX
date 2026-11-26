# Research decisions

## November 26 execution and reading studies

1. Compare the deployed language model and untouched Base on live callback sessions. Train a recurrent tool model, correct chat-turn termination symmetrically on a fresh panel, and measure an equally trained one-pass control. Test independent function schemas. The archives preserve the initial failures, the 20/30 versus 5/30 catalog gain and its weak depth effect.
2. Train one typed tool model using independently generated utility calls and human tool demonstrations. Its public execution gate passed at 17/22 versus 0/22 for Base. Keep these weights fixed for multi-schema selection, the earlier failed transfer, both controlled panels, the untouched instruction sibling and the equal-training depth comparison.
3. Exercise a standalone offline export with actual runtime callbacks, then physically replay the full 131,083-token swaps, 200 language generations and raw text retention. Separately measure prefix-state reuse at fixed outputs and a compute-rebalanced long-context decoder with every token retained.
4. Score human reading on paired natural questions that differ in their correct answer. The first response-only fit lost two whole pairs and seven shared-prefix decisions to Base. The next registered fit learns to separate supported and contradicted claims over the same passage and uses the other 32 independent publisher passages. A failed primary closes its own promotion and attribution work.
5. Evaluate the frozen tool weights on independent human database requests. Run model-selected lookups and aggregates against real SQLite data. Check the generated query's denotation again on a fixed resampling of each table. Only completed grounded answers count.

Every new comparison has a frozen item order, direct comparator, threshold and resource stop. The [study registry](../research/registry.json) identifies the exact inputs. The [paper](../paper/loopqwen.pdf) and [trajectory supplement](../paper/tool-trajectories.csv) describe the endpoints and observed failure modes.


## Current deployment

The qualified combined checkpoint is `research/results/native-language-release-02/model`. On controlled three-fact induction, repeating a real Qwen decoder layer improved completed answers over untouched Base, matched one-pass training and the identical weights with two passes bypassed. The common-prefix probes and independent training seed support this attribution. The release also preserved all measured prior workloads and both 131,083-token register answers. The separate family-relationship study closed when its one-pass control scored 44/64 against 34/64 for three passes. The official-test kinship panel remains unscored. General-language evaluation continues to use the pinned Base as the reference. [Native results and scope](CAPABILITY_SCOREBOARD.md#three-fact-induction-with-a-repeated-native-qwen-layer) · [Release qualification](../research/results/native-language-release-02)

## Word problems

Use the 32-update K2 adapter when correct answers on grade-school word problems are needed within 256 generated tokens. It improved over untouched Base on three measured groups and used less generation time. With 1,024 tokens, Base solved eight more of the 48 fresh questions. The longer-budget gate failed on both sources. Base remains the general-language reference and the higher-accuracy choice on this longer-budget cohort. [Per-source counts and costs](CAPABILITY_SCOREBOARD.md#fresh-word-problems-at-two-output-budgets)

The ordinary-depth model with identical rationale supervision closed most of the development gap. It solved 10/32 in 53.6 seconds. The recurrent model solved 13/32 in 64.6 seconds. The paired difference spans only nine discordant questions from one training lineage. Favoring recurrence as a general architecture requires evidence beyond this cohort. [Matched attribution](../research/results/rationale-dense-01)

The follow-up window tested two responses to the loss of longer-budget accuracy. One model learned verified inverse steps in publisher solutions. Another learned preferences between its own correct and incorrect solutions of comparable length. Both used fresh evaluation questions, and both missed their longer-budget gates. The released GSM1K and SVAMP panel remains frozen. [Current training objective](../research/experiments/gsm-rationale-01.json) · [Completed transfer test](../research/results/rationale-transfer-01)

## Graph recurrence

Both arms of the mixed-retention comparison finished all 16 updates. They reached 21/128 functional sixth hops and 9/128 complete paths. Mixing old maps preserved 57/96 permutation third hops, versus the control's 39/96 and the parent's 96/96. The registered 94/96 floor closed this branch. [Paired archive](../research/results/functional-interleave-01)

The November 24 window had closed with the control at one update. The released record preserves the interruption, memory samples and exact AdamW lineage used to resume after the owner freed unified memory. [Original closure](../research/sessions/interleave-closure-1795489467/closure.json) · [November 25 recovery](../research/sessions/2026-11-25/resumption.json)

## Long context

For the measured 131,072-token prompt, resident inference had the lowest latency. Full-precision cache staging on disk reduced peak MLX allocation and reproduced every generated token. The rationale adapter completed its turn but missed the early key. A test of useful retrieval needs successful answers across multiple positions, inputs and lengths. Broad long-context promotion remains closed. [Full-history measurements](PROJECT_REVIEW.md#disk-backed-memory-and-speed)

## Follow-up reassessment

Three completed comparisons identified distinct limits. Additional arithmetic supervision with a verified inverse operation still lost to Base at 1,024 output tokens. Supervised ledger adaptation learned exact 2K lookup, then selected two wrong records at 16K and lost arithmetic accuracy. Larger prefill blocks reduced latency while changing the generated tokens. These results close the corresponding larger studies. [Measured gates](CAPABILITY_SCOREBOARD.md#follow-up-evidence-on-language-retrieval-and-prefill)

Two separate hypotheses use the original qualified 32-update arithmetic checkpoint. The preference study trains only on comparable correct and wrong solutions sampled from that model. The attention study freezes its weights entirely and trains a passage scorer to reweight existing full-history attention. It tests whether a localized passage actually produces a correct generated answer. A selected passage alone earns no answer credit. [Sampled preference registration](../research/experiments/paired-solutions-01.json) · [Learned attention registration](../research/experiments/attention-memory-01.json)

## Decisions from the trained passage reader

Release the frozen reader for single-key register lookup. It preserved all 32 short math streams, scored 8/8 on the 2K/16K qualification, and followed both versions of the 131,078-token key swap. On natural articles it selected the annotated paragraph 3/12 times and achieved 2/12 exact answers against 4/12 for Base. That failure closes a larger study with these same weights. A distinct training hypothesis would need supervision for matching paraphrased questions to natural passages and a retention check on the released register reader. [128K evidence](../research/results/attention-memory-128k) · [Natural reading results](../research/results/prose-reading-01)

The common-budget reasoning experiment finished at 20/32 for K2 versus 25/32 for Base. Sample voting changed two K2 answers in opposite directions and repaired two Base errors. The release therefore keeps the arithmetic performance claim at the measured short budget. The registered comparison closes voting expansion with these weights. [Complete token accounting](../research/results/shared-budget-vote-01)

## November 25 full-budget reassessment

All-layer human-solution training won at 256 tokens and lost 21/48 to 38/48 when completed mathematical content at 1,024 tokens included answers written in prose. Preserving Base's first 512 reasoning tokens brought the next model to 31/48 versus 30/48 on a fresh panel, with a 7.3% reduction in generation time. That gain was too small to establish consistent superiority. In the independent teacher study, zero of the first 39 public-sibling draws finished within 640 tokens. With at most nine draws remaining, its minimum of ten correct complete demonstrations became unreachable. Its dependent optimization and paired evaluation therefore close. [All-layer archive](../research/results/all-layer-reasoning-01) · [Recurrent deliberation archive](../research/results/recurrent-deliberation-01) · [Teacher-yield rule](../research/experiments/projected-distillation-01.json)

The next independent candidate learns from the publisher's complete worked solution. At aligned answer positions the teacher sees the full solution while the student sees a quarter of its reasoning tokens. The student must predict both the answer tokens and the teacher's hidden states. The experiment preserves the original dense weights, uses a low-rank post-training initialization, and optimizes on arithmetic, annotated reading and plain-language retention. The frozen evaluation presents only the original question to each model. [Registered exposure, decision margin and resource ceiling](../research/experiments/privileged-reasoning-01.json) · [Privileged reasoning distillation](https://aclanthology.org/2025.emnlp-main.36/) · [Low-rank post-training extraction](https://aclanthology.org/2025.findings-ijcnlp.125/)

## End of the full-budget reasoning window

The 896-token switch completed 14 of 16 candidate turns and all 16 Base turns before the 315-second worker limit. On the 14 matched items the candidate scored 9/14 in 123.75 seconds and Base scored 10/14 in 121.77 seconds. All four switched turns ended without a complete visible answer. Base scored 11/16 on the full panel. Even if the candidate had solved both remaining items, it could only have tied Base. The registered two-answer gain was already unreachable. Full-panel candidate accuracy and latency remain unmeasured. Its trained weights, Base prefix identities and completed turns were archived. [Partial paired generations](../research/results/late-recurrent-exit-01)

Base remains the reference for ordinary reasoning. The 512-token finisher produced a small positive result that fell short of the required four additional answers and positive gains on both sources. The 896-token training drafts were limited to 512 generated tokens, so the exit model had no supervision at its deployed switching length. A distinct study could train a halting decision on actual late training prefixes and evaluate whether it improves completed mathematical answers on both sources. The current results provide no score for that hypothesis. [Training draft manifest](../research/data/late-recurrent-exit-training.json) · [Reasoning budget analysis](PROJECT_REVIEW.md#what-the-remaining-reasoning-budget-buys)

## After independent full-budget arithmetic confirmation

The 704-token learned-exit checkpoint passed on 32 items and independently on 48 fresh items using the same weights and policy. The mixed deployment study fixes that qualified model before scoring 56 unseen requests across long-budget arithmetic, short-budget arithmetic, entailment, reading and register lookup. Its positive gate requires a full-budget gain, no family accuracy loss, a lower or equal total time, and exact preservation of the ordinary language path. A pass permits a new independent counterfactual through at least 131,072 input tokens on the same checkpoint. The separate trained K1 control uses the identical 128 finisher-training documents and both original evaluation panels to measure the extra pass's contribution. [Frozen mixed study](../research/experiments/conditional-depth-01.json) · [Matched K1 study](../research/experiments/learned-exit-dense-01.json)

## Tool use and human question edits, November 26

The corrected Qwen chat-boundary comparison qualifies the catalog adapter for a narrow system benefit. Its trained one-pass control answered 19/30 against 20/30 for the loop, with lower elapsed time. The BFCL first-turn transfer comparison then gave 8/24 for Base and 0/24 for the catalog adapter. These findings close a causal claim for the extra pass and a general function-calling claim for the original fit.

A broader four-layer suffix scored 38/50 against 7/50 for Base. It missed the registered six-comparison and seven-chain floors, with four and six successes respectively. Domain-level replay showed a distinct failure mode: on several two-site requests the model supplied a plausible site before making either call. The separately registered public typed-execution fit changes the data, initialization and quantization. Its comparison with untouched Base measures the complete system. A matched one-pass sibling control is required for attribution if the primary passes.

After the three unsuccessful gates, further tuning of the same controlled worlds is closed for these fits. Finish the independently registered public execution and human BoolQ contrast comparisons. For BoolQ, the passage stays fixed and a human-authored edit changes the question and the correct answer. Both-member accuracy and the common-prefix probe prevent a positive decision based only on faster completion. The prefix-cache and full-history register studies have separate engineering endpoints. The 128K successor requires a successful short register gate.
