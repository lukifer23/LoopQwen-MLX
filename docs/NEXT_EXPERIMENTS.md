# Research decisions

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
