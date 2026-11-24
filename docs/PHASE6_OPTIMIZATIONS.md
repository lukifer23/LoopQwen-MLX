# Phase 6 categorical execution

All matched arms use native BF16 Qwen and the validated Gated DeltaNet backward, FP32 adapters, a logical batch of eight, a padded-position cap of 384, no activation checkpointing and one clipped AdamW update per logical batch. The schedule, target count and executed layer positions are recorded separately. Input maps have a fixed eight-class categorical vocabulary at the readout.

The trainer builds the shuffled map index once per epoch. On the exact seed-109 schedule a five-repeat CPU check reduced median scheduling time for all 480 batches from 0.231 seconds to 0.00062 seconds. All 480 sets of IDs and depths agreed. This difference accounts for a small fraction of a training run.

For a single prompt the recurrent evaluator computes the prefix once and keeps one carried representation through all requested exits. Collecting six readouts requires ten prefix layers and six shared-block applications. The independent-depth reference evaluates the prefix six times and the block 21 times. Dense terminal prompts include a hop count, so each requested depth receives its own forward. Independent depth-request latency is timed separately at batches one and eight.

The prefix reuse benchmark measures two consumers of identical token streams. It includes prefix construction, four timed eight-update sequences in counterbalanced order, cached tensor bytes, disk write/read costs and a gradient comparison on one real batch. The matched primary arms retain their registered execution policy. Training throughput uses synchronized update time, excluding initialization, checkpoint writes and evaluation.
