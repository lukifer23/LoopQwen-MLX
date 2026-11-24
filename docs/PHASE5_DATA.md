# Pointer maps and evaluation cohorts

Each example gives a permutation of eight nodes and a starting node. A map ID hashes its eight successor bytes. An example ID binds the map and start. Display order and renderer also enter token-cache keys. The recurrent prompt is identical across requested depths. Dense terminal controls receive the hop count.

| Cohort | Maps | Use |
|---|---:|---|
| Training | 3,840 | One logical epoch, seed 109 |
| Development | 384 | First 96 at early milestones, all 384 at installation |
| Confirmation | 384 | Sealed before training |
| Counterfactuals | 64 families of related requests | Registered development interventions |

Training and development inherit the Phase 4 pointer permutations. The project generates the data under its MIT license. All starts on a given map belong to the same split. Reordering, relabeling, changing the start and swapping successor edges probe distinct properties. Successor swaps preserve the permutation constraint. Direct pointer traversal and one-hot permutation application verify the targets. The archive verifier independently computes orbits and scores probability argmaxes.

The start belongs to a cycle of length four through eight. Through depth three the uniform conditional baseline is 1/7. A predictor fitted to training counts given start and depth scores 53, 62 and 54 of 384. The installation thresholds are 113, 120 and 113 correct. An unseen permutation can share its cycle structure with many training maps.

Generate the data with `scripts/run_phase5.sh data`. The local manifest records generator and split hashes, start and cycle frequencies and terminal distributions for each depth. [Generator](../loopqwen/data/pointer5.py) · [Registration](../experiments/phase5.json)
