# Presentation order as a controlled variable

The Phase 6 namespace inherits the MIT-licensed 3,840 training and 384 development permutations from Phase 5. The builder verifies the parent manifest against the published copy and checks the actual split bytes. A new manifest binds the Phase 6 renderer, parent streams and experiment registration. Labels, map IDs, initial states and the seed-109 epoch permutation are fixed across arms. The sealed cohorts are excluded from model selection.

A SHA-256 fingerprint of the study seed, epoch, map ID, example ID and rendering policy seeds a local PRNG. The randomized renderer uniformly shuffles all eight source/destination pairs. The optional mixed policy uses a registered 25% canonical fraction. A tokenization cache uses the complete rendered prompt. The depth-independent recurrent and three-head inputs keep the same text at every requested hop.

Every final development evaluation pairs the same maps under canonical order, reversal, rotations by one and four, and eight seeded random orders. Training and evaluation use separate seeds and policy names. Reported mean random accuracy weights each map equally. The paired bootstrap draws 384 maps with replacement, retaining all their renderings and comparing arms on the same draw. Exact agreement, joint correctness and the fraction of maps correct under every tested order measure different properties.

The map generator intentionally excludes an early return to the starting state. A first-hop readout scored against a third-hop target is therefore a weak comparator. The primary control trains directly on each requested terminal task. [Renderer](../loopqwen/data/pointer6.py) · [Registration](../experiments/phase6.json)
