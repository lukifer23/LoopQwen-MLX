# Phase 6 observed development results

At seed 109 the model trained on randomized edge displays predicted 2,885 of 3,072 third-hop states across eight prespecified shuffles of 384 unseen maps (93.91%). It correctly completed 2,873 three-hop trajectories (93.52%). Its canonical third-hop score was 360/384 (93.75%). The registered canonical floor required 365 correct. Each of the 12 evaluated presentations used the same underlying maps. The number of independent map clusters is 384.

| Presentation | Hop 1 | Hop 2 | Hop 3 | Complete paths |
|---|---:|---:|---:|---:|
| Canonical | 384/384 | 383/384 | 360/384 | 359/384 |
| Reversed | 384/384 | 380/384 | 359/384 | 359/384 |
| Rotation by one | 384/384 | 384/384 | 365/384 | 365/384 |
| Rotation by four | 384/384 | 382/384 | 358/384 | 356/384 |
| Eight shuffles pooled | 3,072/3,072 | 3,050/3,072 | 2,885/3,072 | 2,873/3,072 |

The third-hop answer was correct under all twelve orders on 328 maps. Four maps failed under every order. Among 163 third-hop errors on random displays for which the predecessor's source-key successor differed from its display-slot successor, 25 predictions matched the display slot. The model still makes errors specific to presentation. The similar canonical and shuffled averages also point to a third-hop accuracy limit shared across presentations.

All five primary arms completed the registered 480 updates. Every arm saw the same ordered 3,840 maps, with one presentation per training map. The dense terminal controls received 3,840 labels and the intermediate-supervised arms received 7,488. The canonical floor closes promotion. Phase 6 cross-order transplants, terminal-only continuation, additional seeds, confirmation, functional-graph transfer and ordinary vocabulary-head training remain gated. The earlier Phase 5 transplant, verbal and retention measurements concern different weights. [Registration](PHASE6_PLAN.md) · [Mechanism and readouts](PHASE6_ARCHITECTURE.md)
