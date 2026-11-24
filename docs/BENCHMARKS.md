# Metal benchmark

The Phase 5 categorical benchmark reports maps per second, state labels per second and peak memory. The historical language-model benchmark below measures token generation and vocabulary training. [Phase 5 native measurements](PHASE5_OPTIMIZATIONS.md) · [Fixed-work inference latency](PHASE5_RESULTS.md#runtime-and-reproducibility)

| K | Prefill tok/s | Decode tok/s | TTFT ms | Peak MLX GiB |
|---:|---:|---:|---:|---:|
| 1 | 2284.3 | 69.1 | 28.0 | 1.54 |
| 2 | 1874.9 | 57.2 | 34.1 | 1.55 |
| 3 | 1579.4 | 47.0 | 40.5 | 1.56 |

| GDN training backend | Checkpoint layers | Forward ms | Value + gradient ms | Peak MLX GiB |
|---|---|---:|---:|---:|
| reference | True | 140.4 | 545.5 | 2.95 |
| native | True | 42.4 | 146.3 | 1.82 |
| native | False | 44.8 | 129.1 | 2.10 |

The Phase 7 pointer benchmark reports batch-one latency separately from the cost of a multi-exit trajectory. The vocabulary pilot projects the final real prompt position through the original tied head. Its next-digit score and the thinking-enabled chat completion score use different input and answer protocols. [Measured costs and quality](PHASE7_RESULTS.md)

## Continuous language anchors

The raw room task explicitly requests three successive rooms. All compared depths receive identical prompt IDs and the same 24-token greedy generation budget. A verified path requires all three states in order and a line or token boundary. A final number alone can earn terminal-answer credit. The thinking calibration counts a numeric answer only in the completed first assistant turn after its thinking segment closes. Each archive retains all charged token IDs.

GSM8K uses two short arithmetic demonstrations and direct numeric completion on 32 reserved publisher training items. ARC-Easy uses 32 four-choice validation items with mean log probability over each candidate continuation and also reports the unnormalized likelihood sum. Both have fixed source hashes and reserved question IDs. Public pretraining overlap is unknown. These development samples support screening and paired comparisons. [Dataset and licenses](../research/data/external-anchor.json) · [GSM8K publisher](https://github.com/openai/grade-school-math) · [ARC publisher](https://huggingface.co/datasets/allenai/ai2_arc)

On 128 unseen arbitrary document-link maps, the trained K=1 and K=3 models scored 61 and 44 correct terminal answers with batch-one median latencies of 107 and 150 ms. Following only the first link gave 51 correct answers on those same questions. The pending mixed-task retention experiment produced no paired score or latency benchmark. The resumed window charged zero worker seconds. [Quality and costs](CAPABILITY_SCOREBOARD.md) · [Resource accounting](../research/sessions/interleave-closure-1795489467/closure.json)
