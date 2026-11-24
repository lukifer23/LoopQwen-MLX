# Phase 7 exposure and completed reasoning answers

The additional exposure improved the existing seed-109 checkpoint. R624 correctly predicted **384/384** canonical third hops and **3,067/3,072** third hops on eight frozen shuffles, compared with 360 and 2,885 at R480. The shuffled difference is **+5.92 percentage points**. All first and second hops were correct under all twelve presentations. The five shuffled third-hop errors were also the only incomplete shuffled trajectories. The worst third-hop condition was rotation by four (380/384). These are repeated displays of 384 development maps and a single training seed.

| Measurement | R480 | R624 |
|---|---:|---:|
| Absolute optimizer updates | 480 | 624 |
| Cumulative training map exposures | 3,840 | 4,992 |
| Distinct training maps | 3,840 | 3,840 |
| Canonical complete paths | 359/384 | 384/384 |
| Complete shuffled paths | 2,873/3,072 | 3,067/3,072 |
| Random third-hop accuracy | 93.91% | 99.84% |

The child restored the complete AdamW generation at update 480. Its next update used epoch one, batch zero. It received 1,152 first-hop, 768 second-hop and 384 third-hop labels in 144 updates (2,304 labels total). Updates took 69.18 synchronized seconds, or 16.65 maps and 33.31 labels per second. Timed training-loop wall time totaled 69.52 seconds across three worker attempts, excluding model loading, checkpoint restore and between-attempt waits. MLX peak was 3.91 GiB and the lowest recorded available system memory after an update was 3.46 GiB. macOS pressure warnings stopped workers at 504, 552 and 624. Each saved a valid checkpoint. After the third warning the run closed at 624. The 720 and 960 milestones were never reached. No execution settings were relaxed. The 300-second update cap was not the limiting factor.

The four numerical installation criteria passed at the safety endpoint. This supports additional exposure on the existing maps. Fresh map diversity and the full two-epoch effect remain unmeasured. The historical full dense comparator saw only one epoch. Phase 6 retains its original failed decision. [Prospective protocol](PHASE7_PLAN.md)

## Paired accuracy and unseen depths

The shuffled third-hop gain had a 95% paired interval of **+4.10 to +7.94 points** from 10,000 bootstrap draws of the 384 maps. The interval describes item uncertainty within this seed. Across all twelve presentations, **379/384** maps had the correct third hop every time. On the eight shuffles, third-hop both-correct consistency was 99.84%, mean correct-state NLL was 0.00915 and ten-bin expected calibration error was 0.00521. The archive includes all per-map scores and probability vectors.

| Correct state out of 384 | Hop 1 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 |
|---|---:|---:|---:|---:|---:|---:|
| R480 canonical | 384 | 383 | 360 | 301 | 187 | 118 |
| R624 canonical | 384 | 384 | 384 | 375 | 349 | 293 |
| R480 fixed shuffle 0 | 384 | 382 | 359 | 307 | 187 | 106 |
| R624 fixed shuffle 0 | 384 | 384 | 383 | 381 | 344 | 284 |

At the sixth hop on shuffle 0, the child got **125/159** new states and **159/225** wrapped states correct. It completed 275 whole six-hop paths. It failed to leave the preceding predicted state in 20 cases and returned prematurely on 26 nonwrapping trajectories. K=6 remains less reliable than K=3. Grouped activation RMS, update RMS, cosine similarity and per-map probability records are in the archive. [Deep predictions](measurements/phase7/deep.json.gz) · [Paired scores](measurements/phase7/decision.json)

## Does the carried state survive a change of order?

The patch replaces the final prompt-position vector after pass one, then runs the recipient's second pass. Each row uses 64 selected map pairs. Selection uses the true trajectories and does not depend on either model prediction.

| Donor → recipient order | Recipient follows donor state | Unpatched | Matched noise |
|---|---:|---:|---:|
| Canonical → canonical | 61/64 | 0/64 | 0/64 |
| Canonical → shuffle | 62/64 | 0/64 | 0/64 |
| Shuffle → canonical | 61/64 | 0/64 | 0/64 |
| Independent shuffles | 62/64 | 0/64 | 0/64 |

On all 64 behavioral cases, relabeling agreed at each hop. The tested reorderings, on-path edits, off-path edits and changed starts each produced the correct three states on all 64 cases. All registered causal gates passed. The transplanted vector can carry information in addition to a node label. The historical full-dense arm achieved 19.11% shuffled third-hop accuracy with 3,840 training exposures versus the child's 4,992. Confirmation and independent-seed replication remain open. [Intervention records and controls](measurements/phase7/causal-624.json.gz)

## Thinking-enabled language calibration

| Untouched Base on the same 8 questions | Concise | Two demonstrations |
|---|---:|---:|
| Generated-token limit per question | 512 | 1,024 |
| Valid numeric answers in first assistant turn | 1/8 | 5/8 |
| Correct on all registered questions | 1/8 | 5/8 |
| Correct among valid answers | 1/1 | 5/5 |
| Actual tokens generated | 3,375 | 5,095 |
| Actual request time, seconds | 53.64 | 84.17 |

Neither prompt reached the registered six valid answers. The 24-question Base-versus-loop comparison and external GSM8K development comparison closed at this gate. These experiments provide no estimate of the loop's completed-answer advantage. Two examples improved completion on these calibration items, but the small cohort also selected the protocol.

The initial scorer mistakenly included the chat end marker in visible answer text and recorded zero valid answers for both prompts. The Base tokenizer's stopping IDs allowed generation beyond that marker, including into additional simulated turns on one question. The corrected audit recovered the exact generated text with the pinned tokenizer and scored only the first assistant turn through its first chat end or EOS. It accepts a visible final number after thinking closes. No questions were regenerated, and all post-boundary tokens and time remain charged. The 16 records, both scorer revisions and token boundaries are archived. [Corrected first-turn scores](measurements/phase7/language/first-turn.json) · [Original generation totals](measurements/phase7/language/decision.json)

## Independent request cost

On 32 warm, fixed-shuffle maps the R624 median single-request categorical latencies at K=1, 2 and 3 were 24.3, 39.2 and 48.8 ms. On 16 maps at K=4, 5 and 6 the medians were 47.8, 59.8 and 70.6 ms for R624 and 47.1, 60.5 and 69.8 ms for R480. The two timing series use different samples and measurement windows. Each request pays for its prefix and every requested block application. Training and evaluation used native MLX/Metal on the M3 Pro, the validated GDN backward implementation, single-unroll trajectory evaluation and an uncached frozen prefix. [Trained depths](measurements/phase7/latency-624.json) · [Untrained depths](measurements/phase7/long-latency.json)

## Full-vocabulary transfer

With the owner's delegation, the follow-up registered R624 as the pilot parent after it cleared every cross-order gate. It restored the installed weights, initialized a fresh AdamW optimizer at LR 0.0001, and optimized equal-weight categorical and vocabulary CE at the selected exit. The prefix and all preceding shared block activations contribute to both outputs. The final prompt position goes through the original suffix and tied 248,320-token projection for vocabulary CE. The target digit appears only in the loss. Each logical batch of eight gets one update.

The first system pressure warning closed the pilot at **60/256** updates, before the planned 64-update milestone. It saw 480 map presentations, 480 categorical targets and 480 vocabulary targets. Each map came from the original training split and received a fresh verbal rendering. There were 40,000 input tokens, 41,280 padded positions, 1,238,400 decoder-layer positions and 480 full vocabulary projections. Updates took **64.36 seconds** (7.46 maps/s, 7.46 vocabulary targets/s, 622 input tokens/s). The training loop took 65.26 seconds. Peak MLX allocation was **4.92 GiB**. The saved boundary was evaluated without another training attempt.

| Same 192 held-out state-machine prompts | First-hop next digit | Second-hop next digit | Third-hop next digit | K=1 scored on third hop |
|---|---:|---:|---:|---:|
| Untouched Base K=1 | 0 | — | — | 0 |
| Installed R624 | 154 | 77 | 39 | 0 |
| Transfer, 60 updates | 192 | 190 | 187 | 0 |

The terminal-digit gain was **148/192 (+77.08 points)** over the same installed model before transfer. Both models used identical tokens and external pass counts. Readout-state accuracy on these verbal prompts rose from 188, 148 and 91 correct to 192 at every trained depth. The same-checkpoint first exit was scored against the same third-hop labels. The map restrictions make hop one and hop three distinct, so this comparison describes the utility of a later exit for this protocol. Untouched Base receives the same depth-independent prompt at ordinary depth. A dense model trained explicitly for the third-hop target remains the appropriate attribution control. The trained room, document and array descriptions shared a single operation with the held-out state-machine wording. [Pilot registration](../experiments/phase7_transfer_parent.json) · [All token predictions](measurements/phase7/transfer/transfer-60.json.gz)

The 148 new correct third-hop vocabulary predictions retained every originally correct item. The transferred model correctly classified 3,052/3,072 original shuffled third-hop presentations, down 0.49 points from 3,067. The registered vocabulary gain, categorical retention and K=1 prose safeguards all passed at the saved endpoint.

| Full V3 retention, 120 passages and 9,328 target tokens | K=1 NLL | K=3 NLL |
|---|---:|---:|
| Untouched Base | 2.9827 | — |
| Installed R624 | 3.0207 | 3.4545 |
| Transferred model | 3.0329 | 3.4881 |

All five per-domain NLLs are archived. For example, general prose reached 3.7599 for Base, 3.8075/4.1996 installed and 3.8190/4.2189 transferred at K=1/K=3. Use the transfer checkpoint at K=3 for the tested three-hop verbal digit task, R624 for the categorical task, and untouched Base at ordinary depth for general prose. A dense model given the same verbal supervision is the next scientific control for attributing the transfer gain. [Retention, gate decisions and full per-domain results](measurements/phase7/transfer/decision.json)

| Registered work | Closeout |
|---|---|
| R480 exact continuation | Interrupted at R624 on system pressure. Parent and child fully evaluated. |
| Exposure, extrapolation, cross-order interventions | Completed on R624. All eligibility gates passed. |
| Thinking calibration | Completed. Both prompts missed the completion floor. |
| Paired thinking-enabled deployment and external benchmark | Failed calibration prerequisite. |
| Full-vocabulary transfer | Interrupted at update 60 on pressure. Held-out digits, bypass, categorical and prose retention completed. |
| Sealed confirmation | Pending independent replication. |

The [reproduction guide](TRAINING.md#phase-7-exact-exposure-continuation) includes runnable training, evaluation and inference commands. The [checksum inventory](measurements/phase7/checksums.json) authenticates both compact releases and all archived scores. Training began at code revision `73af1a8` for exposure and `30794e8` for transfer. The complete updates, ordered example IDs, token digests and parent lineage accompany the adapters. The CPU audit rescores saved predictions. A separate native reload reproduced 48 categorical argmaxes with zero maximum probability difference and all 24 vocabulary argmaxes checked. No sealed item was scored.
