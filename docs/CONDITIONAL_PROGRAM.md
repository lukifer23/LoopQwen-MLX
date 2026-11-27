# Run the instruction-initialized four-layer loop

The local trained model repeats Qwen’s final four hybrid layers and keeps separate recurrent and attention histories on the second visit. A learned readout converts the five hidden states into ordinary next-token logits. The same readout handles natural text and native Qwen tool calls. Its training study includes every supervised group, source annotation, paired SQL result and probability replay. [Fit study](../paper/conditional-fit-study.pdf) · [Architecture](../paper/conditional-feature-study.pdf) · [Frozen comparison](../research/experiments/conditional-program-primary-01.json)

From the project root on Apple Silicon with the locked environment and generated backbone present, run an ordinary question through the existing CLI.

```sh
.venv/bin/python -m loopqwen ask \
  --checkpoint exports/tool-language-packed-05 \
  --session-policy research/results/conditional-program-fit-01 \
  --tool-format qwen --max-turns 3 --max-tokens 384 \
  --prompt 'Explain how to tell whether a difference between two measurements is meaningful.'
```

For a real SQLite example, extract only the input files from the earlier public comparison archive. Both model variants can open the supplied database read-only. Using the original and altered files separately exercises the same question with different underlying values.

```sh
example_dir="$(mktemp -d)"
unzip -q exports/loopqwen-table-policy-01.zip 'table-policy/examples/*' -d "$example_dir"
for version in original altered; do
  .venv/bin/python -m loopqwen ask \
    --checkpoint exports/tool-language-packed-05 \
    --session-policy research/results/conditional-program-fit-01 \
    --tool-format qwen --max-turns 3 --max-tokens 384 \
    --relational-db "$example_dir/table-policy/examples/$version/table.sqlite" \
    --prompt-file "$example_dir/table-policy/examples/question.txt" \
    --compare-base --trace > "$example_dir/$version.json"
done
```

Each trace retains the two complete visible responses, physical prompt and generated IDs, timings, peak allocation, and callbacks. A shared opening and equal output caps make differences attributable to the deployed model configurations. A trained ordinary-depth model with the same instruction initialization, precision and data is required to attribute a performance change specifically to recurrence.

The recurrent policy loads the packed `instruction-all-q4.safetensors` under `exports/conditional-reentry-backbone` and the trained `readout.safetensors` under `research/results/conditional-program-fit-01`. The loader verifies the recorded byte size and SHA-256 of the 424,020,958-byte backbone and the complete fitted readout. The comparison additionally requires the pinned Base reference in the checkpoint. The pretrained instruction sibling, Base weights, tokenizer and fit registration have separate identities. [Weight inventory](../research/results/conditional-program-fit-01/fit.json) · [Native verification](measurements/conditional-trained-native-replay.json)


To reconstruct a missing local instruction backbone from the pinned public donor, run `.venv/bin/python -m scripts.restore_conditional_features conditional-features-01 --restore --backbone-only`. The helper fetches the pinned source if necessary, packs its actual weights with native MLX and checks the sealed SHA-256 before installing the file. To reconstruct missing training features as well, omit `--backbone-only`. The trained readout needs only the packed backbone for inference.


## Held comparison and context status

Across 340 token-matched requests, the fitted loop solved 42/154 objective tool requests and 24/102 ordinary objective questions. Base solved 24 and 18 respectively. The loop generated a statement that retrieved the answer on both table versions for four of 24 independent sources. The earlier Base-rooted loop achieved zero. On the masked reasoning sample the fitted loop delivered 17/48 fully usable answers against 28/48 for Base. In 24 new human instructions the scores were 4/24 and 1/24, with 11 and six materially unsupported responses. End-to-end time fell by 79.5% and cumulative first-token wait rose by 13.8%. This checkpoint missed its release prerequisites before long-context qualification. All 7,591 cached training-feature files were checked against their registered SHA-256 values, and three were independently regenerated from the original token sequences. [Paired results](../paper/conditional-primary.pdf) · [Every runtime call](../paper/conditional-primary-real-calls.csv) · [Rebuild the native features](../scripts/restore_conditional_features.py)


After a provider error, the loop repeatedly issued calls with the same arguments. It obtained zero successful later callbacks on 20 requests with a turn available, compared with 19/41 for Base. On eight requests where both faced an error with turns left, five Base conversations and zero loop conversations got a successful later callback. Neither model delivered a grounded final answer after an error. We verified 211 original training cases for a later repair experiment on 62 distinct databases. The direct comparison and all previously frozen fits retain their original inputs. [Request-level audit](../paper/conditional-primary-tool-recovery.csv) · [Source design and replay](../research/sources/provider-recovery/README.md)
