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
