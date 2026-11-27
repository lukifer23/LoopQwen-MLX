# Running the learned second-visit readout

The policy adds 786,432 trained parameters to the two physical visits of release 05. It reads the change in hidden state at the final repeated transformer layer. The same readout is active on ordinary questions, tool calls and answers after callbacks. This research policy was evaluated with three turns of up to 384 generated tokens each. The accompanying study reports its measured accuracy, retention, latency and memory.

On 112 matched requests the loop delivered 38 successful answers in 127.32 s versus 22 in 587.18 s for Base. The masked reader credited 8/24 human replies for the loop and 16/24 for Base, and 7/8 versus 2/8 current software decisions. Only one of 24 human situations received both exact answers from the loop. This policy missed its release thresholds.

## Run the policy

Restore the qualified `exports/tool-language-packed-05` checkpoint from the local field kit or use the existing export. In the policy add-on, install the supplied wheel in the project's pinned Apple Silicon MLX environment and place the restored checkpoint at `release-05`. The archive contains the three trained matrices, their provenance, three original example requests, the corresponding input wheels and the report. [Release and dependencies](FIELD_KIT.md)

```sh
uv pip install --offline --no-deps runtime/loopqwen-0.1.0-py3-none-any.whl
loopqwen ask --checkpoint release-05 --session-policy policy --tool-format qwen \
  --prompt 'If ice in a glass melts, why can the temperature stay almost constant for a while?' \
  --max-tokens 384 --max-turns 3 --trace --compare-base
```

The three JSON files in `examples/` each contain an exact argument vector for the installed `loopqwen` executable. Run them from the extracted directory. This example replays the first fixed-version wheel question.

```sh
python -c 'import json,subprocess; subprocess.run(["loopqwen",*json.load(open("examples/same_version.json"))],check=True)'
```

Within the original checkout, use `--checkpoint exports/tool-language-packed-05 --session-policy research/results/visit-residual-01` instead. The raw research archive contains every prompt, generated token, callback, training position and optimizer update. The paper reproduces the source-level tests and model-masked review. [Registered decision and limits](../research/experiments/visit-residual-01.json) · [Full evidence](../research/results/visit-residual-01) · [Study and figures](../paper/visit-residual.pdf)

## Reproduce the measurements

```sh
.venv/bin/python -m scripts.visit_residual_oracle
.venv/bin/python -m scripts.research_archive visit-residual-01 --verify
.venv/bin/python -m scripts.visit_residual_exposure --observed research/results/visit-residual-01
.venv/bin/python -m scripts.paper_visit_residual
.venv/bin/python scripts/visit_residual_distribution.py
```

The final command builds the runtime twice from the revision used for evaluation. It independently builds the policy ZIP twice, installs outside the checkout with the model hub offline and network connections disabled, and checks both arms’ generated tokens, actions, observations, terminal answers and recorded attention histories against the primary run. The trained policy uses the qualified release as its parent. Its own full-context quality is subject to the registered promotion criteria.
