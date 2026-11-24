# Released operating points

The native runtime uses Qwen/Qwen3.5-0.8B-Base at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, 752,393,024 unique backbone weights, MLX 0.32.4 and MLX-LM 0.32.2. Each fork below starts from the same Phase 7 verbal adapter `14d34a55a34b3ceef6aa21e230d658e931011c4d60336af17c739d3292ce1954`. Each release includes 1,336,320 block LoRA parameters and the shared 8,192-weight categorical head. The language output uses the original full tied vocabulary projection.

| Release | Operating point | Adapter SHA-256 |
|---|---|---|
| [Dense trace](results/trace-depth-01/model) | K=1 for the explicit three-hop room trace. 96/96 fresh maps in the matched comparison. Categorical and prose retention of this dense fork are unmeasured. | `8be1baed3d8f1226ba91f5b540ee14cfff5bafc96835e92d06313f4b8a40df6b` |
| [Joint state and trace](results/joint-trace-01/model) | K=3 for the registered room wording and intermediate symbolic states. 64/64 complete room traces, 95/96 symbolic third hops. K=3 language NLL 3.487 versus 2.983 for untouched Base K=1. | `75d63d5ca8887fdc7cd4ecb4310fd29f00c39e3cb6b6f653be80487d6af0ff48` |

Both consume 256 maps in 32 updates using the [frozen map split](data/trace-bootstrap-01.json.gz). The dense branch uses 2,304 response targets. The joint branch adds 768 symbolic state labels from separate prompts on those same maps. Prior installation and vocabulary adaptation are shared. The finite room questions use eight-state permutations and explicit three-hop instructions. General instruction following and external reasoning superiority remain unestablished. Use untouched Base at K=1 as the general-language reference. The categorical six-hop release and the single-digit vocabulary release have separately measured task distributions. [Capability scoreboard](../docs/CAPABILITY_SCOREBOARD.md)

```sh
.venv/bin/python scripts/research_predict.py --archive research/results/joint-trace-01 --verify
.venv/bin/python scripts/research_predict.py --archive research/results/trace-depth-01 --depth 1 --verify
```

Exact training commands, budgets and sources are in the [training guide](../docs/TRAINING.md#complete-verified-responses) and the [registrations](experiments). The local optimizer digest and generation path are recorded alongside each compact adapter.
