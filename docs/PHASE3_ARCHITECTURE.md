# Phase 3 recurrent refinement

The text model owns each original layer once. Integer execution sites index the same modules, including their LoRA tensors. The six layers 10–15 execute twice at K=2. A request allocates thirty independent state slots. KV positions, GDN states and convolution histories advance once per input token in each slot. Switching K mid-request would require reconstructing missing histories.

Let the six-layer block be F and its input x. The suffix starts at one of these outputs.

| Mode | Hidden state sent to layer 16 |
|---|---|
| Dense or true K=1 bypass | h = F(x) |
| Direct K=2 | F(h) |
| Stabilized K=2 | h + α (F(h) − h) |
| Input anchor K=2 | F(h + β (x − h)) |

A fixed α=0 still executes both passes. For finite activations it sends the first-pass result into the suffix. Per-pass observers run inside the model and can inspect actual entry and exit activations, including input anchoring. Observers retain reduced evaluation statistics.

The 752,393,024 pretrained parameters are frozen. Each primary arm has 1,336,320 trainable rank-eight adapter parameters. The learned variant adds one FP32 logit initialized at log(0.1/0.9). The fixed variant freezes that scalar and supplies α directly. At any fixed checkpoint the learned α is global across items and difficulties.

MLX MultiOptimizer partitions gradients by parameter path. Each parameter gets one AdamW update. The adapter and gate share the warmup fraction at each update. The gate uses an independent peak LR and defaults to zero weight decay. Gradient accumulation divides by actual scored tokens, followed by joint global-norm clipping at 1. Logs give the normalized gate gradient before clipping, the logit change after AdamW, the effective coefficient and the applied group rates. Fixed and scheduled coefficients override the stored logit. Existing Phase 2 checkpoints keep their original optimizer and tensor layout.
