# Categorical state advancement

In this experiment the official Qwen3.5-0.8B-Base reads a complete permutation and starting node. A shared categorical head reads the last real prompt position. Let `p = P(x)`, `h₁ = F(p)` and `hⱼ₊₁ = F(hⱼ)`. P contains layers 0–9. F contains layers 10–15 and reuses the same six decoder modules and rank-eight adapters. The head applies a parameter-free RMS normalization followed by a shared, bias-free FP32 1024-by-8 projection. Predictions require those 8,192 head weights.

```mermaid
flowchart LR
    X[Map and start] --> P[Prefix P] --> A[Shared F]
    A --> B[Same F] --> C[Same F]
    A --> R[Shared state readout]
    B --> R
    C --> R
```

The recurrent model minimizes the mean categorical cross entropy across executed passes for each map. Hidden activations carry information between applications. Gold states enter only the loss. After physical microbatch accumulation, the trainer averages across the eight maps, clips once at norm one and updates once. The categorical path is uncached and executes 16, 22 or 28 decoder layers. A dense middle readout uses 16 layers. The full dense control uses all 24 and final normalization. Both dense arms train on the requested terminal state; their prompts specify the depth. The frozen middle probe trains only the head. Adaptable arms also train 1,336,320 LoRA parameters.

The registered fallback adds a single shared residual bridge before each extra F. Its update is `B(A(concat(RMSNorm(h), RMSNorm(p))))`. A has shape 2048 by 8 and B has shape 8 by 1024, totaling 24,576 weights. B starts at zero. K=1 bypasses the bridge. The cited study uses separate persistent-state and prelude projections and an identity-biased gate. Our smaller concatenated residual changes a single experimental variable. [Published method, §3.2](https://arxiv.org/html/2608.11233v1) · [Implementation](../loopqwen/training/pointer5.py)
