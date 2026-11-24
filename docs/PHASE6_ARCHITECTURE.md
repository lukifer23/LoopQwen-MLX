# Shared computation under reordered input

The experiment retains the official BF16 Qwen3.5-0.8B-Base text decoder. Layers 0–9 compute the frozen prefix. Each application of layers 10–15 uses the same pretrained weights and 1,336,320 FP32 trainable LoRA parameters. The direct recurrent arm reads the last real prompt position after each pass through a single 8,192-weight head and averages cross entropy across executed passes within each map. The dense terminal controls receive the requested hop count and read after layer 15 or after the full decoder. The equal-label dense control reads one middle representation with three independent 1024-by-8 heads. Its 1,360,896 trainable parameters include 16,384 additional readout weights compared with the recurrent arm.

```mermaid
flowchart LR
    Map["Map entries and start"] --> P["Frozen prefix"] --> F1["F, pass 1"]
    F1 --> F2["Same F, pass 2"] --> F3["Same F, pass 3"]
    F1 --> R["Shared categorical head"]
    F2 --> R
    F3 --> R
    F3 --> S["Decoder suffix"] --> H["Tied vocabulary head"]
```

The suffix and vocabulary head belong to the language-model pathway. The primary categorical experiment ends at R. State labels enter the loss. Only activations enter the next shared block. A categorical unroll is uncached. The language-model cache still binds a distinct attention/GDN state to each execution site at fixed depth.

```mermaid
flowchart LR
    D["Donor F output at final position"] --> X["Replace final position"]
    C["Recipient F output at all other positions"] --> X
    X --> F["Recipient's next F application"] --> R["State readout"]
```

The cross-map target is the recipient map applied to the donor's first state. The registered cases include matched-norm noise and the original recipient trajectory. These interventions replace a whole hidden vector. [Mechanism trainer](../loopqwen/training/mechanism.py) · [Interventions](../loopqwen/evaluation/order_diagnostics.py)
