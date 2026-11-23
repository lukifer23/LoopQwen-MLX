import mlx.core as mx
import mlx.optimizers as optim


def optimizer(c):
    warmup = (
        round(c["steps"] * c["warmup_fraction"])
        if "warmup_fraction" in c
        else c["warmup"]
    )

    def rate(step):
        ramp = c["learning_rate"] * mx.minimum(1.0, (step + 1) / max(1, warmup))
        if c.get("schedule", "constant") == "constant":
            return ramp
        phase = mx.clip((step - warmup) / max(1, c["steps"] - 1 - warmup), 0, 1)
        floor = c.get("min_lr", 0.0)
        return mx.where(
            step < warmup,
            ramp,
            floor + (c["learning_rate"] - floor) * (1 + mx.cos(mx.pi * phase)) / 2,
        )

    return optim.AdamW(
        learning_rate=rate, weight_decay=c["weight_decay"], bias_correction=True
    )
