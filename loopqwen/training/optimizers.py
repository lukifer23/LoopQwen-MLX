import mlx.core as mx
import mlx.optimizers as optim


def optimizer(c):
    def rate(step):
        return c["learning_rate"] * mx.minimum(1.0, (step + 1) / max(1, c["warmup"]))

    return optim.AdamW(
        learning_rate=rate, weight_decay=c["weight_decay"], bias_correction=True
    )
