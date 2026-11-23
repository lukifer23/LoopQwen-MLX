"""Native uncached training update for the pinned MLX-LM Qwen3.5 layout."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import compute_g, normalize_qk
from mlx_lm.models.qwen3_5 import GatedDeltaNet


def native_update(q, k, v, a, b, A_log, dt_bias, state=None):
    # Core promotes gamma to the query dtype. Widen q/k to preserve FP32 decay.
    gamma = compute_g(A_log, a, dt_bias)
    y, state = mx.fast.gated_delta_update(
        q.astype(mx.float32), k.astype(mx.float32), v, gamma, mx.sigmoid(b), state
    )
    return y.astype(q.dtype), state


class TrainingGDN(GatedDeltaNet):
    def __call__(self, x, mask=None, cache=None):
        if (
            not self.training
            or cache is not None
            or mask is not None
            or self.sharding_group is not None
        ):
            return super().__call__(x, mask, cache)
        batch, length, _ = x.shape
        qkv = self.in_proj_qkv(x)
        prefix = mx.zeros(
            (batch, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype
        )
        conv = nn.silu(self.conv1d(mx.concatenate([prefix, qkv], 1)))
        q, k, v = mx.split(conv, [self.key_dim, 2 * self.key_dim], -1)
        q, k = normalize_qk(
            q.reshape(batch, length, self.num_k_heads, self.head_k_dim),
            k.reshape(batch, length, self.num_k_heads, self.head_k_dim),
            inv_scale=self.head_k_dim**-0.5,
            eps=1e-6,
        )
        v = v.reshape(batch, length, self.num_v_heads, self.head_v_dim)
        y, _ = native_update(
            q, k, v, self.in_proj_a(x), self.in_proj_b(x), self.A_log, self.dt_bias
        )
        return self.out_proj(
            self.norm(y, self.in_proj_z(x).reshape(v.shape)).reshape(batch, length, -1)
        )


def select(model, backend):
    if backend not in ("native", "reference"):
        raise ValueError(f"Unknown GDN training backend {backend}")
    cls = TrainingGDN if backend == "native" else GatedDeltaNet
    for layer in model.text.layers:
        if layer.is_linear:
            gdn = layer.linear_attn
            if type(gdn) not in (TrainingGDN, GatedDeltaNet):
                raise TypeError(type(gdn))
            # Specialize only this model instance, keeping every weight and its name.
            object.__setattr__(gdn, "__class__", cls)
