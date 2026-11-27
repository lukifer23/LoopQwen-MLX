"""Reuse a complete four-layer hybrid block with learned residuals on its second visit."""

from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from .export_runtime import load_export_tokenizer
from .io import digest, read
from .looped_model import LoopedModel
from .observation_review import backbone
from .recurrence import Span
from .tool_model import Assistant


class ReentryResidual(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.up.weight = mx.zeros_like(self.up.weight)

    def __call__(self, state):
        x = state.astype(mx.float32)
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        return (x + rms * self.up(nn.silu(self.down(x / rms)))).astype(state.dtype)


class ReentryModel(LoopedModel):
    def __init__(self, text, rank):
        if len(text.layers) < 4 or rank <= 0 or rank > text.args.hidden_size:
            raise ValueError(
                "The re-entry model needs four native layers and a valid residual width"
            )
        start = len(text.layers) - 4
        if [layer.is_linear for layer in text.layers[start:]] != [
            True,
            True,
            True,
            False,
        ]:
            raise ValueError(
                "Expected three Gated DeltaNet layers followed by full attention"
            )
        super().__init__(text, Span(start=start, length=4, passes=2))
        self.freeze()
        self.residuals = [
            ReentryResidual(self.args.hidden_size, rank) for _ in range(4)
        ]
        self.mixing = mx.zeros((self.args.hidden_size,), dtype=mx.float32)

    def first_visit(self, ids, cache=None, window_size=None, observer=None):
        return self._execute(
            self.text.model.embed_tokens(ids),
            self.sites[:-4],
            cache,
            window_size,
            observer,
        )

    def second_visit(self, first, cache=None, window_size=None, observer=None):
        state = first
        for site, residual in zip(self.sites[-4:], self.residuals, strict=True):
            state = self._execute(state, [site], cache, window_size, observer)
            state = residual(state)
        original = first.astype(mx.float32)
        return (
            original + mx.tanh(self.mixing) * (state.astype(mx.float32) - original)
        ).astype(first.dtype)

    def hidden(self, ids, cache=None, window_size=None, observer=None):
        if cache is not None:
            cache.validate(self, window_size)
        first = self.first_visit(ids, cache, window_size, observer)
        return self.text.model.norm(
            self.second_visit(first, cache, window_size, observer)
        )


def from_base(directory, spec):
    if (
        spec.get("reentry_block") != [20, 21, 22, 23]
        or spec["start_layer"] != 20
        or spec["prefix_bits"] != 4
    ):
        raise ValueError("Incompatible recurrent block or Base quantization")
    mx.random.seed(spec["fit_seed"])
    return ReentryModel(backbone(directory, spec), spec["rank"])


class ReentryAssistant(Assistant):
    native_plain = True

    def __init__(self, directory, policy):
        super().__init__(directory, tool_format="qwen")
        policy = Path(policy)
        spec = read(policy / "registration.json")
        fit = read(policy / "fit.json")
        path = policy / "reentry.safetensors"
        if fit["registration"] != digest(policy / "registration.json") or fit[
            "weights"
        ] != digest(path):
            raise ValueError(
                "The learned re-entry parameters or training registration changed"
            )
        if (policy / "checksums.json").is_file():
            inventory = read(policy / "checksums.json")
            if any(
                digest(policy / k) != inventory[k]
                for k in ("registration.json", "fit.json", path.name)
            ):
                raise ValueError("The re-entry policy differs from its archived model")
        net = from_base(directory, spec)
        trained = mx.load(str(path))
        expected = dict(tree_flatten(net.trainable_parameters()))
        if trained.keys() != expected.keys() or any(
            v.shape != expected[k].shape or v.dtype != expected[k].dtype
            for k, v in trained.items()
        ):
            raise ValueError("Trained recurrent parameters do not fit this decoder")
        net.load_weights(list(trained.items()), strict=False)
        net.eval()
        mx.eval(net.parameters())
        self._tool_model = net
        self.tokenizer = load_export_tokenizer(
            directory,
            eos_token_ids=[
                read(Path(directory) / "config.json")["text_config"]["eos_token_id"]
            ],
        )
        self.reuse_prefix = True
