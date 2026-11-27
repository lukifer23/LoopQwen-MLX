"""Learn how much information to use from an actual second visit to four shared layers."""

from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from . import REVISION
from .export_runtime import load_export_tokenizer
from .hardware import require_metal
from .io import digest, read
from .looped_model import LoopedModel
from .recurrence import Span
from .tool_model import Assistant, quantize_prefix
from .worker import acquire

PACKED = "instruction-all-q4.safetensors"


class Correction(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.up.weight = mx.zeros_like(self.up.weight)

    def __call__(self, x):
        return self.up(nn.silu(self.down(x)))


class ConditionalReadout(nn.Module):
    def __init__(self, width, rank, gate_rank, passes):
        super().__init__()
        self.passes = passes
        self.corrections = [Correction(width, rank) for _ in range(4)]
        self.context = Correction(width, gate_rank)
        self.bias = mx.zeros((width,), dtype=mx.float32)

    def __call__(self, features):
        # The final native visit starts at the ordinary exit only in the two-visit model.
        entry, *states = [features[..., i, :].astype(mx.float32) for i in range(5)]
        ordinary = entry if self.passes == 2 else states[-1]
        gate = mx.tanh(self.bias + self.context(mx.fast.rms_norm(ordinary, None, 1e-6)))
        delta = gate * (states[-1] - entry)
        for state, correction in zip(states, self.corrections, strict=True):
            rms = mx.sqrt(mx.mean(mx.square(state), axis=-1, keepdims=True) + 1e-6)
            delta = delta + rms * correction(state / rms)
        return (ordinary + delta).astype(features.dtype)


class ConditionalReentry(LoopedModel):
    def __init__(self, text, spec):
        span = Span(start=20, length=4, passes=spec["passes"])
        if (
            len(text.layers) != 24
            or span.passes not in (1, 2)
            or [l.is_linear for l in text.layers[-4:]] != [True, True, True, False]
        ):
            raise ValueError(
                "Conditional re-entry requires the registered hybrid four-layer suffix"
            )
        super().__init__(text, span)
        self.freeze()
        self.readout = ConditionalReadout(
            text.args.hidden_size, spec["rank"], spec["gate_rank"], span.passes
        )

    def features(
        self, ids, cache=None, window_size=None, observer=None, positions=None
    ):
        if cache is not None:
            cache.validate(self, window_size)
        sites = self.sites
        h = self._execute(
            self.text.model.embed_tokens(ids), sites[:-4], cache, window_size, observer
        )
        entry = h if positions is None else h[:, positions]
        states = []
        for site in sites[-4:]:
            h = self._execute(h, [site], cache, window_size, observer)
            states.append(h if positions is None else h[:, positions])
        return mx.stack([entry, *states], axis=-2)

    def hidden(self, ids, cache=None, window_size=None, observer=None):
        return self.text.model.norm(
            self.readout(self.features(ids, cache, window_size, observer))
        )

    def last_hidden(self, ids, cache):
        features = self.features(ids, cache, positions=slice(-1, None))
        return self.text.model.norm(self.readout(features))


def from_instruction(spec):
    from .research.projected_distillation import tutor

    require_metal()
    acquire()
    original, _ = tutor({"teacher": spec["donor"]})
    text = original.text
    quantize_prefix(text, len(text.layers), 4)
    mx.eval(text.parameters())
    mx.random.seed(spec["fit_seed"])
    model = ConditionalReentry(text, spec)
    model.eval()
    return model


def resident(directory, folder, spec, identity):
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    require_metal()
    acquire()
    directory, folder = Path(directory), Path(folder)
    if (
        read(directory / "provenance.json")["base_revision"] != REVISION
        or digest(directory / "config.json") != spec["config_sha256"]
        or identity["donor"] != spec["donor"]["sha256"]
    ):
        raise ValueError(
            "The checkpoint configuration or instruction backbone identity differs"
        )
    for path in (
        folder / PACKED,
        directory.parent / "conditional-reentry-backbone" / PACKED,
    ):
        if path.exists():
            if (
                digest(path) != identity["sha256"]
                or path.stat().st_size != identity["bytes"]
            ):
                raise ValueError(
                    "The packed instruction weights differ from the registered tensors"
                )
            text = TextModel(
                TextModelArgs.from_dict(read(directory / "config.json")["text_config"])
            )
            quantize_prefix(text, len(text.layers), 4)
            text.load_weights(str(path), strict=True)
            mx.random.seed(spec["fit_seed"])
            model = ConditionalReentry(text, spec)
            mx.eval(model.parameters())
            model.eval()
            return model
    raise FileNotFoundError(
        "Install the conditional recurrent backbone alongside the trained policy"
    )


class ConditionalAssistant(Assistant):
    native_plain = True

    def __init__(self, directory, policy):
        super().__init__(directory, tool_format="qwen")
        policy = Path(policy)
        s, fit = read(policy / "registration.json"), read(policy / "fit.json")
        path = policy / "readout.safetensors"
        if (
            fit["weights"] != digest(path)
            or fit["registration"] != digest(policy / "registration.json")
            or not s.get("conditional_reentry")
        ):
            raise ValueError("Conditional model weights or registration changed")
        if (policy / "checksums.json").exists():
            inventory = read(policy / "checksums.json")
            if any(
                inventory[name] != digest(policy / name)
                for name in ("registration.json", "fit.json", path.name)
            ):
                raise ValueError("Conditional model differs from its archive")
        net = resident(directory, policy, s, fit["backbone"])
        weights = mx.load(str(path))
        expected = dict(tree_flatten(net.readout.parameters()))
        if weights.keys() != expected.keys() or any(
            v.dtype != expected[k].dtype or v.shape != expected[k].shape
            for k, v in weights.items()
        ):
            raise ValueError("Conditional weights do not fit this readout")
        net.readout.load_weights(list(weights.items()), strict=True)
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
