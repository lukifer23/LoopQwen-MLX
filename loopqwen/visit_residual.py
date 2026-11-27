"""Learn a correction from the second visit while keeping the deployed recurrent core frozen."""

from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from .evidence_loop import unit
from .export_runtime import load_export_tokenizer
from .io import digest, read
from .looped_model import LoopedModel
from .tool_model import Assistant, bind


class VisitResidual(nn.Module):
    """The correction vanishes identically when the recurrent visit changes no feature."""

    def __init__(self, width, rank):
        super().__init__()
        if not 0 < rank <= width:
            raise ValueError(
                "Choose a positive residual rank no larger than the hidden width"
            )
        self.context = nn.Linear(width, rank, bias=False)
        self.change = nn.Linear(width, rank, bias=False)
        self.output = nn.Linear(rank, width, bias=False)
        self.output.weight = mx.zeros_like(self.output.weight)

    def __call__(self, first, second):
        return self.apply_change(
            first, second, second.astype(mx.float32) - first.astype(mx.float32)
        )

    def apply_change(self, first, second, change):
        """Keep both decoder states fixed while independently supplying the measured change."""
        before = first.astype(mx.float32)
        after = second.astype(mx.float32)
        direction = self.change(unit(change))
        condition = nn.silu(self.context(unit(before)))
        scale = mx.sqrt(mx.mean(mx.square(after), axis=-1, keepdims=True) + 1e-8)
        return (after + scale * self.output(direction * condition)).astype(second.dtype)


class VisitResidualModel(LoopedModel):
    def __init__(self, donor, rank):
        if (
            donor.span.passes != 2
            or donor.span.length != 1
            or donor.span.start != len(donor.text.layers) - 1
        ):
            raise ValueError(
                "This readout consumes the two actual visits to the final layer"
            )
        super().__init__(donor.text, donor.span, alpha=0.35)
        self.alpha_logit = donor.alpha_logit
        self.alpha_override = donor.alpha_override
        self.freeze()
        self.refinement = VisitResidual(self.args.hidden_size, rank)

    def visits(self, ids, cache=None, window_size=None, observer=None):
        if cache is not None:
            cache.validate(self, window_size)
        sites = self.sites
        first = self._execute(
            self.text.model.embed_tokens(ids), sites[:-1], cache, window_size, observer
        )
        second = self._execute(first, sites[-1:], cache, window_size, observer)
        return first, second

    def hidden(self, ids, cache=None, window_size=None, observer=None):
        return self.text.model.norm(
            self.refinement(*self.visits(ids, cache, window_size, observer))
        )


def from_release(directory, spec):
    folder = Path(directory) / "tool"
    if digest(folder / "tool-model.json") != spec["donor_metadata_sha256"]:
        raise ValueError(
            "Original recurrent decoder metadata differs from the training donor"
        )
    donor = bind(None, folder)
    mx.random.seed(spec["fit_seed"])
    model = VisitResidualModel(donor, spec["residual_rank"])
    assert dict(tree_flatten(model.trainable_parameters())).keys() == {
        "refinement." + key for key, _ in tree_flatten(model.refinement.parameters())
    }
    return model


class VisitResidualAssistant(Assistant):
    native_plain = True

    def __init__(self, directory, policy):
        super().__init__(directory, tool_format="qwen")
        policy = Path(policy)
        spec = read(policy / "registration.json")
        fit = read(policy / "fit.json")
        if not spec.get("visit_residual") or fit["passes"] != 2:
            raise ValueError("Expected a trained second-visit residual")
        path = policy / "visit-residual.safetensors"
        if digest(path) != fit["weights"]:
            raise ValueError("Trained residual weights differ")
        if (policy / "checksums.json").is_file():
            inventory = read(policy / "checksums.json")
            for name in ("registration.json", "fit.json", path.name):
                if digest(policy / name) != inventory[name]:
                    raise ValueError("Published residual policy differs")
        model = from_release(directory, spec)
        weights = mx.load(str(path))
        expected = dict(tree_flatten(model.refinement.parameters()))
        if weights.keys() != expected.keys() or any(
            v.shape != expected[k].shape or v.dtype != expected[k].dtype
            for k, v in weights.items()
        ):
            raise ValueError("Residual parameter identity differs")
        model.refinement.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        model.eval()
        self._tool_model = model
        cfg = read(Path(directory) / "config.json")
        self.tokenizer = load_export_tokenizer(
            directory, eos_token_ids=[cfg["text_config"]["eos_token_id"]]
        )
        self.reuse_prefix = True
