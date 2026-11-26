"""Use one physical gated-delta block for two original depths in the instruction backbone."""

import mlx.core as mx
from mlx import nn

from .deliberation import ordinary_view
from .looped_model import LoopedModel
from .recurrence import Span
from .tool_model import Assistant, bind


class ResidualRepair(nn.Module):
    """Predict the removed layer's residual in units of the input RMS."""

    def __init__(self, width, rank):
        super().__init__()
        self.input = nn.Linear(width, rank, bias=False)
        self.delta = nn.Linear(width, rank, bias=False)
        self.out = nn.Linear(rank, width, bias=False)
        self.out.weight = mx.zeros_like(self.out.weight)
        self.mix = mx.full((width,), 0.3)

    def __call__(self, original, repeated):
        # Layer-wise residual scale varies considerably between positions.
        x = original.astype(mx.float32)
        scale = mx.sqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + 1e-6)
        dx = (repeated.astype(mx.float32) - x) / scale
        h = nn.silu(self.input(x / scale) + self.delta(dx))
        return (x + scale * (self.mix * dx + self.out(h))).astype(original.dtype)


class FoldedDecoder(LoopedModel):
    def __init__(self, text, start, repair, tool_gate=None):
        super().__init__(text, Span(start=start, length=1, passes=2, mode="direct"))
        self.repair = repair
        self.tool_gate = tool_gate
        self.eval()

    @property
    def sites(self):
        sites = super().sites
        return (
            sites
            if self.tool_gate is None
            else (*sites, (len(self.text.layers) - 1, 1))
        )

    def hidden(self, ids, cache=None, window_size=None, observer=None):
        if cache is not None:
            cache.validate(self, window_size)
        i = self.span.start
        sites = self.sites
        kwargs = {"cache": cache, "window_size": window_size, "observer": observer}
        h = self._execute(self.text.model.embed_tokens(ids), sites[: i + 1], **kwargs)
        raw = self._execute(h, [(i, 1)], **kwargs)
        h = self.repair(h, raw)
        end = len(self.text.layers)
        h = self._execute(h, [(k, 0) for k in range(i + 1, end)], **kwargs)
        if self.tool_gate is not None:
            refined = self._execute(h, [(end - 1, 1)], **kwargs)
            h = (h + self.tool_gate * (refined - h)).astype(h.dtype)
        return self.text.model.norm(h)


def fold(tool_model, repair, start):
    ordinary = ordinary_view(tool_model)
    if not (
        ordinary.text.layers[start].is_linear
        and ordinary.text.layers[start + 1].is_linear
    ):
        raise ValueError("Folding requires adjacent gated-delta decoder layers")
    removed = ordinary.text.model.layers.pop(start + 1)
    tool_model.text.model.layers.pop(start + 1)
    plain = FoldedDecoder(ordinary.text, start, repair)
    tools = FoldedDecoder(tool_model.text, start, repair, tool_model.effective_alpha())
    return plain, tools, removed


class FoldedAssistant(Assistant):
    """Keep one instruction backbone resident for ordinary requests and tool calls."""

    def __init__(self, plain, tools, tokenizer):
        self.language = plain
        self._tool_model = tools
        self.tokenizer = tokenizer
        self.dedicated = False
        self.reuse_prefix = False


def from_release(directory, spec):
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    from .io import read

    cfg = read(directory / "config.json")
    tok = load_tokenizer(directory, eos_token_ids=[cfg["text_config"]["eos_token_id"]])
    tools = bind(None, directory / "tool")
    mx.random.seed(spec["seed"])
    repair = ResidualRepair(tools.args.hidden_size, spec["rank"])
    plain, tools, removed = fold(tools, repair, spec["fold_start"])
    return FoldedAssistant(plain, tools, tok), repair, removed
