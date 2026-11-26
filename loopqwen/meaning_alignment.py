"""Align two supplied texts and feed their learned relation through the repeated Qwen layer."""

import mlx.core as mx
from mlx import nn

from .evidence_loop import unit
from .tool_session import prompt


def request(tok, left, right, instruction):
    body = f"First text\n{left}\n\nSecond text\n{right}"
    text = prompt(
        tok,
        [{"role": "system", "content": instruction}, {"role": "user", "content": body}],
    )
    origin = text.index(body)
    starts = [origin + len("First text\n"), origin + len(body) - len(right)]
    encoded = tok._tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    intervals = []
    for start, supplied in zip(starts, (left, right), strict=True):
        positions = [
            i
            for i, (a, b) in enumerate(encoded["offset_mapping"])
            if a < start + len(supplied) and b > start
        ]
        if not positions:
            raise ValueError("Each side of the comparison needs at least one token")
        intervals.append([positions[0], positions[-1] + 1])
    return text, encoded["input_ids"], intervals


class AlignedReader(nn.Module):
    def __init__(self, decoder, width):
        super().__init__()
        self.decoder = decoder
        hidden = decoder.args.hidden_size
        self.encode = nn.Linear(hidden, width, bias=False)
        self.compare = nn.Linear(4 * width + 2, width)
        self.reduce = nn.Linear(4 * width, width)
        self.write = nn.Linear(width, hidden, bias=False)
        self.write.weight = self.write.weight * 0.05
        self.position_gain = mx.array(1.0)

    def relation(self, left, right, lm, rm):
        def project(x):
            return unit(self.encode(unit(x)))

        left, right = project(left), project(right)
        scores = (left @ right.swapaxes(-1, -2)) / left.shape[-1] ** 0.5

        def compare(x, y, weights, mask, other):
            weights = mx.softmax(mx.where(other[:, None, :], weights, -1e9), -1)
            matched = weights @ y

            def positions(m):
                return mx.arange(m.shape[1])[None, :] / mx.maximum(
                    1, m.sum(-1, keepdims=True) - 1
                )

            # Contextual alignment plus signed and unsigned displacement of matching evidence.
            shift = (
                (weights @ positions(other)[..., None]) - positions(mask)[..., None]
            ) * self.position_gain
            local = nn.gelu(
                self.compare(
                    mx.concatenate(
                        [
                            x,
                            matched,
                            mx.abs(x - matched),
                            x * matched,
                            shift,
                            mx.abs(shift),
                        ],
                        -1,
                    )
                )
            )
            mean = (local * mask[..., None]).sum(1) / mask.sum(-1, keepdims=True)
            peak = mx.max(mx.where(mask[..., None], local, -1e9), axis=1)
            return mx.concatenate([mean, peak], -1)

        a = compare(left, right, scores, lm, rm)
        b = compare(right, left, scores.swapaxes(-1, -2), rm, lm)
        return self.write(
            nn.gelu(self.reduce(mx.concatenate([a + b, mx.abs(a - b)], -1)))
        )

    def condition(self, prefix, message, mask):
        magnitude = mx.sqrt(
            mx.mean(prefix.astype(mx.float32) ** 2, axis=-1, keepdims=True) + 1e-8
        )
        return (
            prefix
            + (magnitude * message[:, None, :] * mask[..., None]).astype(prefix.dtype)
        ).astype(prefix.dtype)

    def read_suffix(self, prefix, cache=None):
        model = self.decoder
        return model.text.model.norm(
            model._execute(prefix, model.sites[model.span.start :], cache)
        )


class PairRequest:
    """The current request owns its two causal source memories and a joint comparison."""

    def __init__(self, net, spans, prompt_length):
        self.net, self.spans, self.prompt_length = net, spans, prompt_length
        self.args = net.decoder.args

    def make_cache(self):
        self.at = 0
        self.parts = [[], []]
        self.relation = None
        return self.net.decoder.make_cache()

    def eval(self):
        self.net.eval()

    def hidden(self, ids, cache):
        model = self.net.decoder
        cache.validate(model, None)
        h = model._execute(
            model.text.model.embed_tokens(ids), model.sites[: model.span.start], cache
        )
        end = self.at + ids.shape[1]
        if self.relation is None:
            for parts, (a, b) in zip(self.parts, self.spans, strict=True):
                lo, hi = max(self.at, a), min(end, b)
                if lo < hi:
                    parts.append(h[:, lo - self.at : hi - self.at])
            if end >= self.prompt_length:
                left, right = [mx.concatenate(v, 1) for v in self.parts]
                self.relation = self.net.relation(
                    left,
                    right,
                    mx.ones(left.shape[:2], mx.bool_),
                    mx.ones(right.shape[:2], mx.bool_),
                )
                mx.eval(self.relation)
                self.parts.clear()
        if self.relation is not None:
            mask = (mx.arange(self.at, end) >= self.prompt_length - 1)[None, :]
            h = self.net.condition(h, self.relation, mask)
        self.at = end
        return self.net.read_suffix(h, cache)

    def project(self, h):
        return self.net.decoder.project(h)
