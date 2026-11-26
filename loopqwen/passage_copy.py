"""Shared Qwen language decoding with a learned distribution over user-supplied text."""

import mlx.core as mx
from mlx import nn

from .evidence_loop import unit
from .tool_session import prompt


def request(tok, title, passage, question):
    if not passage.strip() or not question.strip():
        raise ValueError("A passage and question are required")
    body = f"Question\n{question}\n\n{title}\n{passage}"
    text = prompt(
        tok,
        [
            {
                "role": "system",
                "content": "Answer the question using the passage. Give a short answer. If the passage does not provide an answer, reply Cannot answer.",
            },
            {"role": "user", "content": body},
        ],
    )
    # Locate the actual supplied text after chat serialization, including the
    # tokenizer's leading-space convention. Never consult a reference answer.
    start = text.index(body) + len(body) - len(passage)
    encoded = tok._tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = encoded["offset_mapping"]
    positions = [
        i for i, (a, b) in enumerate(offsets) if a < start + len(passage) and b > start
    ]
    if not positions:
        raise ValueError("Empty encoded passage")
    return text, encoded["input_ids"], (positions[0], positions[-1] + 1), offsets, start


class CopyReader(nn.Module):
    def __init__(self, decoder, width):
        super().__init__()
        self.decoder = decoder
        h = decoder.args.hidden_size
        self.key = nn.Linear(h, width)
        self.query = nn.Linear(h, width)
        self.mix = nn.Linear(h, 1)
        self.mix.weight = mx.zeros_like(self.mix.weight)
        self.mix.bias = mx.array([-1.0])
        self.continuation = mx.array(1.0)

    def probabilities(
        self, hidden, keys, previous_ids, preceding_source_ids, source_mask
    ):
        q = unit(self.query(hidden.astype(mx.float32)))
        logits = q @ keys.swapaxes(-1, -2) / q.shape[-1] ** 0.5
        # Repeated copies can proceed from the previously observed token. The
        # learned coefficient and mixture can override this cue at every step.
        follow = previous_ids[..., None] == preceding_source_ids[:, None, :]
        logits = logits + self.continuation * follow.astype(mx.float32)
        attention = mx.softmax(mx.where(source_mask[:, None, :], logits, -1e9), axis=-1)
        gate = mx.sigmoid(self.mix(hidden.astype(mx.float32)))
        return attention, gate

    def source_keys(self, hidden):
        return unit(self.key(hidden.astype(mx.float32)))


class PassageRequest:
    """A request-owned pointer memory populated by the ordinary causal prefill."""

    def __init__(self, reader, source):
        self.reader = reader
        self.args = reader.decoder.args
        self.source = source

    def eval(self):
        self.reader.eval()

    def make_cache(self):
        self.keys, self.ids, self.gates = [], [], []
        self.memory = None
        self.offset = 0
        return self.reader.decoder.make_cache()

    def hidden(self, ids, cache):
        hidden = self.reader.decoder.hidden(ids, cache)
        end = self.offset + ids.shape[1]
        a = max(self.offset, self.source[0])
        b = min(end, self.source[1])
        if a < b:
            slice_ = slice(a - self.offset, b - self.offset)
            keys = self.reader.source_keys(hidden[:, slice_])
            mx.eval(keys)
            self.keys.append(keys)
            self.ids.append(ids[:, slice_])
        self.last = ids[:, -1:]
        self.offset = end
        return hidden

    def project(self, hidden):
        if self.memory is None:
            if self.offset < self.source[1]:
                raise ValueError("Read the complete supplied passage before decoding")
            keys = mx.concatenate(self.keys, axis=1)
            ids = mx.concatenate(self.ids, axis=1)
            previous = mx.pad(ids[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
            mx.eval(keys, ids, previous)
            self.memory = keys, ids, previous
            self.keys.clear()
            self.ids.clear()
        keys, ids, previous = self.memory
        attention, gate = self.reader.probabilities(
            hidden, keys, self.last, previous, mx.ones(ids.shape, mx.bool_)
        )
        vocab = mx.softmax(
            self.reader.decoder.project(hidden).astype(mx.float32), axis=-1
        )
        probabilities = (1 - gate[0, 0]) * vocab[0, 0]
        probabilities = probabilities.at[ids[0]].add(gate[0, 0] * attention[0, 0])
        self.gates.append(float(gate.item()))
        return mx.log(mx.maximum(probabilities, 1e-35))[None, None, :]
