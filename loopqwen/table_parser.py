"""Learn column, operation and literal pointers with a shared native Qwen layer."""

import mlx.core as mx
import mlx.nn as nn

from .evidence_loop import unit
from .operand_planner import LAYER, adapted_attention_layer
from .table_spans import CONDITIONS, SPAN, arguments, words
from .tool_session import compact

COLUMN_SLOTS, WORD_SLOTS = 24, 64


def encode(model, tok, text, question, columns):
    enc = tok.backend_tokenizer.encode(text, add_special_tokens=False)
    if enc.ids != tok.encode(text, add_special_tokens=False):
        raise ValueError("Schema parser tokenization differs")
    # Locate these particular JSON-encoded header occurrences in the real
    # caller's schema, even if a header name occurs again in the user message.
    array = compact(columns)
    marker = '"enum":' + array
    cursor = text.index(marker) + len('"enum":[')
    column_spans = []
    for col in columns:
        literal = compact(col)
        if text[cursor : cursor + len(literal)] != literal:
            raise ValueError("Different serialized column order")
        column_spans.append((cursor, cursor + len(literal)))
        cursor += len(literal) + 1
    # The pinned chat template strips leading whitespace from the user turn.
    # Whitespace carries no word indices, so align the rendered substring.
    surface = question.strip()
    offset = text.rindex(surface)
    word_spans = [(a + offset, b + offset) for a, b in words(surface)]
    if not columns or not word_spans:
        raise ValueError("Need a nonempty question and table schema")
    prefix = model._execute(
        model.text.model.embed_tokens(mx.array([enc.ids])),
        [(i, 0) for i in range(LAYER)],
    )[0]
    features = unit(prefix)

    def gather(spans):
        out = []
        for left, right in spans:
            ix = [i for i, (a, b) in enumerate(enc.offsets) if b > left and a < right]
            if not ix:
                raise ValueError("Missing schema or question tokens")
            out.append(
                mx.concatenate([features[ix[-1]], features[ix[0] : ix[-1] + 1].mean(0)])
            )
        return mx.stack(out).astype(mx.bfloat16)

    cols, ws = gather(column_spans), gather(word_spans)
    cols = mx.pad(cols, ((0, max(0, COLUMN_SLOTS - cols.shape[0])), (0, 0)))
    ws = mx.pad(ws, ((0, max(0, WORD_SLOTS - ws.shape[0])), (0, 0)))
    mx.eval(cols, ws)
    return cols, ws, len(columns), len(word_spans)


class TableParser(nn.Module):
    def __init__(self, model, rank, depth):
        super().__init__()
        self.depth = depth
        self.layer = adapted_attention_layer(model)
        size = model.args.hidden_size
        self.compress = nn.Linear(2 * size, rank)
        self.types = mx.random.normal((2, rank)) * 0.1
        self.slots = mx.random.normal((1 + CONDITIONS, rank)) * 0.1
        self.expand = nn.Linear(rank, size)
        self.aggregate = nn.Linear(size, 6)
        self.count = nn.Linear(size, CONDITIONS + 1)
        self.comparison = nn.Linear(size, 3)
        self.column_key = nn.Linear(size, rank)
        self.word_key = nn.Linear(size, rank)
        self.column_query = nn.Linear(size, rank)
        self.start_query = nn.Linear(size, rank)
        self.end_query = nn.Linear(size, rank)

    def __call__(self, cols, ws, nc, nw):
        b, c = cols.shape[:2]
        n = ws.shape[1]

        def compress(x, kind):
            return mx.tanh(self.compress(x.astype(mx.float32)) + self.types[kind])

        slots = mx.broadcast_to(self.slots, (b, *self.slots.shape))
        states = mx.concatenate([compress(cols, 0), compress(ws, 1), slots], axis=1)
        h = (0.2 * self.expand(states)).astype(mx.bfloat16)
        cm = mx.arange(c)[None, :] < nc[:, None]
        wm = mx.arange(n)[None, :] < nw[:, None]
        mask = mx.concatenate([cm, wm, mx.ones((b, 1 + CONDITIONS), mx.bool_)], axis=1)
        mask = mx.where(mask[:, None, None, :], 0.0, -1e9).astype(h.dtype)
        for _ in range(self.depth):
            h = self.layer(h, mask=mask)
        h = unit(h)
        readout = h[:, c + n :]
        ck = unit(self.column_key(h[:, :c]))
        wk = unit(self.word_key(h[:, c : c + n]))

        def pointer(query, key, valid):
            x = unit(query) @ key.swapaxes(-1, -2) / (key.shape[-1] ** 0.5)
            return mx.where(valid[:, None, :], x, -1e8)

        cls = pointer(self.column_query(readout), ck, cm)
        cond = readout[:, 1:]
        return (
            self.aggregate(readout[:, 0]),
            self.count(readout[:, 0]),
            cls[:, 0],
            cls[:, 1:],
            self.comparison(cond),
            pointer(self.start_query(cond), wk, wm),
            pointer(self.end_query(cond), wk, wm),
        )


def predict(net, encoded, question, columns):
    c, w, nc, nw = encoded
    agg, num, sel, col, op, start, end = net(
        c[None], w[None], mx.array([nc]), mx.array([nw])
    )
    i = mx.arange(start.shape[-1])
    allowed = (i[:, None] <= i[None, :]) & (i[None, :] < i[:, None] + SPAN)
    joint = mx.where(allowed, start[..., None] + end[:, :, None, :], -1e8)
    spans = mx.argmax(joint.reshape(1, CONDITIONS, -1), -1)[0]
    choices = [mx.argmax(x, -1)[0] for x in (agg, num, sel, col, op)]
    mx.eval(choices, spans)
    agg, num, sel, col, op = [v.tolist() for v in choices]
    indices = [
        (col[j], op[j], *divmod(k, start.shape[-1]))
        for j, k in enumerate(spans.tolist()[:num])
    ]
    return arguments(question, columns, sel, agg, indices), {
        "select": sel,
        "aggregate": agg,
        "conditions": indices,
    }


def query_decoder(net, question, tool, evidence):
    """Use only the supplied model, question and public query_table schema."""
    from .structured_call import decoder

    columns = tool.parameters["properties"]["column"]["enum"]

    def choose(model, tok, text):
        features = encode(model, tok, text, question, columns)
        args, plan = predict(net, features, question, columns)
        evidence.update(plan)
        return {"name": tool.name, "arguments": args}

    return decoder(choose)
