"""A question state attends twice to user-supplied documents with the same native Qwen layer."""

import mlx.core as mx

from .attention_memory import attach_bias
from .native_evidence import AnchoredNativeEvidence, prefix_regions


def messages(question, titles, articles):
    if not titles or len(titles) != len(articles):
        raise ValueError("Supply a matching title and passage for each document")
    blocks = [title + "\n" + body for title, body in zip(titles, articles, strict=True)]
    return [
        {
            "role": "system",
            "content": "Use the supplied documents to answer the question. Give a short answer phrase.",
        },
        {
            "role": "user",
            "content": "Documents\n\n"
            + "\n\n".join(blocks)
            + "\n\nQuestion\n"
            + question,
        },
    ]


class DocumentReader(AnchoredNativeEvidence):
    def __init__(self, model, rank):
        super().__init__(model, rank, [0, 1])
        # This reader learns document retrieval with actual native query/key scores.
        for unused in ("readout", "language", "format", "content", "gate"):
            delattr(self, unused)
        self.gradient_min_length = 12

    def scores(self, facts, q, depth):
        _, inputs = self.states(facts, q, depth)
        # Equal numbers of distinct supervised heads in both arms.
        first = self.native_scores(inputs[0])[:, :2, :-1].mean(axis=1)
        last = self.native_scores(inputs[-1])[:, 2:4, :-1].mean(axis=1)
        return first, last


def features(model, tok, question, titles, articles):
    # Encode the full independent document contents and the actual question using native layers 0..14.
    texts = [a + "\n" + b for a, b in zip(titles, articles, strict=True)] + [question]
    x, _ = prefix_regions(model, tok, texts, 1)
    return x


class FocusedDecoder:
    def __init__(self, model, section):
        self.model = model
        self.section = section

    def __getattr__(self, name):
        return getattr(self.model, name)

    def make_cache(self):
        cache = self.model.make_cache()
        attach_bias(cache, self.section)
        return cache


def locate(tok, prompt, title, article):
    needle = title + "\n" + article
    a = prompt.index(needle)
    b = a + len(needle)
    encoded = tok._tokenizer(
        prompt, add_special_tokens=False, return_offsets_mapping=True
    )
    if encoded["input_ids"] != tok.encode(prompt, add_special_tokens=False):
        raise ValueError("Inconsistent document tokenization")
    span = [
        i
        for i, (start, end) in enumerate(encoded["offset_mapping"])
        if end > a and start < b
    ]
    return span[0], span[-1] + 1, len(encoded["input_ids"])


def focus(model, tok, reader, question, titles, articles, text, depth, gain):
    x = features(model, tok, question, titles, articles)
    a, b = reader.scores(x[None, :-1], x[None, -1], depth)
    first = mx.argmax(a, -1).item()
    masked = mx.where(mx.arange(len(titles)) == first, -1e8, b[0])
    second = mx.argmax(masked).item()
    start, end, n = locate(tok, text, titles[second], articles[second])
    return FocusedDecoder(
        model, {"start": start, "stop": end, "length": n, "gain": gain}
    ), [
        first,
        second,
    ]
