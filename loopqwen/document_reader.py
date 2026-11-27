"""Rank original text spans by question overlap and bias the full recurrent attention."""

import math
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
from mlx import nn

from .io import digest, read


class SpanRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.ones((7,))

    def __call__(self, features):
        return features @ self.weight


class DocumentReader:
    def __init__(self, tokenizer, ranker):
        self.tokenizer = tokenizer
        self.ranker = ranker
        self.piece = lru_cache(maxsize=32768)(lambda i: tokenizer.decode([i]))
        self.end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def split(self, ids):
        # Split only a final question paragraph within a supplied user message.
        recent = ids[-384:]
        if self.end not in recent:
            return None
        end = len(ids) - 1 - recent[::-1].index(self.end)
        if (
            self.tokenizer.decode(ids[end + 1 :])
            != "\n<|im_start|>assistant\n<think>\n"
        ):
            return None
        start = max(0, end - 256)
        tail = self.tokenizer.decode(ids[start:end])
        location = tail.rfind("\n\n")
        if location < 0:
            return None
        encoded = self.tokenizer._tokenizer(
            tail, add_special_tokens=False, return_offsets_mapping=True
        )
        if encoded["input_ids"] != ids[start:end]:
            return None
        split = next(
            (
                i
                for i, (_, b) in enumerate(encoded["offset_mapping"])
                if b > location + 2
            ),
            None,
        )
        if split is None or not 2 <= end - start - split <= 128:
            return None
        return ids[: start + split], ids[start + split : end]

    def spans(self, ids):
        start = 0
        spans = []
        for i, token in enumerate(ids):
            if "\n" in self.piece(token) or i == len(ids) - 1:
                stop = i + 1
                while stop - start > 192:
                    spans.append((start, start + 192))
                    start += 128
                if start < stop:
                    spans.append((start, stop))
                start = stop
        return spans

    def features(self, ids, query):
        spans = self.spans(ids)

        def symbols(tokens):
            return [v for x in tokens if (v := self.piece(x).strip().casefold())]

        query = symbols(query)
        terms = {}
        for n in (1, 2, 3):
            for i in range(len(query) - n + 1):
                key = tuple(query[i : i + n])
                if key not in terms:
                    terms[key] = len(terms)
        if not terms or not spans:
            return None
        positions = []
        columns = []
        for row, (a, b) in enumerate(spans):
            tokens = symbols(ids[a:b])
            matches = {
                k
                for n in (1, 2, 3)
                for i in range(len(tokens) - n + 1)
                if (k := terms.get(tuple(tokens[i : i + n]))) is not None
            }
            for k in sorted(matches):
                positions.append(row)
                columns.append(k)
        mask = mx.zeros((len(spans), len(terms)), dtype=mx.float32)
        if positions:
            mask = mask.at[mx.array(positions), mx.array(columns)].add(1)
        count = mask.sum(axis=0)
        idf = mx.log((len(spans) + 1) / (count + 0.5))
        lengths = mx.array([len(k) for k in terms])
        values = []
        for n in (1, 2, 3):
            selected = lengths == n
            weight = idf * selected
            values.extend(
                [
                    (mask @ weight) / mx.maximum((weight * (count > 0)).sum(), 1),
                    (mask * weight).max(axis=1),
                ]
            )
        values.append(mx.log1p(mx.array([b - a for a, b in spans])) / 5)
        strong = (mask * ((lengths >= 2) & (count <= max(1, len(spans) * 0.003)))).sum(
            axis=1
        )
        return mx.stack(values, axis=1), spans, strong

    def select(self, ids):
        parts = self.split(ids)
        if parts is None:
            return None
        representation = self.features(*parts)
        if representation is None:
            return None
        features, spans, strong = representation
        ix = mx.argmax(self.ranker(features)).item()
        if strong[ix].item() == 0:
            return None
        start, stop = spans[ix]
        return {
            "start": start,
            "stop": stop,
            "gain": 5 + max(0, math.log(len(ids) / 2048)),
            "length": len(ids),
        }


class SelectedSpan:
    def __init__(self, section):
        self.section = section

    def select(self, model, ids):
        return self.section


def install(session, tokenizer, directory, checkpoint):
    from .register_session import RegisterSession

    if not isinstance(session, RegisterSession):
        raise TypeError(
            "The document reader requires the full-history packed register runtime"
        )
    path = Path(directory)
    spec = read(path / "registration.json")
    fit = read(path / "fit.json")
    weights = path / "span-ranker.safetensors"
    if (
        not spec.get("document_frequency_reader")
        or digest(weights) != fit["weights"]
        or digest(Path(checkpoint) / "tool-export-files.json")
        != spec["release_inventory"]
    ):
        raise ValueError("Incompatible document-reader weights or release")
    inventory = path / "checksums.json"
    if inventory.is_file() and any(
        digest(path / n) != read(inventory)[n]
        for n in ("registration.json", "fit.json", weights.name)
    ):
        raise ValueError("Document-reader inventory differs")
    ranker = SpanRanker()
    ranker.load_weights(str(weights), strict=True)
    ranker.eval()
    session.document_reader = DocumentReader(tokenizer, ranker)
