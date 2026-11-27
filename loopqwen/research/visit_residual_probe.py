"""Interchange second-visit changes between real questions on the same training background."""

from collections import defaultdict

import mlx.core as mx

from ..identity import fingerprint


def cohort(examples, size=48):
    pairs = defaultdict(list)
    for i, v in enumerate(examples):
        if v["group"] == "effects":
            pairs[v["source"]].append(i)
    assert len(pairs) == 192 and all(len(p) == 2 for p in pairs.values())
    keys = sorted(pairs, key=lambda source: fingerprint([351907, source]))
    return [(source, pairs[source]) for source in keys[:size]]


def read(net, bank, examples):
    pairs = cohort(examples)
    records = []
    for start in range(0, len(pairs), 8):
        first, second, ids, foils, rows = [], [], [], [], []
        for source, indexes in pairs[start : start + 8]:
            for side, i in enumerate(indexes):
                path, labels, _, _, (position, foil) = bank[i]
                states = mx.load(str(path))
                first.append(states["first"][position])
                second.append(states["second"][position])
                ids.append(labels[position])
                foils.append(foil)
                rows.append(
                    {
                        "source": source,
                        "id": examples[i]["id"],
                        "paired_id": examples[indexes[1 - side]]["id"],
                        "position": position,
                        "target": labels[position],
                        "foil": foil,
                    }
                )
        x, y = mx.stack(first), mx.stack(second)
        delta = y.astype(mx.float32) - x.astype(mx.float32)
        swapped = delta.reshape(-1, 2, delta.shape[-1])[:, ::-1].reshape(delta.shape)
        actual = net.refinement(x, y)
        interventions = {
            "donor": y,
            "learned": actual,
            "paired_change": net.refinement.apply_change(x, y, swapped),
        }
        assert mx.array_equal(
            net.refinement.apply_change(x, y, mx.zeros_like(delta)), y
        ).item()
        for condition, state in interventions.items():
            logits = net.project(net.text.model.norm(state)).astype(mx.float32)
            gold = logits[mx.arange(len(rows)), mx.array(ids)]
            margin = gold - logits[mx.arange(len(rows)), mx.array(foils)]
            nll = mx.logsumexp(logits, axis=-1) - gold
            best = mx.argmax(logits, axis=-1)
            for r, m, loss, top in zip(
                rows, margin.tolist(), nll.tolist(), best.tolist(), strict=True
            ):
                r[condition] = {"margin": m, "nll": loss, "top_token": top}
        records.extend(rows)
    assert len(records) == 96
    return records
