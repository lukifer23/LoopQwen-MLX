"""Verify a projected recurrent update and the partitions in its source-balanced corpus."""

import hashlib
from collections import Counter
from pathlib import Path

import mlx.core as mx

from ..io import digest, read


def return_receipt(model, ids):
    ids = ids[:32]
    prefix = model.prefix_hidden(mx.array([ids]))
    ordinary = model.ordinary(prefix)
    baseline = model.text.model.norm(ordinary)
    repeated, gate = model.refine(ordinary)
    original = model.text.layers[-1].self_attn
    adapted = model.refinement.self_attn
    shared = {
        key: getattr(original, key).weight is getattr(adapted, key).linear.weight
        for key in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    return {
        "ids": ids,
        "max_hidden_change": mx.max(mx.abs(baseline - repeated)).item(),
        "shared_original_projections": shared,
        "mean_gate": mx.mean(gate).item(),
    }


def check_inputs(spec, items, training):
    material = read(spec["training"])
    panel = read(spec["material"])
    assert digest(Path(spec["training"])) == spec["training_sha256"]
    assert digest(Path(spec["material"])) == spec["panel_sha256"]
    works = {w["source"]: w for w in material["works"]}
    assert len(works) == 344 and len(training) == 3072 and len(panel["rows"]) == 96
    assert panel["works"] == [w for w in material["works"] if w["split"] == "test"]
    assert Counter(w["suite"] for w in items) == {
        "fresh": 96,
        "prior_works": 128,
        "wiki": 52,
        "legacy": 120,
    }
    # The primary source membership is disjoint before measuring any prediction.
    seen = set()
    for split, rows in (("test", panel["rows"]), ("train", training)):
        counts = Counter()
        for r in rows:
            work = works[r["source"]]
            assert work["split"] == split and work["category"] == r["category"]
            assert len(r["ids"]) == 129 and r["offset"] + 129 <= work["whole_tokens"]
            assert (
                r["id"]
                == hashlib.sha256(
                    b"".join(i.to_bytes(4, "little") for i in r["ids"])
                ).hexdigest()
            )
            if split == "train":
                assert (
                    len(set(r["positions"])) == 32
                    and min(r["positions"]) >= 0
                    and max(r["positions"]) < 128
                )
            words = {tuple(r["ids"][i : i + 24]) for i in range(106)}
            assert seen.isdisjoint(words)
            seen.update(words)
            counts[r["source"]] += 1
        assert counts == {
            s: w["windows"] for s, w in works.items() if w["split"] == split
        }
    for w in items:
        if w["suite"] != "fresh":
            assert all(
                tuple(w["ids"][i : i + 24]) not in seen
                for i in range(len(w["ids"]) - 23)
            )


def check_material(root, spec, items, training):
    check_inputs(spec, items, training)
    first = read(root / "initial-return.json")
    assert first["ids"] == training[0]["ids"][:32] and first["max_hidden_change"] == 0
    assert len(first["shared_original_projections"]) == 4 and all(
        first["shared_original_projections"].values()
    )
    assert abs(first["mean_gate"] - spec["initial_gate"]) < 1e-5
    final = read(root / "trained-return.json")
    assert final["ids"] == first["ids"] and final["max_hidden_change"] > 0
    assert final["shared_original_projections"] == first["shared_original_projections"]
    weights = read(root / "fit.json")["parameters"]
    assert "return_up.weight" in weights and "return_down.weight" in weights
