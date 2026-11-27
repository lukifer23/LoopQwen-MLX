"""Independently reconstruct the two-release gradient study and archive the exact research weights."""

from collections import Counter
from pathlib import Path
import shutil
from statistics import mean

import numpy as np
from safetensors import safe_open

from loopqwen.io import read, digest
from scripts.changed_locale_fit import PLAN, OUT
from scripts.full_document_features import DIR
from scripts.learner_prefix_material import FIT, RESULT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

POLICY = OUT / "research-policy"


def tensors(src):
    with safe_open(str(src), framework="numpy") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def run():
    result = OUT / "receipt.json"
    committed(result)
    a = read(result)
    s = read(PLAN)
    assert a["plan"] == digest(PLAN)
    folder = DIR / digest(PLAN)
    params = folder / "step.safetensors"
    assert a["weights"] == digest(params)
    mats = tensors(params)
    parent = tensors(RESULT / FIT / "step.safetensors")
    fixed = [k for k, x in parent.items() if np.array_equal(x, mats[k])]
    assert len(fixed) == len(parent) == 7
    for v, doc in enumerate(a["features"]):
        assert digest(folder / f"v{v}-prefix.safetensors") == doc["prefix"]
        assert all(
            digest(folder / f"v{v}-{i}.safetensors") == r["sha256"]
            for i, r in enumerate(doc["rows"])
        )
    assert Counter(
        (r["epoch"], r["key"], r["version"]) for r in a["history"]
    ) == Counter(
        (e, k, v) for k in s["train"] for v in range(2) for e in range(s["epochs"])
    )
    out = {}
    for stage in ("initial", "one", "two"):
        rows = a["held"][stage]
        assert len(rows) == 8
        by = {i: [r for r in rows if r["key"] == i] for i in s["held"]}
        out[stage] = sum(all(r["margin"] > 0 for r in rr) for rr in by.values())

    def gain(old, new, metric):
        return mean(
            x[metric] - y[metric]
            for x, y in zip(a["held"][old], a["held"][new], strict=True)
        )

    assert abs(gain("initial", "two", "address") - a["held"]["address_gain"]) < 1e-7
    assert abs(gain("one", "two", "nll") - a["held"]["second_gain"]) < 1e-7
    POLICY.mkdir(exist_ok=True)
    cfg = read(RESULT / FIT / "registration.json")
    spec = {
        k: cfg[k]
        for k in (
            "checkpoint",
            "base_text_sha256",
            "config_sha256",
            "donor",
            "compute_precision",
            "passes",
            "shared_query",
            "revisits",
            "rank",
            "gate_rank",
        )
    }
    persist(
        POLICY / "registration.json",
        dict(
            **spec,
            id="changed-locale-fit",
            fit_seed=s["seed"],
            content_read=True,
            bounded_write=True,
            original_plan=digest(PLAN),
            original_result=digest(result),
        ),
    )
    if not (POLICY / "step.safetensors").exists():
        shutil.copyfile(params, POLICY / "step.safetensors")
    assert digest(POLICY / "step.safetensors") == a["weights"]
    persist(
        POLICY / "fit.json",
        dict(
            weights=a["weights"],
            registration=digest(POLICY / "registration.json"),
            backbone=read(RESULT / FIT / "fit.json")["backbone"],
            updates=len(a["history"]),
            trainable=sum(v.size for k, v in mats.items() if k not in parent),
        ),
    )
    persist(
        POLICY / "checksums.json",
        {
            p: digest(POLICY / p)
            for p in ("registration.json", "step.safetensors", "fit.json")
        },
    )
    persist(
        OUT / "verification.json",
        dict(
            plan=digest(PLAN),
            record=digest(result),
            weights=a["weights"],
            parent_tensors=fixed,
            dual_correct=out,
            all_held_probability_gains=sum(
                x["nll"] > y["nll"]
                for x, y in zip(a["held"]["initial"], a["held"]["two"], strict=True)
            ),
            held_nll_gain=gain("initial", "two", "nll"),
            feature_digests=22,
            source_versions=2,
        ),
    )
    print(out, flush=True)


if __name__ == "__main__":
    run()
