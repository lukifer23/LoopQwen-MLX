"""Evaluate the pretraining registration from archived development evidence."""

import gzip
import json


def read(path):
    buf = path.read_bytes()
    return json.loads(gzip.decompress(buf) if path.suffix == ".gz" else buf)


def decide(root):
    spec = read(root / "registration.json")
    prior = read(root / "dataset.json")["prior"]
    runs = read(root / "training.json")
    floors = [
        max(p["prior"], p["majority"], p["conditional"]) + spec["prior_margin"]
        for p in prior
    ]

    def clears(metrics):
        return all(
            m["accuracy"] >= floor for m, floor in zip(metrics, floors, strict=True)
        )

    def mean(arm):
        return sum(m["accuracy"] for m in runs[arm]["metrics"]) / 3

    selected = "bridge" if "bridge" in runs else "recurrent"
    result = read(root / f"predictions/{selected}-480.json.gz")
    diag = read(root / f"predictions/diagnostics-{selected}.json.gz")
    controls = ("middle", "full")
    best = max(controls, key=mean)
    pair = diag["intervention"]["summary"]
    metrics = result["metrics"]
    deep = metrics[2]
    first_on_third = sum(r["pred"][0] == r["gold"][2] for r in result["records"])
    shift = {
        k: [m["prediction_agreement"] / m["n"] for m in diag["behavior"][k]["metrics"]]
        for k in ("relabel", "order")
    }
    effect = {
        k: (v["donor_transition"] - v["natural_donor_transition"]) / v["n"]
        for k in ("other_start", "cross")
        for v in [pair[k + "/all"]]
    }
    extra = root / f"extrapolation/{selected}.json"
    extrap = None
    if extra.exists():
        unwrapped = [r for r in read(extra) if not r["wrapped"]]
        extrap = all(
            r["correct"] / r["n"]
            >= max(1 / 7, r["prior"] / r["n"]) + spec["prior_margin"]
            for r in unwrapped
        )
    persistent = clears(runs["outcome"]["metrics"]) if "outcome" in runs else None
    checks = dict(
        installation=clears(metrics) and deep["correct"] > first_on_third,
        trajectory=deep["trajectory"] / deep["n"] >= spec["trajectory_floor"],
        relabel_and_order=all(
            v >= spec["counterfactual_consistency_floor"]
            for xs in shift.values()
            for v in xs
        ),
        same_and_cross_map_patch=all(
            v >= spec["intervention_delta"] for v in effect.values()
        ),
        unseen_nonwrapping_depths=extrap,
        dense_advantage=mean(selected) >= mean(best) + spec["dense_margin"],
        persisted_with_terminal_loss=persistent,
    )
    return dict(
        selected=selected,
        accuracy_floors=floors,
        first_on_three_hop_terminal=first_on_third,
        third_on_three_hop_terminal=deep["correct"],
        consistency=shift,
        patch_delta=effect,
        strongest_dense=best,
        mean_terminal_advantage=mean(selected) - mean(best),
        checks=checks,
        eligible_for_confirmation=all(checks.values()),
    )
