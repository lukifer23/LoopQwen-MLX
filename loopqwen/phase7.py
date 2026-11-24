"""Phase 7 bounded research stages."""

import argparse
from .training.exposure import ARM, ARCH, OUT, S, train_until
from .data import pointer5, pointer6
from .evaluation.mechanism import evaluate, unpack
from .evaluation.order_scores import metrics, paired, deeper
from .io import read, save


def exposure(*, close=False):
    rows = pointer5.rows("development")
    original = unpack(ARCH / "randomized/orders-480.json.gz")
    parent = metrics(original["records"])
    early = ["canonical", "random0"]
    from .training.mechanism import generation
    from .io import checkpoint_location

    if (OUT / "decision.json").exists():
        return read(OUT / "decision.json")
    if close:
        st = read(checkpoint_location(ARM.root) / "state.json")
        if st["status"] != "unsafe":
            raise ValueError("Early closeout requires a recorded resource stop")
    elif (ARM.root / f"at-{ARM.milestones[0]}.json").exists():
        st = read(generation(ARM, ARM.milestones[0]) / "state.json")
    else:
        st = train_until(ARM.milestones[0])
    n = st["step"]
    child = metrics(
        evaluate(ARM, n, rows[:96], early, 3, OUT / f"mid-{n}.json.gz")["records"]
    )
    ids = {r["example_id"] for r in rows[:96]}
    prior = metrics(
        [
            r
            for r in original["records"]
            if r["example_id"] in ids and r["condition"] in early
        ]
    )
    regress = any(
        round(96 * (a - b)) >= 6
        for k in early
        for a, b in zip(
            prior["conditions"][k]["accuracy"][:2],
            child["conditions"][k]["accuracy"][:2],
        )
    )
    save(OUT / "midpoint.json", dict(parent=prior, child=child, stop=regress))
    if not close and not regress and st["status"] != "time_limit":
        st = train_until(ARM.milestones[-1])
    n = st["step"]
    final = evaluate(ARM, n, rows, pointer6.SPEC["orders"], 3, OUT / "endpoint.json.gz")
    if (
        original["identity"]["token_streams"] != final["identity"]["token_streams"]
        or original["identity"]["items"] != final["identity"]["items"]
    ):
        raise ValueError("Unmatched presentations")
    child = metrics(final["records"])
    comp = paired(parent, child)
    c = child["conditions"]
    g = S["gate"]
    checks = dict(
        canonical=min(c["canonical"]["accuracy"]) >= g["canonical"],
        random=min(c["mean_random"]["accuracy"]) >= g["random"],
        path=c["mean_random"]["trajectory"][-1] >= g["path"],
        gain=comp["difference"][-1] >= g["gain"],
    )
    deep = evaluate(ARM, n, rows, early, 6, OUT / "deep.json.gz", observe=True)
    result = dict(
        step=n,
        training_status=st["status"],
        reached_planned_endpoint=n == ARM.milestones[-1],
        checks=checks,
        installed=all(checks.values()),
        scores=child,
        paired=comp,
        deep=deeper(deep["records"]),
    )
    from .evaluation.mechanism_cost import latency

    latency(ARM, n)
    if result["installed"]:
        from .evaluation.order_diagnostics import diagnose

        d = diagnose(ARM, n)
        checks["relabel"] = (
            min(
                m["prediction_agreement"] / m["n"]
                for m in d["behavior"]["relabel"]["metrics"]
            )
            >= g["relabel"]
        )
        checks["patch"] = all(
            (s["donor_transition"] - s["natural_donor_transition"]) / s["n"]
            >= g["patch_delta"]
            for v in d["interventions"].values()
            for s in [v["summary"]["cross/all"]]
        )
        checks["historical_dense"] = (
            c["mean_random"]["accuracy"][-1]
            > read(ARCH / "results.json")["arms"]["full"]["evaluations"][-1]["scores"][
                "conditions"
            ]["mean_random"]["accuracy"][-1]
        )
    result["robust"] = all(checks.values())
    save(OUT / "decision.json", result)
    print(n, checks, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=["exposure", "close-exposure", "language", "long-latency", "transfer"],
    )
    stage = p.parse_args().stage
    if stage in ("exposure", "close-exposure"):
        exposure(close=stage == "close-exposure")
    elif stage == "language":
        from .evaluation.language7 import run

        run()
        from .evaluation.chat_turn7 import audit

        audit()
    elif stage == "long-latency":
        from .evaluation.depth_cost7 import run

        run()
    else:
        from .training.transfer7 import run

        run()
