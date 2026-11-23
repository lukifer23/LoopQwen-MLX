"""Apply the registered 100K promotion rule and assess the 250K pair on development only."""

import json

from pathlib import Path
from ..io import read, save

OUT = Path("outputs/phase3")


def check():
    rows = read(OUT / "main.json")["rows"]
    selected = read(OUT / "selection.json")["selected"]
    by = {(r["arm"], r["mode"]): r for r in rows}
    gated = by[selected, "trained"]
    dense = by["dense", "trained"]
    skip = by[selected, "bypass"]
    if not (
        gated["correct"] > skip["correct"]
        and gated["correct"] >= dense["correct"]
        and gated["retention"] <= dense["retention"] + 0.05
    ):
        raise ValueError("Registered development promotion rule did not pass")
    return selected


def evaluate():
    from ..data import v2, v3, retention3
    from .phase3 import REGISTER, measure, compare

    chosen = check()
    study = read(REGISTER)
    v3.build()
    retention3.build()
    general = [
        json.loads(line) for line in (retention3.ROOT / "validation.jsonl").open()
    ]
    summary = {}
    for name, samples, prose in (
        (
            "v2_validation",
            v2.rows("validation", "trace")[:60],
            v2.rows("language_validation", "language"),
        ),
        ("development", v2.rows("development", "trace", v3.ROOT), general),
        ("development_ood", v2.rows("development_ood", "trace", v3.ROOT), general),
    ):
        registration = {
            **study,
            "stage": "250K development",
            "v3": read(v3.ROOT / "manifest.json"),
            "retention": read(retention3.ROOT / "manifest.json"),
            "generation": {**study["generation"], "split": name, "count": len(samples)},
        }
        scores = {}
        for arm, mode in (
            ("dense", "trained"),
            (chosen, "trained"),
            (chosen, "bypass"),
        ):
            scores[arm + "/" + mode] = measure(
                arm, 250000, mode, samples, prose, registration
            )
        if (
            len(
                {(r["order"], r["input_tokens"], r["targets"]) for r in scores.values()}
            )
            != 1
        ):
            raise AssertionError("Unmatched promoted cohort")
        g = scores[chosen + "/trained"]
        summary[name] = dict(
            metrics={
                k: dict(
                    n=len(samples),
                    correct=r["certificate"]["correct"],
                    verified=r["certificate"]["verified"],
                    retention=r["retention"]["nll"],
                    final_nll=r["likelihood"]["nll"]["final"],
                    trace_nll=r["likelihood"]["trace_nll"],
                    alpha=r["alpha"],
                    truncated=r["certificate"]["truncated"],
                    seconds=r["generations"]["seconds"],
                )
                for k, r in scores.items()
            },
            against_dense=compare(scores["dense/trained"], g),
            against_bypass=compare(scores[chosen + "/bypass"], g),
        )
        save(OUT / "promotion-progress.json", summary)
    dev = summary["development"]["metrics"]
    ood = summary["development_ood"]["metrics"]
    g, d, b = dev[chosen + "/trained"], dev["dense/trained"], dev[chosen + "/bypass"]
    eligible = (
        g["correct"] > max(d["correct"], b["correct"])
        and ood[chosen + "/trained"]["correct"] >= ood["dense/trained"]["correct"]
        and g["retention"] <= d["retention"] + 0.05
    )
    save(
        OUT / "promotion.json",
        dict(selected=chosen, confirm_eligible=eligible, splits=summary),
    )


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["check"]:
        print(check())
    elif sys.argv[1:] == ["evaluate"]:
        evaluate()
    else:
        raise SystemExit("Expected check or evaluate")
