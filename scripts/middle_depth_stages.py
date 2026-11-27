"""Dispatch enrolled follow-on fits and measurements using verified parent identities."""

import argparse
from pathlib import Path
from statistics import fmean

from loopqwen.io import read, save, digest
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_wave import dispatch, fit, generate

ARMS = ("ordinary", "loop")
RUN = Path("checkpoints/middle-depth")


def forks():
    paths = {arm: fit(arm, 576) for arm in ARMS}
    for condition, stage in (("simple", "simple"), ("actual", "corrected")):
        children = {}
        for arm in ARMS if condition == "simple" else reversed(ARMS):
            dispatch("correction", "fit", "--arm", arm, "--condition", condition)
            children[arm] = RUN / "correction" / condition / arm / "step-0032"
        generate(stage, list(ARMS), [128], children)


def select_history():
    scores = {}
    for stage in ("terminal", "simple", "corrected"):
        rows = [
            r
            for r in read(ROOT / "review" / stage / "summary.json")
            if r["arm"] == "loop" and r["cap"] == 128
        ]
        assert len(rows) == 3 and all(r["episodes"] == 9 for r in rows)
        scores[stage] = {r["family"]: r["success"] for r in rows}
    chosen = all(
        scores["corrected"]["tools"] > scores[k]["tools"]
        and sum(scores["corrected"].values()) > sum(scores[k].values())
        and all(scores["corrected"][f] >= scores[k][f] for f in ("reasoning", "chat"))
        for k in ("simple", "terminal")
    )
    models = {
        arm: (
            RUN / "correction" / "actual" / arm / "step-0032"
            if chosen
            else RUN / "4619" / arm / "step-0576"
        )
        for arm in ARMS
    }
    save(
        ROOT / "history-decision.json",
        dict(
            scores=scores,
            passed=chosen,
            stage="corrected" if chosen else "terminal",
            models={
                k: dict(path=str(p), weights=digest(p / "adapters.safetensors"))
                for k, p in models.items()
            },
        ),
    )
    if chosen:
        generate("corrected", list(ARMS), [512], models)


def selected():
    r = read(ROOT / "history-decision.json")
    paths = {k: Path(v["path"]) for k, v in r["models"].items()}
    for k, p in paths.items():
        assert digest(p / "adapters.safetensors") == r["models"][k]["weights"]
    return paths


def documents(
    stage, policies, arms, caps, followups=False, worlds=("original", "edited")
):
    for cap in caps:
        for i, book in enumerate(("anne", "book")):
            for j, world in enumerate(worlds):
                for arm in arms if (i + j) % 2 == 0 else reversed(arms):
                    ref = ROOT / "documents" / stage / book / arm / str(cap)
                    expected = [world + "-first"]
                    if followups and book == "anne":
                        expected += [
                            world + "-" + k for k in ("own", "shared", "unanswerable")
                        ]
                    if all((ref / (k + ".json.gz")).exists() for k in expected):
                        continue
                    args = [
                        "run",
                        "--book",
                        book,
                        "--world",
                        world,
                        "--arm",
                        arm,
                        "--stage",
                        stage,
                        "--tokens",
                        cap,
                        "--cache-mode",
                        "resident",
                    ]
                    if arm in policies:
                        args += ["--policy", policies[arm]]
                    if followups and book == "anne":
                        args += ["--followups"]
                    dispatch("documents", *args)


def short_parent_books():
    parents = selected()
    # Treat the unchanged Base trajectories as one immutable set for every
    # parent comparison, while alternating first model by book and edition.
    documents("parent", parents, ["loop", "base", "ordinary"], [128, 512])


def long_fit():
    parents = selected()
    for arm in ARMS:
        dispatch("long_fit", "fit", "--arm", arm, "--policy", parents[arm])


def long_children():
    models = {arm: RUN / "long" / arm for arm in ARMS}
    for arm, p in models.items():
        assert (
            digest(p / "adapters.safetensors")
            == read(ROOT / f"long-fit-{arm}.json")["weights"]
        )
    generate("deployed", list(ARMS), [128, 512], models)
    documents("child", models, ["ordinary", "loop"], [128, 512])


def select_long():
    policy = read(ROOT / "history-decision.json")

    def books(stage, arm):
        scores = {}
        for r in read(ROOT / "document-review" / stage / "scored.json"):
            if r["arm"] == arm and r["cap"] == 512 and r["kind"] == "first":
                scores.setdefault(r["source"], []).append(r["grade"]["correct"])
        assert set(scores) == {"anne", "book"} and all(
            len(v) == 2 for v in scores.values()
        )
        return {k: sum(v) for k, v in scores.items()}

    def short(stage, arm):
        rows = [
            r
            for r in read(ROOT / "review" / stage / "summary.json")
            if r["arm"] == arm and r["cap"] == 512
        ]
        assert len(rows) == 3
        return fmean(r["success"] / r["episodes"] for r in rows)

    evidence = {}
    for arm in ARMS:
        evidence[arm] = dict(
            parent=books("parent", arm),
            child=books("child", arm),
            parent_short=short(policy["stage"], arm),
            child_short=short("deployed", arm),
        )
    x = evidence["loop"]
    keep = (
        sum(x["child"].values()) > sum(x["parent"].values())
        and all(x["child"][k] >= x["parent"][k] for k in x["child"])
        and x["child_short"] >= x["parent_short"]
    )
    paths = {a: RUN / "long" / a for a in ARMS} if keep else selected()
    save(
        ROOT / "long-decision.json",
        dict(
            evidence=evidence,
            passed=keep,
            short_stage="deployed" if keep else policy["stage"],
            document_stage="child" if keep else "parent",
            models={
                k: dict(path=str(p), weights=digest(p / "adapters.safetensors"))
                for k, p in paths.items()
            },
        ),
    )


if __name__ == "__main__":
    ops = {
        "forks": forks,
        "history": select_history,
        "parent-books": short_parent_books,
        "long-fit": long_fit,
        "children": long_children,
        "select": select_long,
    }
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=ops)
    ops[p.parse_args().stage]()
