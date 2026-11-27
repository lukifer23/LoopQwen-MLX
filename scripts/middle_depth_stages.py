"""Dispatch enrolled follow-on fits and measurements using verified parent identities."""

import argparse
from pathlib import Path
from statistics import fmean

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_wave import dispatch, fit, generate

ARMS = ("ordinary", "loop")
RUN = Path("checkpoints/middle-depth")


def forks():
    for arm in ARMS:
        fit(arm, 576)
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
        {
            "scores": scores,
            "passed": chosen,
            "stage": "corrected" if chosen else "terminal",
            "models": {
                k: {"path": str(p), "weights": digest(p / "adapters.safetensors")}
                for k, p in models.items()
            },
        },
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
    for i, book in enumerate(("anne", "book")):
        for cap in caps:
            for j, world in enumerate(worlds):
                for arm in arms if (i + j) % 2 == 0 else reversed(arms):
                    document_request(stage, policies, arm, cap, book, world, followups)


def document_request(stage, policies, arm, cap, book, world, followups=False):
    ref = ROOT / "documents" / stage / book / arm / str(cap)
    expected = ["first"]
    extra = followups and book == "anne" and world != "irrelevant"
    if extra:
        expected.extend(("own", "shared", "unanswerable"))
    if all((ref / f"{world}-{k}.json.gz").is_file() for k in expected):
        return
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
    if extra:
        args += ["--followups"]
    dispatch("documents", *args)


def final_books():
    decision = read(ROOT / "long-decision.json")
    models = {k: Path(v["path"]) for k, v in decision["models"].items()}
    for arm, path in models.items():
        if digest(path / "adapters.safetensors") != decision["models"][arm]["weights"]:
            raise ValueError("A selected document checkpoint has changed")
    arms = ("loop", "base", "ordinary")
    stages = {a: decision["document_stage"] for a in models} | {"base": "parent"}
    # Finish primary matched turns and stability checks before extra headroom.
    for i, book in enumerate(("anne", "book")):
        for cap in (128, 512):
            for j, world in enumerate(("original", "edited", "irrelevant")):
                for arm in arms if (i + j) % 2 == 0 else reversed(arms):
                    document_request(stages[arm], models, arm, cap, book, world, True)
    # Freeze the paired extended-output enrollment using only earlier stopping
    # status and the mandatory originals. No answer text or scores enter selection.
    reasons = {}
    previous = {}
    for book in ("anne", "book"):
        for world in ("original", "edited", "irrelevant"):
            key = book + "/" + world
            stops = []
            for arm in arms:
                folder = ROOT / "documents" / stages[arm] / book / arm / "512"
                kinds = ["first"]
                if book == "anne" and world != "irrelevant":
                    kinds += ["own", "shared", "unanswerable"]
                for kind in kinds:
                    path = folder / (world + "-" + kind + ".json.gz")
                    previous[str(path.relative_to(ROOT))] = digest(path)
                    if unpack(path)["trace"]["status"] == "token_limit":
                        stops.append([arm, kind])
            if world == "original" or stops:
                reasons[key] = stops
    enrollment = {
        "sources": previous,
        "reasons": reasons,
        "models": decision["models"],
        "registration": digest(
            Path("research/diagnostics/middle-depth/output-headroom.md")
        ),
    }
    log = ROOT / "headroom" / "complete-books.json"
    if log.exists() and read(log) != enrollment:
        raise ValueError("Full-document headroom enrollment changed")
    save(log, enrollment)
    for index, key in enumerate(reasons):
        book, world = key.split("/")
        for arm in arms if index % 2 == 0 else reversed(arms):
            document_request(stages[arm], models, arm, 4096, book, world, True)


def early_book():
    paths = {arm: RUN / "4619" / arm / "step-0144" for arm in ARMS}
    for world, arms in (("original", ARMS), ("edited", reversed(ARMS))):
        for arm in arms:
            document_request("epoch-one", paths, arm, 512, "anne", world)


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
            if (
                r["arm"] == arm
                and r["cap"] == 512
                and r["kind"] == "first"
                and r["world"] in ("original", "edited")
            ):
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
        evidence[arm] = {
            "parent": books("parent", arm),
            "child": books("child", arm),
            "parent_short": short(policy["stage"], arm),
            "child_short": short("deployed", arm),
        }
    x = evidence["loop"]
    keep = (
        sum(x["child"].values()) > sum(x["parent"].values())
        and all(x["child"][k] >= x["parent"][k] for k in x["child"])
        and x["child_short"] >= x["parent_short"]
    )
    paths = {a: RUN / "long" / a for a in ARMS} if keep else selected()
    save(
        ROOT / "long-decision.json",
        {
            "evidence": evidence,
            "passed": keep,
            "short_stage": "deployed" if keep else policy["stage"],
            "document_stage": "child" if keep else "parent",
            "models": {
                k: {"path": str(p), "weights": digest(p / "adapters.safetensors")}
                for k, p in paths.items()
            },
        },
    )


if __name__ == "__main__":
    ops = {
        "forks": forks,
        "early-book": early_book,
        "history": select_history,
        "parent-books": short_parent_books,
        "final-books": final_books,
        "long-fit": long_fit,
        "children": long_children,
        "select": select_long,
    }
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=ops)
    ops[p.parse_args().stage]()
