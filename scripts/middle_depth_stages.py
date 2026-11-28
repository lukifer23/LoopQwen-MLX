"""Dispatch enrolled follow-on fits and measurements using verified parent identities."""

import argparse
from pathlib import Path
from statistics import fmean

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import BUDGET, ROOT
from scripts.middle_depth_recover import fit_directory
from scripts.middle_depth_wave import NativeResourceStop, dispatch, fit, generate

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
            else fit_directory(arm) / "step-0576"
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


def document_files(stage, arm, cap, book, world, followups):
    folder = ROOT / "documents" / stage / book / arm / str(cap)
    kinds = ["first"]
    if followups and book == "anne" and world != "irrelevant":
        kinds.extend(("own", "shared", "unanswerable"))
    return {kind: folder / f"{world}-{kind}.json.gz" for kind in kinds}


def documents(
    stage, policies, arms, caps, followups=False, worlds=("original", "edited")
):
    # One unavailable operating point should not discard the other model's
    # registered trials or the independent book. Keep the original order and
    # require complete paired data before advancing any selection decision.
    coverage = {}
    manifest = ROOT / "document-panels" / (stage + ".json")
    for i, book in enumerate(("anne", "book")):
        for cap in caps:
            for j, world in enumerate(worlds):
                for arm in arms if (i + j) % 2 == 0 else reversed(arms):
                    key = f"{book}/{arm}/{cap}/{world}"
                    expected = document_files(
                        stage, arm, cap, book, world, followups
                    ).values()
                    spent = sum(
                        r["seconds"]
                        for f in (ROOT / "charges").glob("*.json")
                        if (r := read(f))["stage"] == "long"
                    )
                    if spent >= BUDGET["long"] and not all(
                        p.is_file() for p in expected
                    ):
                        item = {"status": "budget_exhausted", "spent": spent}
                    else:
                        try:
                            document_request(
                                stage, policies, arm, cap, book, world, followups
                            )
                        except NativeResourceStop as error:
                            item = {
                                "status": "resource_stop",
                                "charge": str(error.charge.relative_to(ROOT)),
                                "charge_sha": digest(error.charge),
                            }
                        else:
                            item = {"status": "complete"}
                    item["outputs"] = {
                        str(p.relative_to(ROOT)): digest(p)
                        for p in expected
                        if p.is_file()
                    }
                    if item["status"] == "complete" and len(item["outputs"]) != len(
                        expected
                    ):
                        raise ValueError(f"Missing long-document output for {key}")
                    coverage[key] = item
                    save(manifest, coverage)
    incomplete = [k for k, v in coverage.items() if v["status"] != "complete"]
    if incomplete:
        raise RuntimeError(f"Document panel {stage} is incomplete: {incomplete}")


def document_request(stage, policies, arm, cap, book, world, followups=False):
    from scripts.middle_depth_documents import EXTRA, FROZEN, STABLE

    files = document_files(stage, arm, cap, book, world, followups)
    extra = "own" in files
    if all(p.is_file() for p in files.values()):
        spec = read(FROZEN)[book]
        weights = (
            read(policies[arm] / "registration.json")["weights"]
            if arm in policies
            else "base-dc7cdfe"
        )
        first_history = None
        for kind, path in files.items():
            raw = unpack(path)
            if (
                raw["weights"] != weights
                or raw["frozen"] != digest(FROZEN)
                or raw["source"] != spec["source"]
            ):
                raise ValueError(f"Wrong checkpoint or source in {path}")
            turns = raw["trace"]["turns"]
            if not 1 <= len(turns) <= 3 or any(
                len(t["token_ids"]) > cap for t in turns
            ):
                raise ValueError(f"Saved generation policy differs in {path}")
            if kind == "own":
                if (
                    not first_history
                    or raw["history"][: len(first_history)] != first_history
                ):
                    raise ValueError(f"Saved own-history follow-up changed in {path}")
                continue
            if kind == "first":
                first_history = raw["history"]
                opening = (
                    read(STABLE)["rows"][book]
                    if world == "irrelevant"
                    else spec["worlds"][world]
                )
            else:
                opening = read(EXTRA)["rows"][world + "/" + kind]
            ids = raw["trace"]["turns"][0]["prompt_ids"]
            if len(ids) != opening["length"] or fingerprint(ids) != opening["input"]:
                raise ValueError(f"Saved physical opening changed in {path}")
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


def final_books(
    arms=("loop", "base", "ordinary"),
    decision_file="long-decision.json",
    enrollment_file="complete-books.json",
    primary_only=False,
):
    decision = read(ROOT / decision_file)
    models = {k: Path(v["path"]) for k, v in decision["models"].items()}
    for arm, path in models.items():
        if digest(path / "adapters.safetensors") != decision["models"][arm]["weights"]:
            raise ValueError("A selected document checkpoint has changed")
    stages = {a: decision["document_stage"] for a in models} | {"base": "parent"}
    # Finish primary matched turns and stability checks before extra headroom.
    for i, book in enumerate(("anne", "book")):
        for cap in (128, 512):
            for j, world in enumerate(("original", "edited", "irrelevant")):
                for arm in arms if (i + j) % 2 == 0 else reversed(arms):
                    document_request(stages[arm], models, arm, cap, book, world, True)
    if primary_only:
        return
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
    log = ROOT / "headroom" / enrollment_file
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
    attempts = {}
    for arm in ARMS:
        try:
            dispatch("long_fit", "fit", "--arm", arm, "--policy", parents[arm])
        except NativeResourceStop as error:
            attempts[arm] = {
                "status": "resource_stop",
                "charge": str(error.charge.relative_to(ROOT)),
                "charge_sha": digest(error.charge),
            }
        else:
            path = ROOT / f"long-fit-{arm}.json"
            attempts[arm] = {
                "status": "complete",
                "receipt": path.name,
                "sha": digest(path),
            }
        save(ROOT / "long-fit-panel.json", attempts)
    if any(row["status"] != "complete" for row in attempts.values()):
        raise RuntimeError(
            "Full-document fitting stopped before both siblings finished; see long-fit-panel.json"
        )


def verify_long_fit(arm, parent, model):
    from math import isfinite

    import numpy as np
    from safetensors import safe_open

    from scripts.middle_depth_long_fit import SPEC

    trained = read(ROOT / f"long-fit-{arm}.json")
    spec = read(SPEC)
    weights = "adapters.safetensors"
    policy = read(model / "registration.json")
    if (
        trained["parent"] != digest(parent / weights)
        or trained["parent"] != policy["short_parent"]
        or trained["weights"] != digest(model / weights)
        or trained["weights"] != policy["weights"]
        or policy["plan"] != digest(SPEC)
    ):
        raise ValueError(f"Invalid full-document parent or deployed weights for {arm}")
    log = trained["log"]
    if [r["step"] for r in log] != list(range(1, spec["steps"] + 1)) or any(
        not isfinite(r["gradient"])
        or r["gradient"] <= 0
        or len(r["loss"]) != len(spec["rows"])
        or not all(map(isfinite, r["loss"]))
        for r in log
    ):
        raise ValueError(f"Incomplete or nonfinite full-document updates for {arm}")
    with (
        safe_open(str(parent / weights), framework="np") as a,
        safe_open(str(model / weights), framework="np") as b,
    ):
        names = a.keys()
        if set(names) != set(b.keys()):
            raise ValueError(f"Adapter inventory changed for {arm}")
        changed = []
        for name in names:
            before, after = a.get_tensor(name), b.get_tensor(name)
            if (
                before.shape != after.shape
                or before.dtype != after.dtype
                or not np.isfinite(after).all()
            ):
                raise ValueError(f"Invalid full-document tensor {name}")
            if before.tobytes() != after.tobytes():
                changed.append(name)
    allowed = {f"text.model.layers.23.mlp.down_proj.lora_{part}" for part in ("a", "b")}
    if (
        not changed
        or set(changed) - allowed
        or set(changed) != set(trained["changed_tensors"])
    ):
        raise ValueError(
            f"Unexpected full-document parameter changes in {arm}: {changed}"
        )
    return {
        "parent": trained["parent"],
        "weights": trained["weights"],
        "steps": len(log),
        "changed": changed,
    }


def long_children():
    models = {arm: RUN / "long" / arm for arm in ARMS}
    parents = selected()
    save(
        ROOT / "long-fit-integrity.json",
        {arm: verify_long_fit(arm, parents[arm], p) for arm, p in models.items()},
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
    # Report the best eligible ordinary sibling as well as the recipe-matched
    # one. All eligible source groups have the registered equal-family weights.
    control = evidence["ordinary"]
    ordinary_macro = {
        variant: (3 * control[variant + "_short"] + sum(control[variant].values()) / 4)
        / 4
        for variant in ("parent", "child")
    }
    best_ordinary = (
        "child" if ordinary_macro["child"] > ordinary_macro["parent"] else "parent"
    )
    save(
        ROOT / "long-decision.json",
        {
            "evidence": evidence,
            "passed": keep,
            "short_stage": "deployed" if keep else policy["stage"],
            "document_stage": "child" if keep else "parent",
            "ordinary_development_macro": ordinary_macro,
            "strongest_ordinary": {
                "long_stage": best_ordinary,
                "short_stage": "deployed"
                if best_ordinary == "child"
                else policy["stage"],
                "path": str(
                    RUN / "long" / "ordinary"
                    if best_ordinary == "child"
                    else selected()["ordinary"]
                ),
            },
            "models": {
                k: {"path": str(p), "weights": digest(p / "adapters.safetensors")}
                for k, p in paths.items()
            },
        },
    )


def final_decisions():
    from argparse import Namespace

    from scripts.middle_depth_decision import measure

    chosen = read(ROOT / "long-decision.json")
    for cap in (128, 512):
        for control, pair in (
            ("matched", (chosen["short_stage"], chosen["document_stage"])),
            (
                "strongest",
                (
                    chosen["strongest_ordinary"]["short_stage"],
                    chosen["strongest_ordinary"]["long_stage"],
                ),
            ),
        ):
            measure(
                Namespace(
                    name=f"chosen-{control}-{cap}",
                    cap=cap,
                    candidate_short=chosen["short_stage"],
                    candidate_long=chosen["document_stage"],
                    ordinary_short=pair[0],
                    ordinary_long=pair[1],
                    base_short="terminal",
                    base_long="parent",
                )
            )


if __name__ == "__main__":
    ops = {
        "forks": forks,
        "early-book": early_book,
        "history": select_history,
        "parent-books": short_parent_books,
        "final-books": final_books,
        "decisions": final_decisions,
        "long-fit": long_fit,
        "children": long_children,
        "select": select_long,
    }
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=ops)
    ops[p.parse_args().stage]()
