"""Two bounded verified generations with a static gold-data training control."""

import gc
from pathlib import Path

import mlx.core as mx
import numpy as np
from . import core, bootstrap
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save, checkpoint_location
from ..training.exposure import ARM
from ..training.mechanism import requests
from ..training.transfer7 import prompt
from ..training.verified_trace import verdict


def render(row):
    return (
        prompt(row, "document").rsplit("\n", 1)[0]
        + " Follow exactly three references. Write the three visited document numbers in order, separated by commas.\nVisited documents:"
    )


def gold_text(row):
    return " " + ", ".join(map(str, row["states"][:3]))


def candidates(m, t, pool, s, b, folder, weight):
    path = folder / "candidates.json.gz"
    identity = fingerprint([weight, s, pool, digest(Path(__file__)), core.assets()])
    saved = unpack(path) if path.exists() else dict(identity=identity, records=[])
    if saved["identity"] != identity:
        raise ValueError("Incompatible rollout checkpoint")
    rows = saved["records"]
    picked = {}
    for i, r in enumerate(pool):
        ids = t.encode(render(r), add_special_tokens=False)
        for sample in range(s["samples_per_map"]):
            j = i * s["samples_per_map"] + sample
            seed = int(fingerprint([s["seed"], r["map_id"], sample])[:8], 16)
            if j == len(rows):
                gen = core.generate(
                    m,
                    t,
                    ids,
                    s["generation_cap"],
                    b.check,
                    line=True,
                    temperature=s["temperature"],
                    seed=seed,
                )
                rows.append(
                    dict(
                        map_id=r["map_id"],
                        map=r["map"],
                        start=r["start"],
                        order=r["order"],
                        seed=seed,
                        **gen,
                    )
                )
                packed(path, saved)
            rec = rows[j]
            if (
                rec["prompt_ids"] != ids
                or rec["map_id"] != r["map_id"]
                or seed != rec["seed"]
            ):
                raise ValueError("Resumed candidate schedule changed")
            score = verdict(r["map"], r["start"], rec["visible"].strip(), rec["stop"])
            rec["score"] = score
            if score["verified"] and rec["complete"]:
                picked.setdefault(r["map_id"], rec)
    packed(path, saved)
    return picked, dict(
        maps=len(pool),
        attempts=len(rows),
        accepted=sum(r["score"]["verified"] for r in rows),
        distinct=len(picked),
        tokens=sum(len(r["token_ids"]) for r in rows),
        rollout_seconds=sum(r["seconds"] for r in rows),
    )


def evaluate(m, t, w, rows, s, b, out):
    return bootstrap.traces(m, t, rows, s, b, out, w, render=render)


def run():
    s = read("research/experiments/verified-generations-01.json")
    name = s["id"]
    out = core.OUT / name
    if core.archived_result(out, s):
        return
    b = core.Budget(name, s["limits"])
    try:
        groups = core.reserve(
            name,
            s["seed"],
            {
                **{f"pool{j+1}": s["candidates_per_round"] for j in range(s["rounds"])},
                "dev": s["dev_maps"],
            },
            functional=True,
        )
        parent = Path(s["parent_path"])
        initial = s["parent_sha256"]
        if (
            digest(parent / "adapters.safetensors") != initial
            or not read(parent.parent / "decision.json")["promoted"]
        ):
            raise ValueError("Parent did not qualify")
        key = fingerprint(
            [
                s,
                groups,
                core.assets(),
                digest(Path(__file__)),
                digest(Path(core.__file__)),
                digest(Path(bootstrap.__file__)),
            ]
        )
        cursor = out / "progress.json"
        st = (
            read(cursor)
            if cursor.exists()
            else dict(identity=key, parent=initial, stages={}, status="running")
        )
        if st["identity"] != key:
            raise ValueError("Incompatible self-training lineage")
        if "parent_eval" not in st:
            m, t = core.model(parent, 3)
            st["parent_eval"] = evaluate(
                m, t, initial, groups["dev"], s, b, out / "parent-eval.json.gz"
            )
            st["categorical"] = bootstrap.categorical(m, t, b)
            save(cursor, st)
            del m, t
            gc.collect()
            mx.clear_cache()
        previous = {arm: parent for arm in ("self", "gold")}
        for j in range(1, s["rounds"] + 1):
            stage = out / f"round{j}"
            pool = groups[f"pool{j}"]
            current = st["stages"].setdefault(str(j), {})
            if not (stage / "selection.json").exists():
                source = previous["self"]
                w = digest(source / "adapters.safetensors")
                m, t = core.model(source, 3)
                picked, summary = candidates(m, t, pool, s, b, stage, w)
                chosen = list(picked)[: s["select"]]
                selection = dict(
                    parent=w,
                    self_ids=chosen,
                    gold_ids=[r["map_id"] for r in pool[: s["select"]]],
                    responses={i: picked[i]["token_ids"] for i in chosen},
                    summary=summary,
                    candidates_sha256=digest(stage / "candidates.json.gz"),
                )
                save(stage / "selection.json", selection)
                del m, t
                gc.collect()
                mx.clear_cache()
            selected = read(stage / "selection.json")
            if selected["parent"] != digest(
                previous["self"] / "adapters.safetensors"
            ) or selected["candidates_sha256"] != digest(stage / "candidates.json.gz"):
                raise ValueError("Rollout or teacher changed")
            selection_id = digest(stage / "selection.json")
            if current.get("selection", selection_id) != selection_id:
                raise ValueError("Curriculum changed")
            current.update(selection=selection_id, yield_summary=selected["summary"])
            save(cursor, st)
            if len(selected["self_ids"]) < s["select"]:
                st["status"] = "insufficient verified training maps"
                break
            if j == 1 and st["parent_eval"]["verified"] / s["dev_maps"] >= 0.95:
                st["status"] = "parent at registered development ceiling"
                break
            by_id = {r["map_id"]: r for r in pool}
            for arm in ("gold", "self"):
                artifact = stage / arm
                source = previous[arm]
                weight = digest(source / "adapters.safetensors")
                if arm in current:
                    if current[arm]["parent"] != weight or current[arm][
                        "weights"
                    ] != digest(artifact / "adapters.safetensors"):
                        raise ValueError("Archived generation differs")
                else:
                    rows = [by_id[i] for i in selected[arm + "_ids"]]
                    m, t = core.model(source, 3)
                    docs = []
                    equal = 0
                    for r in rows:
                        gold = core.doc(t, render(r), gold_text(r))
                        ids = gold["ids"][: gold["prefix"]]
                        ans = (
                            gold["ids"][gold["prefix"] :]
                            if arm == "gold"
                            else selected["responses"][r["map_id"]]
                        )
                        equal += ans == gold["ids"][gold["prefix"] :]
                        docs.append(dict(ids=ids + ans, prefix=len(ids)))
                    states = [
                        dict(ids=d["ids"], states=d["row"]["states"][:3])
                        for d in requests(rows, 3, ARM, t, {})
                    ]
                    cp = Path("checkpoints/research") / name / f"round{j}" / arm
                    run_id = fingerprint([key, weight, selection_id, arm, j])
                    tr = core.train(
                        m,
                        docs * s["repeats"],
                        s,
                        cp,
                        b,
                        run_id,
                        state_docs=states * s["repeats"],
                    )
                    gen = checkpoint_location(cp)
                    w = digest(gen / "adapters.safetensors")
                    if not artifact.exists():
                        bootstrap.release(gen, artifact)
                    if digest(artifact / "adapters.safetensors") != w:
                        raise ValueError("Generation differs")
                    summary = evaluate(
                        m, t, w, groups["dev"], s, b, stage / (arm + "-eval.json.gz")
                    )
                    current[arm] = dict(
                        parent=weight,
                        weights=w,
                        gold_identical=equal,
                        score=summary,
                        categorical=bootstrap.categorical(m, t, b),
                        updates=tr["step"],
                        update_seconds=sum(r["seconds"] for r in tr["history"]),
                        vocab_targets=sum(r["targets"] for r in tr["history"]),
                        state_labels=sum(r["state_labels"] for r in tr["history"]),
                    )
                    save(cursor, st)
                    del m, t
                    gc.collect()
                    mx.clear_cache()
                previous[arm] = artifact
            a = unpack(stage / "self-eval.json.gz")["records"]
            g = unpack(stage / "gold-eval.json.gz")["records"]
            if [r["prompt_ids"] for r in a] != [r["prompt_ids"] for r in g]:
                raise ValueError("Unpaired evaluation")
            delta = np.array(
                [
                    int(x["score"]["verified"]) - int(y["score"]["verified"])
                    for x, y in zip(a, g, strict=True)
                ]
            )
            draws = np.random.default_rng(s["seed"]).integers(
                0, len(delta), (5000, len(delta))
            )
            current["paired"] = dict(
                wins=int((delta > 0).sum()),
                losses=int((delta < 0).sum()),
                delta=float(delta.mean()),
                ci=np.quantile(delta[draws].mean(1), [0.025, 0.975]).tolist(),
            )
            save(cursor, st)
            if any(
                (st["categorical"][-1] - current[arm]["categorical"][-1]) / 96 > 0.02
                for arm in previous
            ):
                st["status"] = "categorical retention stop"
                break
        else:
            st["status"] = "completed two generations"
        st["passed"] = (
            st["status"] == "completed two generations"
            and current["paired"]["delta"] >= 0.03
        )
        save(out / "decision.json", st)
        print(st["status"], st["passed"], flush=True)
    finally:
        b.close()


if __name__ == "__main__":
    run()
