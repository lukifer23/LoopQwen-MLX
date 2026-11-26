"""Compare the replicated recurrent finisher to an exposure-matched trained K1 finisher."""

import gc
from pathlib import Path

import mlx.core as mx

from .. import REVISION
from ..deployment import load_trained
from ..evaluation.answers import numeric
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import checkpoint_location, digest, read, save
from . import (
    bootstrap,
    core,
    deliberation,
    learned_exit,
    rationale_transfer,
    route_training,
)

NAME = "learned-exit-dense-01"
REG = Path("research/experiments") / (NAME + ".json")


def material(s):
    info = s["expert"]
    route_training.qualified_expert("arithmetic", info)
    original = Path(info["model"]).parent
    primary = read(original / "registration.json")
    pool = read(deliberation.TRAIN)
    drafts = unpack(Path(primary["drafts"]))
    extra = unpack(original / "extensions.json.gz")
    if digest(Path(primary["drafts"])) != primary["draft_sha256"] or any(
        r["id"] != d["id"] or r["id"] != e["id"]
        for r, d, e in zip(pool, drafts, extra, strict=True)
    ):
        raise ValueError("Training prefixes do not match")
    rows, expected = [], []
    for stage in (info["qualification"], info["confirmation"]):
        arc = Path(stage["directory"])
        rows.extend(read(Path("research/data") / (arc.name + ".json")))
        expected.extend(unpack(arc / "candidate.json.gz")["records"])
    if (
        len({r["id"] for r in rows}) != len(rows)
        or [r["prompt_ids"] for r in rows] != [r["prompt_ids"] for r in expected]
        or {r["id"] for r in rows}.intersection(r["id"] for r in pool)
    ):
        raise ValueError("Unpaired attribution cohort")
    return primary, pool, drafts, extra, rows, expected


def run():
    s = read(REG)
    primary, pool, drafts, extra, rows, expected = material(s)
    if (
        read("research/registry.json")["experiments"][NAME]["registration"]
        != digest(REG)
        or digest(Path(s["initial_dense"]) / "adapters.safetensors")
        != s["dense_sha256"]
    ):
        raise ValueError("Unregistered matched control")
    root = core.OUT / NAME
    root.mkdir(parents=True, exist_ok=False)
    b = core.Budget(NAME, s["limits"])
    result = {"registration": digest(REG), "data": fingerprint(rows), "passed": False}
    try:
        m, t, state = load_trained(s["initial_dense"])
        if m.span.passes != 1:
            raise ValueError("Control has recurrent computation")
        docs, traces = learned_exit.examples(primary, t, pool, drafts, extra)
        reference = read(
            Path(s["expert"]["qualification"]["directory"]) / "decision.json"
        )
        if fingerprint(docs) != reference["documents"]:
            raise ValueError("Dense training documents differ")
        m.checkpointing = True
        cp = Path("checkpoints/research") / NAME
        st = core.train(
            m,
            docs,
            dict(primary, id=NAME, depth=1, deployment=state["spec"]["deployment"]),
            cp,
            b,
            s["dense_sha256"],
        )
        gen = checkpoint_location(cp)
        bootstrap.release(gen, root / "model")
        w = digest(gen / "adapters.safetensors")
        learned_exit.fit_readout(primary, m, traces, core.atom(t, "</think>"), b, root)
        policy = {
            "version": 2,
            "revision": REVISION,
            "finisher_depth": 1,
            "switch_after": primary["switch_after"],
            "adapter_sha256": w,
            "readout_sha256": digest(root / "model/exit-readout.safetensors"),
        }
        save(root / "model/deliberation.json", policy)
        nt = sum(x["targets"] for x in st["history"])
        if nt != reference["targets"] or st["step"] != reference["updates"]:
            raise ValueError("Unmatched update exposure")
        result.update(
            dense_weights=w, documents=fingerprint(docs), targets=nt, updates=st["step"]
        )
        del m, t
        gc.collect()
        mx.clear_cache()
        by = {"dense": [], "recurrent": []}
        ret = {}
        half = len(rows) // 2
        for offset, order in (
            (0, ("dense", "recurrent")),
            (half, ("recurrent", "dense")),
        ):
            for arm in order:
                is_dense = arm == "dense"
                m, t, _ = load_trained(
                    root / "model" if is_dense else s["expert"]["model"]
                )
                for i in range(offset, offset + half):
                    row = rows[i]
                    val = deliberation.decode(
                        m,
                        t,
                        row["prompt_ids"],
                        primary["generation_cap"],
                        b.check,
                        thinking=True,
                    )
                    if not is_dense and val["token_ids"] != expected[i]["token_ids"]:
                        raise ValueError(
                            "Deployed recurrent output differs from the archived pair"
                        )
                    answer = numeric(val["visible"]) if val["complete"] else None
                    by[arm].append(
                        dict(
                            **val,
                            id=row["id"],
                            gold=row["gold"],
                            answer=answer,
                            correct=answer == row["gold"],
                        )
                    )
                    packed(
                        root / (arm + ".json.gz"),
                        {
                            "weights": w if is_dense else s["expert"]["weights"],
                            "records": by[arm],
                        },
                    )
                if offset:
                    ret[arm] = bootstrap.retention(m, 1 if is_dense else 2, b)
                del m, t
                gc.collect()
                mx.clear_cache()
        scores = rationale_transfer.summarize(
            rows, by["dense"], by["recurrent"], cap=primary["generation_cap"]
        )
        for obj in scores.values():
            obj["dense"] = obj.pop("base")
            obj["recurrent"] = obj.pop("candidate")
        a = scores["all"]
        passed = (
            a["wins"] - a["losses"] >= s["net_floor"]
            and all(
                scores[k]["wins"] >= scores[k]["losses"] for k in primary["sources"]
            )
            and a["recurrent"]["seconds"] <= s["max_time_ratio"] * a["dense"]["seconds"]
            and ret["recurrent"]["nll"] <= ret["dense"]["nll"] + 0.03
        )
        result.update(status="completed", passed=passed, scores=scores, retention=ret)
    except (TimeoutError, InterruptedError, MemoryError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result["status"], result["passed"], flush=True)


if __name__ == "__main__":
    run()
