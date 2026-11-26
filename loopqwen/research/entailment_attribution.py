"""Repeat natural inference with an equally trained ordinary-depth decoder."""

import gc
from pathlib import Path

import mlx.core as mx

from ..deployment import load_trained
from ..evaluation.mechanism import unpack
from ..io import checkpoint_location, digest, read, save
from . import bootstrap, core, entailment, inference_data, route_training

NAME = "entailment-dense-01"
REG = Path("research/experiments") / (NAME + ".json")


def run():
    s = read(REG)
    route_training.qualified_expert("entailment", s["expert"])
    arc = Path(s["expert"]["qualification"]["directory"])
    first = read(arc / "registration.json")
    if (
        read("research/registry.json")["experiments"][NAME]["registration"]
        != digest(REG)
        or digest(Path(s["dense_control"]) / "adapters.safetensors")
        != s["dense_sha256"]
    ):
        raise ValueError("Attribution control or registration changed")
    panels = [
        Path("research/data") / (Path(s["expert"][name]["directory"]).name + ".json")
        for name in ("qualification", "confirmation")
    ]
    first_panel, confirm = map(read, panels)
    rows = first_panel + confirm
    docs, manifest = inference_data.training(first, first_panel)
    if manifest != read(entailment.TRAIN):
        raise ValueError("Different training documents")
    out = core.OUT / NAME
    out.mkdir(parents=True, exist_ok=False)
    budget = core.Budget(NAME, s["limits"])
    result = {
        "passed": False,
        "registration": digest(REG),
        "training": digest(entailment.TRAIN),
        "panels": [digest(p) for p in panels],
    }
    try:
        m, _, state = load_trained(s["dense_control"])
        if m.span.passes != 1:
            raise ValueError("Expected the original dense training twin")
        m.checkpointing = True
        cp = Path("checkpoints/research") / NAME
        trained = core.train(
            m,
            docs,
            dict(first, id=NAME, depth=1, deployment=state["spec"]["deployment"]),
            cp,
            budget,
            s["dense_sha256"],
        )
        gen = checkpoint_location(cp)
        bootstrap.release(gen, out / "model")
        w = digest(gen / "adapters.safetensors")
        nt = sum(h["targets"] for h in trained["history"])
        if (
            nt != manifest["response_tokens"]
            or nt != read(arc / "decision.json")["targets"]
        ):
            raise ValueError("Dense token exposure changed")
        result.update(weights=w, targets=nt, updates=trained["step"])
        del m
        gc.collect()
        mx.clear_cache()
        scores, ret = entailment.evaluate(
            first,
            rows,
            s["expert"]["model"],
            s["expert"]["weights"],
            budget,
            out,
            dense_control=(out / "model", w),
        )
        refs = [
            r
            for step in ("qualification", "confirmation")
            for r in unpack(Path(s["expert"][step]["directory"]) / "candidate.json.gz")[
                "records"
            ]
        ]
        runs = unpack(out / "recurrent.json.gz")["records"]
        if any(
            any(a[k] != b[k] for k in ("prompt_ids", "token_ids", "probe_logits"))
            for a, b in zip(runs, refs, strict=True)
        ):
            raise ValueError("Frozen recurrent output changed")
        a = scores["all"]
        c, d = a["recurrent"], a["dense"]
        passed = (
            a["wins"] - a["losses"] >= s["net_floor"]
            and all(
                scores[k]["wins"] >= scores[k]["losses"] for k in ("esnli", "multinli")
            )
            and c["probe"] > d["probe"]
            and c["triplets"] > d["triplets"]
            and c["seconds"] <= s["max_time_ratio"] * d["seconds"]
            and ret["recurrent"]["nll"] <= ret["dense"]["nll"] + 0.03
        )
        result.update(status="completed", scores=scores, retention=ret, passed=passed)
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        budget.close()
    save(out / "decision.json", result)
    print(result["status"], result["passed"], flush=True)


if __name__ == "__main__":
    run()
