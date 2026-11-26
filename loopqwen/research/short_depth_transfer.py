"""Replicate the short arithmetic matched-depth result across published sources."""

import gc
from pathlib import Path

import mlx.core as mx

from ..deployment import load_trained
from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from . import core, rationale, rationale_transfer

NAME = "short-depth-transfer-01"


def run():
    reg = Path("research/experiments") / (NAME + ".json")
    s = read(reg)
    panel = Path(s["panel"])
    registry = read("research/registry.json")
    if registry["experiments"][NAME]["registration"] != digest(reg) or registry[
        "datasets"
    ][str(panel)]["sha256"] != digest(panel):
        raise ValueError("Unregistered transfer comparison")
    prev = read(Path(s["qualification"]) / "decision.json")
    if not prev["passed"]:
        raise ValueError("Failed matched development comparison")
    rows = read(panel)
    out = core.OUT / NAME
    out.mkdir(exist_ok=False, parents=True)
    result = dict(registration=digest(reg), data=digest(panel), passed=False)
    records = {"dense": [], "recurrent": []}
    reference = unpack("research/results/rationale-transfer-01/candidate.json.gz")[
        "records"
    ]
    b = core.Budget(NAME, s["limits"])
    try:
        for start, arms in [(0, ("recurrent", "dense")), (24, ("dense", "recurrent"))]:
            for arm in arms:
                model = Path(s[arm])
                w = digest(model / "adapters.safetensors")
                if w != prev["weights" if arm == "dense" else "recurrent_weights"]:
                    raise ValueError("Wrong matched weights")
                m, t, _ = load_trained(model)
                partial = out / (arm + "-half.json.gz")
                rationale.assess(m, t, w, rows[start : start + 24], s, b, partial)
                for i, r in enumerate(unpack(partial)["records"], start):
                    if (
                        arm == "recurrent"
                        and r["token_ids"] != reference[i]["token_ids"][:256]
                    ):
                        raise ValueError("Recurrent replay changed")
                    records[arm].append(dict(**r, seconds_at_256=r["seconds"]))
                packed(out / (arm + ".json.gz"), dict(weights=w, records=records[arm]))
                partial.unlink()
                del m, t
                gc.collect()
                mx.clear_cache()
        scores = rationale_transfer.summarize(
            rows, records["dense"], records["recurrent"], cap=256
        )
        for v in scores.values():
            v["dense"] = v.pop("base")
            v["recurrent"] = v.pop("candidate")
        a = scores["all"]
        retention = {k: prev[k]["retention"]["nll"] for k in records}
        result.update(
            status="completed",
            scores=scores,
            retention=retention,
            passed=all(
                v["wins"] - v["losses"] >= 2 for k, v in scores.items() if k != "all"
            )
            and a["p"] < 0.05
            and a["recurrent"]["seconds"] <= 1.30 * a["dense"]["seconds"]
            and retention["recurrent"] <= retention["dense"] + 0.03,
        )
    except (TimeoutError, MemoryError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(out / "decision.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    run()
