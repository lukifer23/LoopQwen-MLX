"""Freeze a disjoint full-budget arithmetic pair for a qualified unchanged checkpoint."""

import argparse
from pathlib import Path

from ..io import digest, read, save
from . import core, deliberation, paired_solutions, rationale_transfer


def material(s):
    prior = [r for path in s["excluded_panels"] for r in read(path)]
    used = {q for path in s["excluded_training"] for q in read(path)["ordered_questions"]}
    _, rows = paired_solutions.material(s, prior, exclude_ids=used)
    return rows


def run(name, freeze):
    reg = Path("research/experiments") / (name + ".json")
    panel = Path("research/data") / (name + ".json")
    s = read(reg)
    original = Path(s["model"]).parent
    if (
        not read(original / "decision.json")["passed"]
        or digest(original / "checksums.json") != s["qualification"]
    ):
        raise ValueError("The primary qualification failed or changed")
    from scripts.research_answer_audit import score

    semantic = score(original.name, read(Path(s["audit"])))
    if semantic["wins"] <= semantic["losses"]:
        raise ValueError("The original content advantage did not survive review")
    rows = material(s)
    if freeze and not panel.exists():
        save(panel, rows)
    if rows != read(panel):
        raise ValueError("Confirmation material changed")
    if freeze:
        return
    idx = read("research/registry.json")
    if idx["datasets"][str(panel)]["sha256"] != digest(panel) or idx["experiments"][
        name
    ]["registration"] != digest(reg):
        raise ValueError("Unregistered confirmation")
    root = core.OUT / name
    root.mkdir(exist_ok=True, parents=True)
    if (root / "decision.json").exists():
        raise FileExistsError(root)
    w = digest(Path(s["model"]) / "adapters.safetensors")
    if w != s["weights"]:
        raise ValueError("The qualified model weights changed")
    result = {
        "passed": False,
        "weights": w,
        "registration": digest(reg),
        "data": digest(panel),
        "qualification": s["qualification"],
    }
    b = core.Budget(name, s["limits"])
    try:
        rationale_transfer.evaluate_pair(
            root,
            s["model"],
            rows,
            s,
            w,
            b,
            result,
            retention=True,
            decoder=deliberation.decode,
        )
        a = result["full"]
        result["passed"] = (
            a["all"]["wins"] - a["all"]["losses"] >= s["net_floor"]
            and all(a[k]["wins"] > a[k]["losses"] for k in s["sources"])
            and a["all"]["candidate"]["seconds"] <= a["all"]["base"]["seconds"]
            and result["retention"]["candidate"]["nll"]
            <= result["retention"]["base"]["nll"] + 0.03
        )
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result["status"], result["passed"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("study")
    p.add_argument("--freeze", action="store_true")
    args = p.parse_args()
    run(args.study, args.freeze)
