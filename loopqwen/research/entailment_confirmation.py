"""Repeat a qualified entailment pair on new premises using a held-out instruction."""

import argparse
from pathlib import Path

from ..identity import fingerprint
from ..io import digest, read, save
from . import core, entailment, inference_data


def prepare(s):
    original = Path(s["model"]).parent
    first = read(original / "registration.json")
    before = read(Path("research/data") / (first["id"] + ".json"))
    if (
        digest(original / "checksums.json") != s["qualification"]
        or not read(original / "decision.json")["passed"]
        or digest(Path(s["model"]) / "adapters.safetensors") != s["weights"]
        or digest(Path(s["audit"])) != s["audit_sha256"]
    ):
        raise ValueError("Primary weights, evidence or audited responses differ")
    from scripts.research_relation_audit import audit

    primary = audit(first["id"], read(s["audit"]))["scores"]
    if any(v["wins"] <= v["losses"] for v in primary.values()):
        raise ValueError("Blinded primary gain required on both sources")
    train_groups, keys = inference_data.training_groups(first, before)
    excluded = [
        r for groups in train_groups.values() for group in groups for r in group
    ]
    manifest = read(entailment.TRAIN)
    train_keys = keys - {r["premise"].casefold() for r in before}
    if (
        set(manifest["ordered_questions"]) != {r["id"] for r in excluded}
        or fingerprint(sorted(train_keys)) != manifest["premises"]
    ):
        raise ValueError("Primary training population differs")
    return inference_data.material(s, before + excluded)


def run(name, freeze):
    reg = Path("research/experiments") / (name + ".json")
    panel = Path("research/data") / (name + ".json")
    s = read(reg)
    rows = prepare(s)
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
    root.mkdir(parents=True, exist_ok=False)
    b = core.Budget(name, s["limits"])
    result = {
        "passed": False,
        "registration": digest(reg),
        "data": digest(panel),
        "weights": s["weights"],
        "qualification": s["qualification"],
        "training": digest(entailment.TRAIN),
    }
    try:
        scores, ret = entailment.evaluate(s, rows, s["model"], s["weights"], b, root)
        result.update(
            status="completed",
            scores=scores,
            retention=ret,
            passed=entailment.passed(s, scores, ret),
        )
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        result.update(status="interrupted", reason=str(exc))
    finally:
        b.close()
    save(root / "decision.json", result)
    print(result["status"], result["passed"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    run(args.study, args.freeze)
