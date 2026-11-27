"""Freeze the paired 4,096-token reruns from stopping status alone."""

import argparse
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_development import SOURCE, WORLDS
from scripts.middle_depth_wave import dispatch

REG = Path("research/diagnostics/middle-depth/output-headroom.md")
ARMS = ("base", "loop", "ordinary", "instruction")


def run(stage, policies):
    folder = ROOT / "short" / stage / "512"
    plan = ROOT / "headroom" / (stage + ".json")
    sources = read(SOURCE)
    items = set()
    reasons = {}
    hashes = {}
    for family in ("reasoning", "tools", "chat"):
        r = next(r for r in sources if r["family"] == family)
        items.add(r["id"] + "/original")
    for r in sources:
        for world in (*WORLDS, *(("own", "shared") if "followup" in r else ())):
            key = r["id"] + "/" + world
            for arm in ARMS:
                path = folder / arm / (r["id"] + "-" + world + ".json.gz")
                if not path.exists() and arm in ("base", "instruction"):
                    path = ROOT / "short" / "terminal" / "512" / arm / path.name
                row = unpack(path)
                hashes[key + "/" + arm] = digest(path)
                if row["trace"]["status"] == "token_limit":
                    reasons.setdefault(key, []).append(arm)
                    items.add(
                        r["id"]
                        + "/"
                        + ("original" if world in ("own", "shared") else world)
                    )
    spec = {
        "registration": digest(REG),
        "items": sorted(items),
        "stops": reasons,
        "source_trajectories": hashes,
    }
    if plan.exists() and read(plan) != spec:
        raise ValueError("Changed earlier generations after headroom enrollment")
    save(plan, spec)
    for block in (0, 1):
        for arm in ARMS if block == 0 else reversed(ARMS):
            args = [
                "collect",
                "--arm",
                arm,
                "--stage",
                stage,
                "--tokens",
                4096,
                "--block",
                block,
                "--items",
                plan,
            ]
            if arm in policies:
                args += ["--policy", policies[arm]]
            dispatch("development", *args)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--loop", type=Path, required=True)
    p.add_argument("--ordinary", type=Path, required=True)
    a = p.parse_args()
    run(a.stage, {"loop": a.loop, "ordinary": a.ordinary})
