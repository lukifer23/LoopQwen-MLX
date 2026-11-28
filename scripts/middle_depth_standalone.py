"""Build two independent model folders and replay the registered offline pair."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep, time

import psutil

from loopqwen import REVISION
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.io import digest, read, save
from loopqwen.worker import child_lease
from scripts.middle_depth_archive import ARCHIVE, verify
from scripts.middle_depth_budget import ROOT, worker
from scripts.middle_depth_cold import GUARD
from scripts.middle_depth_development import SOURCE, instance
from scripts.verify_runtime_wheel import build, trace

MODELS = Path("exports/standalone-middle-576")
REPORT = ROOT / "standalone"
PLAN = SOURCE.with_name("standalone-pair.md")
STAGE = "source-supervised"
ENTRY = """import sys,runpy
from pathlib import Path
site=Path(sys.argv.pop(1)).resolve()
sys.path.insert(0,str(site))
runpy.run_module('loopqwen.standalone_replay',run_name='__main__')
for n,m in list(sys.modules.items()):
    if n=='loopqwen' or n.startswith('loopqwen.'):
        files=[m.__file__] if getattr(m,'__file__',None) else list(m.__path__)
        assert files and all(Path(f).resolve().is_relative_to(site) for f in files),n
"""


def check(folder, arm, spec=None):
    if spec is None:
        spec = read(REPORT / (arm + "-files.json"))
    actual = {
        str(p.relative_to(folder)): digest(p) for p in folder.rglob("*") if p.is_file()
    }
    if actual != spec["files"] or any(p.is_symlink() for p in folder.rglob("*")):
        raise ValueError(f"Standalone {arm} differs from its registered bytes")
    return spec


def prepare(arms=("base", "loop")):
    if all((MODELS / arm).is_dir() for arm in arms):
        for arm in arms:
            check(MODELS / arm, arm)
        return
    verify(ARCHIVE, STAGE)
    registered = read(ARCHIVE / "manifests" / (STAGE + ".json"))
    MODELS.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=MODELS) as t:
        tmp = Path(t)
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if arms == ("ordinary",):
            base = check(MODELS / "base", "base")
            rev = base["revision"]
            wheel = next((MODELS / "base").glob("*.whl"))
        else:
            wheel = build(tmp, rev)
        for arm in arms:
            target = MODELS / arm
            if target.exists():
                check(target, arm)
                continue
            root = tmp / arm
            root.mkdir()
            for name, r in registered["files"].items():
                if name.startswith("donor/"):
                    rel = Path(name).relative_to("donor")
                elif arm != "base" and name.startswith(f"models/{STAGE}/{arm}/"):
                    rel = Path(name).name
                else:
                    continue
                dest = root / rel
                dest.parent.mkdir(exist_ok=True, parents=True)
                subprocess.run(
                    ["cp", "-c", str((ARCHIVE / name).resolve()), dest], check=True
                )
                if dest.is_symlink() or digest(dest) != r["sha256"]:
                    raise ValueError(f"Incorrect standalone copy {dest}")
            subprocess.run(["cp", "-c", wheel, root / wheel.name], check=True)
            if arm != "ordinary":
                save(
                    root / "standalone-middle.json",
                    {
                        "schema": 1,
                        "arm": arm,
                        "base_revision": REVISION,
                        "policy_sha256": digest(root / "registration.json")
                        if arm == "loop"
                        else None,
                    },
                )
            files = {
                str(f.relative_to(root)): digest(f)
                for f in root.rglob("*")
                if f.is_file()
            }
            save(
                REPORT / (arm + "-files.json"),
                {
                    "files": files,
                    "revision": rev,
                    "source_manifest": digest(
                        ARCHIVE / "manifests" / (STAGE + ".json")
                    ),
                },
            )
            check(root, arm)
            root.rename(target)


def full_fit_bundle(policy):
    """Build an independently movable folder for the verified full-document fit."""
    from scripts.middle_depth_direct import parent
    from scripts.middle_depth_stages import verify_long_fit

    proof = verify_long_fit("loop", parent()[1], policy)
    template = MODELS / "loop"
    parent_manifest = check(template, "loop")
    if parent_manifest["files"]["adapters.safetensors"] != proof["parent"]:
        raise ValueError("Standalone template differs from the training parent")
    target = MODELS.parent / "standalone-middle-long" / proof["weights"]
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(parent_manifest, files=dict(parent_manifest["files"]), fit=proof)
    spec = read(template / "standalone-middle.json")
    spec["policy_sha256"] = digest(policy / "registration.json")
    with TemporaryDirectory(dir=target.parent) as tmp:
        marker = Path(tmp) / "standalone-middle.json"
        save(marker, spec)
        replacements = {
            "standalone-middle.json": marker,
            "registration.json": policy / "registration.json",
            "adapters.safetensors": policy / "adapters.safetensors",
        }
        manifest["files"].update({k: digest(v) for k, v in replacements.items()})
        if not target.exists():
            root = Path(tmp) / "model"
            subprocess.run(["cp", "-cR", template, root], check=True)
            for name, src in replacements.items():
                subprocess.run(["cp", src, root / name], check=True)
            check(root, "loop", manifest)
            root.rename(target)
        check(target, "loop", manifest)
    save(REPORT / "long-loop-files.json", manifest)
    return target


BENCH_PLAN = SOURCE.with_name("standalone-benchmark.md")
BENCH = REPORT / "selected-benchmark"


def benchmark_model(arm):
    from scripts.middle_depth_direct import CHOSEN

    decision = read(CHOSEN)
    if arm == "loop" and decision["retained_full_fit"]:
        directory = full_fit_bundle(Path(decision["models"]["loop"]["path"]))
        spec = read(REPORT / "long-loop-files.json")
    else:
        directory = MODELS / arm
        spec = read(REPORT / (arm + "-files.json"))
    if (
        arm == "loop"
        and spec["files"]["adapters.safetensors"]
        != decision["models"]["loop"]["weights"]
    ):
        raise ValueError(
            "Selected standalone weights differ from the scored checkpoint"
        )
    return directory, spec


def replay(arm, benchmark=False, cap=512):
    keys = []
    plan = BENCH_PLAN if benchmark else PLAN
    directory, spec = benchmark_model(arm) if benchmark else (MODELS / arm, None)
    output = BENCH / str(cap) if benchmark else REPORT
    job = "installed-benchmark/" + str(cap) if benchmark else "standalone-pair"
    grade_stage = (
        read(ROOT / "direct-decision.json")["short_stage"]
        if benchmark and arm == "loop"
        else "terminal"
    )
    with (
        worker("profile", job + "/" + arm) as guard,
        TemporaryDirectory() as t,
    ):
        tmp = Path(t)
        model = tmp / "model"
        subprocess.run(["cp", "-cR", directory, model], check=True)
        manifest = check(model, arm, spec)
        (wheel,) = model.glob("*.whl")
        site = tmp / "site"
        subprocess.run(
            ["uv", "pip", "install", "--offline", "--no-deps", "--target", site, wheel],
            check=True,
        )
        jobs = []
        for task in (
            r
            for r in read(SOURCE)
            if benchmark or r["id"] in ("archive", "volunteer", "tiles", "parcels")
        ):
            for world in (
                ("original", "changed", "irrelevant")
                if benchmark
                else ("original", "changed")
            ):
                folder = tmp / (task["id"] + "-" + world)
                folder.mkdir()
                question, _ = instance(task, world, folder)
                job = {"prompts": [question]}
                keys.append((task["id"], world))
                if task.get("provider") == "calculator":
                    job["calculator"] = True
                if task.get("provider") == "csv":
                    job["csv"] = str(folder / task["document"])
                if world == "original" and "followup" in task:
                    job["prompts"].append(task["followup"])
                    keys.append((task["id"], "own"))
                jobs.append(job)
        requests = tmp / "requests.json"
        save(requests, jobs)
        lease = child_lease()
        lease["env"].update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HOME=str(tmp / "empty"),
            LOOPQWEN_COLD_METRICS=str(tmp / "peak.json"),
        )
        logfile = tmp / "output.jsonl"
        launched = time()
        with (
            logfile.open("w") as out,
            subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    GUARD + ENTRY,
                    str(site),
                    "--checkpoint",
                    model,
                    "--requests",
                    requests,
                    "--max-tokens",
                    str(cap),
                ],
                cwd=tmp,
                stdout=out,
                **lease,
            ) as child,
        ):
            try:
                while child.poll() is None:
                    guard()
                    sleep(0.1)
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
        ended = time()
        if child.returncode or psutil.pid_exists(child.pid):
            packed(
                output / f"failed-{arm}-{child.pid}.json.gz",
                {
                    "arm": arm,
                    "pid": child.pid,
                    "exit": child.returncode,
                    "requests": keys,
                    "lines": logfile.read_text().splitlines(),
                    "manifest": manifest,
                    "plan": digest(plan),
                },
            )
            raise RuntimeError(
                f"Standalone native PID {child.pid}, exit {child.returncode}"
            )
        header, *lines = map(json.loads, logfile.read_text().splitlines())
        assert header["pid"] == child.pid and len(lines) == len(keys) == (
            29 if benchmark else 10
        )
        grades = {
            (r["source"], r["world"]): r
            for r in read(ROOT / "review" / grade_stage / "scored.json")
            if r["arm"] == arm and r["cap"] == cap
        }
        rows = []
        for (source, world), entry in zip(keys, lines, strict=True):
            ref = grades[source, world]
            path = ROOT / ref["file"]
            assert digest(path) == ref["sha"]
            same = trace(entry["result"]) == trace(unpack(path)["trace"])
            rows.append(
                dict(
                    source=source,
                    world=world,
                    identical=same,
                    grade=ref["grade"] if same else None,
                    reference=ref["file"],
                    ref_sha=ref["sha"],
                    **entry,
                )
            )
        result = {
            "arm": arm,
            "registration": digest(plan),
            "manifest": manifest,
            "loaded": header,
            "launched": launched,
            "exited": ended,
            "pid_absent": True,
            "peak": read(tmp / "peak.json")["peak_bytes"],
            "passed": all(x["identical"] for x in rows),
            "rows": rows,
        }
        packed(output / (arm + "-replay.json.gz"), result)
        if not result["passed"]:
            raise ValueError(
                "Installed model differs from recorded development trajectory"
            )


def run():
    from scripts.middle_depth_wave import dispatch

    prepare()
    summaries = {}
    last = None
    for arm in ("base", "loop"):
        dispatch("standalone", "replay", "--arm", arm)
        r = unpack(REPORT / (arm + "-replay.json.gz"))
        if last and last["exited"] > r["launched"]:
            raise ValueError("Model lifetimes overlapped")
        last = r
        summaries[arm] = {
            "pid": r["loaded"]["pid"],
            "load_seconds": r["loaded"]["loaded_seconds"],
            "process_seconds": r["exited"] - r["launched"],
            "generation_seconds": sum(x["elapsed"] for x in r["rows"]),
            "correct": sum(x["grade"]["correct"] for x in r["rows"]),
            "matched": len(r["rows"]),
            "peak": r["peak"],
            "trace": digest(REPORT / (arm + "-replay.json.gz")),
        }
    save(
        REPORT / "pair.json",
        {"order": list(summaries), "arms": summaries, "plan": digest(PLAN)},
    )


def benchmark():
    from scripts.middle_depth_wave import dispatch

    summaries = {}
    previous_exit = None
    for cap, order in ((128, ("loop", "base")), (512, ("base", "loop"))):
        for arm in order:
            directory, spec = benchmark_model(arm)
            check(directory, arm, spec)
            path = BENCH / str(cap) / (arm + "-replay.json.gz")
            if not path.exists():
                dispatch("standalone", "benchmark-arm", "--arm", arm, "--cap", cap)
            r = unpack(path)
            if (
                r["manifest"] != spec
                or not r["passed"]
                or not r["pid_absent"]
                or r["registration"] != digest(BENCH_PLAN)
                or len(r["rows"]) != 29
                or (previous_exit and previous_exit > r["launched"])
            ):
                raise ValueError(f"Invalid completed benchmark {path}")
            previous_exit = r["exited"]
            groups = {}
            for kind in ("opening", "own_history"):
                rows = [
                    x
                    for x in r["rows"]
                    if (x["world"] == "own") == (kind == "own_history")
                ]
                for x in rows:
                    if (
                        digest(ROOT / x["reference"]) != x["ref_sha"]
                        or not x["identical"]
                    ):
                        raise ValueError("Benchmark reference changed")
                groups[kind] = {
                    "n": len(rows),
                    "correct": sum(x["grade"]["correct"] for x in rows),
                    "elapsed": sum(x["elapsed"] for x in rows),
                    "mean_capped_time_to_correct": sum(
                        min(x["elapsed"], 8) if x["grade"]["correct"] else 8
                        for x in rows
                    )
                    / len(rows),
                }
            summaries[f"{cap}/{arm}"] = {
                "trace": digest(path),
                "pid": r["loaded"]["pid"],
                "load_seconds": r["loaded"]["loaded_seconds"],
                "process_seconds": r["exited"] - r["launched"],
                "peak": r["peak"],
                "groups": groups,
            }
            save(BENCH / "result.json", {"plan": digest(BENCH_PLAN), "runs": summaries})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=(
            "build",
            "build-ordinary",
            "replay",
            "run",
            "benchmark-arm",
            "benchmark",
        ),
    )
    p.add_argument("--arm", choices=("base", "loop"))
    p.add_argument("--cap", type=int, choices=(128, 512), default=512)
    a = p.parse_args()
    if a.action == "build":
        prepare()
    elif a.action == "build-ordinary":
        prepare(("ordinary",))
    elif a.action == "benchmark":
        benchmark()
    elif a.action == "benchmark-arm":
        replay(a.arm, benchmark=True, cap=a.cap)
    elif a.action == "replay":
        replay(a.arm)
    else:
        run()
