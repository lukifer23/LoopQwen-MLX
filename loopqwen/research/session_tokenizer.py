"""Measure complete field streams while retaining one tokenizer per offline assistant."""

import argparse
import resource
import subprocess
import sys
import weakref
from pathlib import Path

import mlx.core as mx

from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from ..package_tools import snapshot
from ..tool_model import Assistant
from ..worker import child_lease
from .core import Budget
from .mixed_field import measure_request
from .registered import registered


def worker(name, policy, result):
    spec = registered(name)
    material = read(spec["material"])
    if snapshot() != material["environment"]:
        raise ValueError("Field environment changed")
    if any(digest(Path(k)) != v["sha256"] for k, v in material["files"].items()):
        raise ValueError("Field source files changed")
    assistant = None
    previous = None
    rows = []

    def check():
        if mx.get_peak_memory() > spec["limits"]["peak_gib"] * 2**30:
            raise MemoryError("MLX allocation exceeds the registered ceiling")

    def make():
        return Assistant(Path(spec["checkpoint"]), reuse_tokenizer=(policy == "reuse"))

    for w in material["rows"]:
        assistant, r = measure_request(assistant, make, w, material, spec, check)
        r["same_tokenizer"] = previous is not None and previous() is assistant.tokenizer
        previous = weakref.ref(assistant.tokenizer)
        rows.append(r)
    # On macOS this is a byte count. The registered machine is a Darwin host.
    packed(
        result,
        dict(
            rows=rows,
            high_water_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        ),
    )


def outcome(root, spec):
    from scripts.tool_release import tokens

    ref = unpack(Path(spec["field"]) / "candidate.json.gz")
    summaries = {s: [] for s in ("stock", "reuse")}
    exact = []
    for i, side in enumerate(spec["order"]):
        run = unpack(root / f"{i}-{side}.json.gz")
        rows = run["rows"]
        switch_s = total = first = 0.0
        switches = fresh = peak = 0
        last = None
        same_in_stream = 0
        for old, r in zip(ref, rows, strict=True):
            a, b = old["trajectory"], r["trajectory"]
            assert (old["id"], old["family"]) == (r["id"], r["family"])
            assert 0 < r["first_token_s"] <= r["wall_s"] and r["request_peak"] > 0
            same_in_stream += (
                tokens(a) == tokens(b)
                and a["status"] == b["status"]
                and a["answer"] == b["answer"]
                and old["score"] == r["score"]
            )
            mode = "language" if r["family"] == "reading" else "tools"
            if last is not None and mode != last:
                switches += 1
                switch_s += r["first_token_s"]
            last = mode
            fresh += not r["same_tokenizer"]
            total += r["wall_s"]
            first += r["first_token_s"]
            peak = max(peak, r["request_peak"])
        assert len(rows) == 50 and switches == 15
        exact.append(same_in_stream)
        summaries[side].append(
            dict(
                switch_ttft=switch_s,
                all_ttft=first,
                seconds=total,
                peak=peak,
                constructions=fresh,
                rss=run["high_water_rss_bytes"],
            )
        )
    metrics = {}
    for side, rs in summaries.items():
        metrics[side] = {
            k: (
                max(r[k] for r in rs) if k in ("peak", "rss") else sum(r[k] for r in rs)
            )
            for k in rs[0]
        }
    a, b = metrics["reuse"], metrics["stock"]
    passed = (
        all(v == 50 for v in exact)
        and a["constructions"] == 2
        and b["constructions"] == 32
        and (
            a["switch_ttft"] <= (1 - spec["reduction"]) * b["switch_ttft"]
            and a["all_ttft"] < b["all_ttft"]
            and a["seconds"] <= b["seconds"]
            and a["peak"] <= b["peak"] + spec["mlx_tolerance"]
            and a["rss"] <= b["rss"] + spec["rss_tolerance"]
            and all(
                x["switch_ttft"] < y["switch_ttft"]
                for x, y in zip(summaries["reuse"], summaries["stock"], strict=True)
            )
        )
    )
    return dict(passed=bool(passed), exact=exact, metrics=metrics, streams=summaries)


def verify(root, d):
    spec = read(root / "registration.json")
    assert outcome(root, spec) == d


def run(name):
    spec = registered(name)
    out = Path("outputs/research") / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "decision.json").exists():
        raise FileExistsError(out)
    for k in ("field", "release"):
        p = Path(spec[k])
        if digest(p / "checksums.json") != spec[k + "_sha256"]:
            raise ValueError("Field archive identity changed")
    release = Path(spec["release"])
    if not read(release / "decision.json")["passed"] or any(
        digest(Path(spec["checkpoint"]) / k) != h
        for k, h in read(release / "export-files.json").items()
    ):
        raise ValueError("Unqualified packed bundle")
    bud = Budget(name, spec["limits"])
    try:
        for i, s in enumerate(spec["order"]):
            dest = out / f"{i}-{s}.json.gz"
            if dest.exists():
                continue
            opt = child_lease()
            opt["env"].update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            cmd = [
                sys.executable,
                "-m",
                "loopqwen.research.session_tokenizer",
                name,
                "--worker",
                s,
                "--result",
                str(dest),
            ]
            with subprocess.Popen(cmd, **opt) as child:
                try:
                    while child.poll() is None:
                        try:
                            child.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            bud.check()
                except BaseException:
                    child.terminate()
                    raise
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, cmd)
            print("finished", i, s, flush=True)
        decision = outcome(out, spec)
        save(out / "decision.json", decision)
        print(decision, flush=True)
    finally:
        bud.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("--worker", choices=("reuse", "stock"))
    p.add_argument("--result", type=Path)
    args = p.parse_args()
    if args.worker:
        worker(args.name, args.worker, args.result)
    else:
        run(args.name)
