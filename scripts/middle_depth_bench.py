"""Run five 50-item canonical benchmarks, one independent model process at a time."""

import argparse
from math import comb, sqrt
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import sleep, time

import numpy as np

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.worker import child_lease
from scripts.middle_depth_bench_sources import DATA, OUT, PLAN, TASKS
from scripts.middle_depth_budget import worker
from scripts.middle_depth_cold import GUARD
from scripts.middle_depth_standalone import MODELS, check

PYTHON = DATA / "env/bin/python"
REQUESTS = DATA / "requests.json"
TOKENS = OUT / "tokens.json"
ENROLL = OUT / "enrollment.json"


def freeze_tokens():
    from tokenizers import Tokenizer

    frozen = read(ENROLL)
    if frozen["requests"] != digest(REQUESTS) or frozen["plan"] != digest(PLAN):
        raise ValueError("Standard benchmark enrollment changed")
    tok = Tokenizer.from_file(str(MODELS / "base/tokenizer.json"))
    out = {}
    for row in read(REQUESTS):
        pairs = []
        for context, continuation in row["requests"]:
            prefix = tok.encode(context.rstrip(), add_special_tokens=False).ids
            whole = tok.encode(context + continuation, add_special_tokens=False).ids
            if (
                not prefix
                or whole[: len(prefix)] != prefix
                or len(whole) <= len(prefix)
            ):
                raise ValueError(f"Ambiguous tokenization on {row['key']}")
            pairs.append([prefix, whole[len(prefix) :]])
        out[row["key"]] = fingerprint(pairs)
    if len(out) != 250 or (TOKENS.exists() and read(TOKENS) != out):
        raise ValueError("Frozen input mismatch")
    save(TOKENS, out)


def verify(arm):
    subprocess.run(
        [PYTHON, "-m", "scripts.middle_depth_bench_sources", arm], check=True
    )


def run_arm(arm):
    if arm != "base":
        base = read(OUT / "base-process.json")
        if base["exit"] != 0 or digest(OUT / "base-grades.json") != base["grades"]:
            raise ValueError("Base verification must finish first")
    manifest = check(MODELS / arm, arm)
    with worker("short", "standard-five/" + arm) as guard, TemporaryDirectory() as tmp:
        temp = Path(tmp)
        site = temp / "site"
        (wheel,) = (MODELS / arm).glob("*.whl")
        subprocess.run(
            ["uv", "pip", "install", "--offline", "--no-deps", "--target", site, wheel],
            check=True,
        )
        env = child_lease()
        env["env"].update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HOME=str(temp / "empty"),
            LOOPQWEN_COLD_METRICS=str(temp / "metrics.json"),
        )
        script = Path("scripts/middle_depth_bench_native.py").resolve()
        code = (
            GUARD
            + "\nimport runpy\nfrom pathlib import Path\nsite=Path(sys.argv.pop(1)).resolve()\nsys.path.insert(0,str(site))\nentry=sys.argv.pop(1)\nrunpy.run_path(entry,run_name='__main__')\nimport loopqwen\nassert Path(loopqwen.__file__).resolve().is_relative_to(site)\n"
        )
        cmd = [
            sys.executable,
            "-I",
            "-c",
            code,
            str(site),
            str(script),
            "--checkpoint",
            (MODELS / arm).resolve(),
            "--requests",
            REQUESTS.resolve(),
            "--tokens",
            TOKENS.resolve(),
            "--output",
            (OUT / arm).resolve(),
        ]
        start = time()
        with subprocess.Popen(cmd, cwd=temp, **env) as p:
            try:
                while p.poll() is None:
                    guard()
                    sleep(0.15)
            finally:
                if p.poll() is None:
                    p.terminate()
                    p.wait()
        end = time()
        receipt = {
            "pid": p.pid,
            "exit": p.returncode,
            "launched": start,
            "exited": end,
            "cold_seconds": end - start,
            "manifest": manifest,
            "enrollment": digest(ENROLL),
            "tokens": digest(TOKENS),
            "plan": digest(PLAN),
            "runner": digest(script),
            "peak": read(temp / "metrics.json")["peak_bytes"],
        }
        save(OUT / (arm + "-process.json"), receipt)
        if p.returncode:
            raise RuntimeError(f"Installed benchmark worker exited {p.returncode}")
    verify(arm)
    receipt["grades"] = digest(OUT / (arm + "-grades.json"))
    save(OUT / (arm + "-process.json"), receipt)


def wilson(k, n):
    z = 1.959963984540054
    center = (k + z * z / 2) / (n + z * z)
    half = z * sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return [max(0, center - half), min(1, center + half)]


def summarize():
    tables = {}
    rng = np.random.default_rng(4619)
    reps = []
    for arm in ("base", "loop"):
        receipt = read(OUT / (arm + "-process.json"))
        if receipt["grades"] != digest(OUT / (arm + "-grades.json")) or receipt["exit"]:
            raise ValueError("Missing verified grades")
        tables[arm] = {r["key"]: r for r in read(OUT / (arm + "-grades.json"))}
    if (
        tables["base"].keys() != tables["loop"].keys()
        or read(OUT / "base-process.json")["exited"]
        > read(OUT / "loop-process.json")["launched"]
    ):
        raise ValueError("Mismatched benchmark or overlapping model lifetimes")
    groups = {}
    for name in TASKS:
        keys = [k for k in tables["base"] if k.split("/")[0] == name]
        if len(keys) != 50:
            raise ValueError(name)
        paired = []
        result = {}
        for arm, table in tables.items():
            hits = sum(table[k]["correct"] for k in keys)
            seconds = sum(table[k]["seconds"] for k in keys)
            result[arm] = {
                "hits": hits,
                "n": 50,
                "raw_accuracy": sum(table[k]["raw_correct"] for k in keys) / 50,
                "input_positions": sum(table[k]["input_positions"] for k in keys),
                "scored_tokens": sum(table[k]["scored_tokens"] for k in keys),
                "accuracy": hits / 50,
                "wilson95": wilson(hits, 50),
                "seconds": seconds,
                "examples_per_second": 50 / seconds,
                "peak_mlx": max(table[k]["peak"] for k in keys),
            }
        for k in keys:
            if (
                tables["base"][k]["token_sha"] != tables["loop"][k]["token_sha"]
                or tables["base"][k]["gold"] != tables["loop"][k]["gold"]
            ):
                raise ValueError("Mismatched paired tokens or gold")
            paired.append(
                int(tables["loop"][k]["correct"]) - int(tables["base"][k]["correct"])
            )
        gains, losses = paired.count(1), paired.count(-1)
        nd = gains + losses
        p = min(1, 2 * sum(comb(nd, i) for i in range(min(gains, losses) + 1)) / 2**nd)
        draws = rng.choice(paired, size=(50000, 50)).mean(axis=1)
        reps.append(draws)
        groups[name] = {
            **result,
            "gains": gains,
            "losses": losses,
            "delta": sum(paired) / 50,
            "paired95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "exact_mcnemar_p": p,
            "zero_discordance_95_upper": 1 - 0.05 ** (1 / len(keys))
            if nd == 0
            else None,
        }
    floor = 0.0
    for i, (name, r) in enumerate(
        sorted(groups.items(), key=lambda it: it[1]["exact_mcnemar_p"])
    ):
        floor = max(floor, min(1, r["exact_mcnemar_p"] * (len(groups) - i)))
        r["holm_p"] = floor
        r["material_gain"] = r["delta"] >= 0.10 and floor < 0.05
    delta = sum(x["delta"] for x in groups.values()) / 5
    interval = np.quantile(np.mean(reps, axis=0), [0.025, 0.975]).tolist()
    save(
        OUT / "result.json",
        {
            "benchmarks": groups,
            "macro_delta": delta,
            "mean_accuracy": {
                arm: sum(r[arm]["accuracy"] for r in groups.values()) / len(groups)
                for arm in tables
            },
            "serving": {
                arm: {
                    "wall_seconds_across_items": sum(
                        r[arm]["seconds"] for r in groups.values()
                    ),
                    "peak_mlx": max(r[arm]["peak_mlx"] for r in groups.values()),
                    "bundle_logical_bytes": sum(
                        (MODELS / arm / k).stat().st_size
                        for k in read(OUT / (arm + "-process.json"))["manifest"][
                            "files"
                        ]
                    ),
                }
                for arm in tables
            },
            "interval_scope": "Seeded stratified paired-item bootstrap conditional on this benchmark set. For zero switches the bootstrap delta is degenerate. The exact one-sided upper bound on the switch probability is reported separately. The fixed Base-first order can affect time comparisons.",
            "paired95": interval,
            "parity_lower_bound_passes": interval[0] > -0.02,
            "plan": digest(PLAN),
            "enrollment": digest(ENROLL),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("tokens", "base", "loop", "pair", "report"))
    stage = p.parse_args().stage
    if stage == "tokens":
        freeze_tokens()
    elif stage == "report":
        summarize()
    elif stage == "pair":
        freeze_tokens()
        for arm in ("base", "loop"):
            path = OUT / (arm + "-process.json")
            if not path.exists() or "grades" not in read(path):
                subprocess.run(
                    [sys.executable, "-m", "scripts.middle_depth_bench", arm],
                    check=True,
                )
        summarize()
    else:
        run_arm(stage)
