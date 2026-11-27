"""Run a same-weight, installed-runtime crossover through real two-message tool conversations."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.io import digest, read, save
from loopqwen.research.core import Budget
from loopqwen.research.file_evidence_environment import capture
from loopqwen.research.native_function_study import executions
from loopqwen.research.registered import registered
from loopqwen.worker import child_lease
from scripts.verify_runtime_wheel import ROOT, build, trace

NAME = "cache-state-conversations-01"

# Measure the actual first emitted token with the ordinary public progress callback.
# Keep the CLI's normal stdout separate from the timing receipt.
CHILD = """
import json, sys
from pathlib import Path
from time import perf_counter
def offline(name, args):
    if name in ('socket.connect', 'socket.getaddrinfo'):
        raise RuntimeError('Network attempted from isolated installation')
sys.addaudithook(offline)
site = Path(sys.argv.pop(1)).resolve()
receipt = Path(sys.argv.pop(1))
sys.path.insert(0, str(site))
import mlx.core as mx
from loopqwen import tool_session
from loopqwen.cli import main
original = tool_session.generate
first = []
def measured(*args, **kwargs):
    stamps = {}
    prior = kwargs.get('progress')
    def tick(n):
        if not first:
            stamps.setdefault(n, perf_counter())
        if prior:
            prior(n)
    out = original(*args, **{**kwargs, 'progress': tick})
    if not first:
        first.append(stamps[out['prompt_tokens']])
    return out
tool_session.generate = measured
main()
mx.synchronize()
assert all(Path(m.__file__).resolve().is_relative_to(site)
           for k, m in sys.modules.items()
           if (k == 'loopqwen' or k.startswith('loopqwen.')) and getattr(m, '__file__', None))
receipt.write_text(json.dumps({'first': first[0], 'peak': mx.get_peak_memory()}))
"""


def wheel(spec):
    directory = ROOT / "exports/cache-state-runtime-01"
    directory.mkdir(exist_ok=True)
    one = build(directory, spec["wheel_revision"])
    with TemporaryDirectory() as tmp:
        two = build(Path(tmp), spec["wheel_revision"])
        if digest(one) != digest(two):
            raise ValueError("Runtime builds differed at the pinned source revision")
    return one


def fixtures(s, work):
    panel = read(s["material"])
    for case in s["cases"]:
        flags = case["flags"]
        if "--file" in flags:
            path = Path(flags[1])
            assert digest(ROOT / path) == s["file_sha256"]
            (work / path).parent.mkdir(parents=True, exist_ok=True)
            (work / path).write_bytes((ROOT / path).read_bytes())
        if "--keyed-csv" in flags:
            name = flags[1]
            (work / name).write_bytes(panel["documents"][name].encode())


def execute(s, case, mode, site, work, guard):
    timing = work / "timing.json"
    timing.unlink(missing_ok=True)
    args = [
        sys.executable,
        "-I",
        "-c",
        CHILD,
        str(site),
        str(timing),
        "ask",
        "--checkpoint",
        str(ROOT / s["checkpoint"]),
        "--trace",
        "--retain-conversation-prefix",
        "--tool-format",
        case["format"],
        "--max-tokens",
        str(s["max_tokens"]),
        "--max-turns",
        str(s["max_turns"]),
        "--prompt",
        case["prompts"][0],
        "--follow-up",
        case["prompts"][1],
        *case["flags"],
    ]
    if mode == "state":
        args.append("--cache-state-prefill")
    lease = child_lease()
    cache = work / "empty-model-cache"
    lease["env"].update(
        HF_HOME=str(cache),
        HF_HUB_CACHE=str(cache / "hub"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    start = perf_counter()
    with subprocess.Popen(
        args, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **lease
    ) as child:
        try:
            while True:
                guard.check()
                try:
                    stdout, stderr = child.communicate(timeout=0.7)
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()
    end = perf_counter()
    if child.returncode:
        raise RuntimeError(stderr.decode()[-4000:])
    measured = read(timing)
    if measured["peak"] > s["limits"]["peak_gib"] * 2**30:
        raise MemoryError("Installed child MLX memory limit")
    exchanges = json.loads(stdout)["exchanges"]
    assert [r["question"] for r in exchanges] == case["prompts"]
    return {
        "id": case["id"],
        "cold_s": end - start,
        "cold_first_s": measured["first"] - start,
        "peak": measured["peak"],
        "exchanges": exchanges,
    }


def summary(items):
    turns = [t for q in items for r in q["exchanges"] for t in r["turns"]]
    return {
        "requests": sum(len(q["exchanges"]) for q in items),
        "cold_s": sum(q["cold_s"] for q in items),
        "cold_first_s": sum(q["cold_first_s"] for q in items),
        "peak": max(q["peak"] for q in items),
        "turn_open_s": sum(t["ttft_s"] for t in turns),
        "decode_tps": sum(len(t["token_ids"]) - 1 for t in turns)
        / sum(t["seconds"] - t["ttft_s"] for t in turns),
        "output_tokens": sum(len(t["token_ids"]) for t in turns),
        "reused": sum(t.get("reused_prompt_tokens", 0) for t in turns),
        "state_blocks": sum(t.get("cache_state_blocks", 0) for t in turns),
        "callback_domains": sorted(
            {
                q["id"]
                for q in items
                if any(
                    "error" not in e["observation"]
                    for r in q["exchanges"]
                    for e in executions(r)
                )
            }
        ),
    }


def decide(s, rounds):
    assert [block["mode"] for block in rounds] == s["order"]
    full = {}
    reference = {}
    identical = exercised = True
    for block in rounds:
        items = block["items"]
        assert [q["id"] for q in items] == [c["id"] for c in s["cases"]]
        for item in items:
            assert 0 < item["cold_first_s"] < item["cold_s"] and item["peak"] > 0
            obs = item["exchanges"]
            streams = [
                [trace(q), [t.get("reused_prompt_tokens", 0) for t in q["turns"]]]
                for q in obs
            ]
            identical &= streams == reference.setdefault(item["id"], streams)
            for exchange in obs:
                for t in exchange["turns"]:
                    expected = (
                        len(t["prompt_ids"]) - t.get("reused_prompt_tokens", 0) - 1
                    ) // 128
                    exercised &= t.get("cache_state_blocks", 0) == (
                        expected if block["mode"] == "state" else 0
                    )
            if block["mode"] == "state":
                exercised &= summary([item])["state_blocks"] > 0
        full.setdefault(block["mode"], []).extend(items)
    totals = {k: summary(rows) for k, rows in full.items()}
    a, b = totals["state"], totals["stock"]
    ratios = []
    for j in range(0, len(rounds), 2):
        by = {r["mode"]: summary(r["items"]) for r in rounds[j : j + 2]}
        ratios.append(by["state"]["turn_open_s"] / by["stock"]["turn_open_s"])
    gates = {
        "identical": identical,
        "exercised": exercised,
        "live_tools": len(a["callback_domains"]) >= 2
        and a["callback_domains"] == b["callback_domains"],
        "reused": a["reused"] == b["reused"] > 0,
        "peak": a["peak"] <= b["peak"],
        "cold": a["cold_s"] < b["cold_s"],
        "cold_first": a["cold_first_s"] < b["cold_first_s"],
        "native_open": a["turn_open_s"] <= 0.99 * b["turn_open_s"]
        and sum(r < 1 for r in ratios) >= 3,
        "decode": a["decode_tps"] >= b["decode_tps"],
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "totals": totals,
        "native_open_ratios": ratios,
    }


def verify(root, decision):
    s = registered(NAME)
    assert read(root / "registration.json") == s
    if decision.get("status") == "interrupted":
        assert not decision["passed"] and decision["reason"]
        return
    rounds = unpack(root / "rounds.json.gz")
    assert decide(s, rounds) == decision
    build_info = read(root / "installed-wheel.json")
    assert (
        build_info["source"] == s["wheel_revision"]
        and build_info["independent_builds"] == 2
    )


def run():
    s = registered(NAME)
    assert digest(Path(s["requires"]) / "checksums.json") == s["prerequisite_sha256"]
    assert read(Path(s["requires"]) / "decision.json")["passed"]
    root = ROOT / "outputs/research" / NAME
    root.mkdir(exist_ok=True)
    if (root / "decision.json").exists() or (root / "rounds.json.gz").exists():
        raise ValueError("A measured crossover already exists")
    save(root / "registration.json", s)
    capture(
        root,
        ("scripts/cache_state_conversations.py", "scripts/verify_runtime_wheel.py"),
    )
    guard = Budget(NAME, s["limits"])
    try:
        runtime = wheel(s)
        save(
            root / "installed-wheel.json",
            {
                "source": s["wheel_revision"],
                "independent_builds": 2,
                "wheel": runtime.name,
                "sha256": digest(runtime),
                "bytes": runtime.stat().st_size,
            },
        )
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            site = work / "installed"
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--offline",
                    "--no-deps",
                    "--target",
                    str(site),
                    str(runtime),
                ],
                check=True,
            )
            fixtures(s, work)
            rounds = []
            for i, mode in enumerate(s["order"]):
                items = [execute(s, c, mode, site, work, guard) for c in s["cases"]]
                rounds.append({"mode": mode, "items": items})
                packed(root / "rounds.json.gz", rounds)
                print("Installed round", i + 1, mode, flush=True)
            decision = decide(s, rounds)
    except (MemoryError, TimeoutError, InterruptedError) as e:
        decision = {"status": "interrupted", "passed": False, "reason": str(e)}
    finally:
        guard.close()
    save(root / "decision.json", decision)
    verify(root, decision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-wheel",
        action="store_true",
        help="Rebuild and compare both offline wheel copies",
    )
    if parser.parse_args().build_wheel:
        print(wheel(registered(NAME)))
    else:
        run()
