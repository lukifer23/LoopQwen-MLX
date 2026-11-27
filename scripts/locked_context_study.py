"""Run and independently grade changed long documents in the installed two-decoder CLI."""

import gzip
import json
import re
import subprocess
import sys
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from loopqwen import REVISION
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.file_tools import tools as file_tools
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.core import Budget
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.file_evidence_environment import capture
from loopqwen.research.native_function_study import executions, replay
from loopqwen.research.registered import registered
from loopqwen.worker import child_lease
from scripts.locked_context_material import SOURCE, material, request
from scripts.verify_runtime_wheel import CODE, ROOT

NAME = "locked-context-01"
# Enforce the peak during long prefill even though Budget lives in the parent.
GUARD = """
import sys
limit = int(sys.argv.pop(1))
def offline(name, args):
    if name in ('socket.connect', 'socket.getaddrinfo'):
        raise RuntimeError('Network request from offline installation')
sys.addaudithook(offline)
"""
HOOK = """
import mlx.core as mx
from loopqwen import tool_session
generate = tool_session.generate
def bounded(*args, **kwargs):
    callback = kwargs['progress']
    def tick(n):
        if mx.get_peak_memory() > limit:
            raise MemoryError('Long-context child MLX peak')
        callback(n)
    return generate(*args, **{**kwargs, 'progress': tick})
tool_session.generate = bounded
"""


def score(row, result):
    text = (result["answer"] or "").strip()
    numbers = re.findall(r"(?<![\w.])\d+(?:[.,]\d+)*(?!\w|\.\d)", text)
    normalized = [s.replace(",", "") for s in numbers]
    version, size = row["version"], str(row["bytes"])
    measured = any(
        e["call"] == {"name": "file_bytes", "arguments": {"path": "uv.lock"}}
        and e["observation"] == {"bytes": row["bytes"]}
        for e in executions({"turns": result["turns"][:-1]})
    )
    complete = result["status"] == "answered"
    return {
        "version": version in normalized,
        "size": size in normalized,
        "measured": measured,
        "complete": complete,
        "grounded": complete
        and measured
        and sorted(normalized) == sorted([version, size]),
    }


def decide(spec, data, runs):
    rows = []
    assert len(runs) == len(data["rows"]) == 2
    sides = ("recurrent", "base")
    for w, pair in zip(data["rows"], runs, strict=True):
        user = request(
            gzip.decompress(
                (SOURCE / (w["revision"] + ".lock.gz")).read_bytes()
            ).decode()
        )
        assert (
            pair["initial_prompt_identical"]
            and pair["prefill_cache_state"]
            and pair["base_revision"] == REVISION
        )
        assert pair["base_source"] == "certified_release_text" and pair[
            "order"
        ] == list(sides)
        assert (
            pair["max_tokens_per_turn"] == spec["max_tokens"]
            and pair["max_turns"] == spec["max_turns"]
        )
        assert (
            pair["runs"]["recurrent"][0]["turns"][0]["prompt_ids"]
            == pair["runs"]["base"][0]["turns"][0]["prompt_ids"]
        )
        for side in sides:
            (q,) = pair["runs"][side]
            turns = q["turns"]
            initial = turns[0]["prompt_ids"]
            assert (
                q["question"] == user
                and q["initial_tokens"] == len(initial) == w["prompt_tokens"] >= 131072
            )
            assert q["initial_sha256"] == fingerprint(initial) == w["prompt_sha256"]
            sites = {(layer, 0) for layer in range(3, 24, 4)}
            if side == "recurrent":
                sites.add((23, 1))
            for t in turns:
                hist = t["kv_history"]
                n = len(t["prompt_ids"]) + len(t["token_ids"]) - 1
                assert {(h["layer"], h["turn"]) for h in hist} == sites
                assert all(h["offset"] == h["retained"] == n for h in hist)
                assert (
                    t.get("cache_state_blocks", 0)
                    == (len(t["prompt_ids"]) - t.get("reused_prompt_tokens", 0) - 1)
                    // 128
                )
            assert spec["limits"]["peak_gib"] * 2**30 >= q["peak_mlx_bytes"] > 0
            assert 0 < q["first_token_seconds"] < q["request_seconds"]
            rows.append(
                dict(
                    revision=w["revision"],
                    side=side,
                    **score(w, q),
                    answer=q["answer"],
                    status=q["status"],
                    calls=executions(q),
                    first_s=q["first_token_seconds"],
                    wall_s=q["request_seconds"],
                    peak=q["peak_mlx_bytes"],
                    initial_tokens=len(initial),
                    prefill_s=turns[0]["ttft_s"],
                    sites=len(sites),
                    generated=q["output_tokens"],
                    decode_tokens=sum(len(t["token_ids"]) - 1 for t in turns),
                    decode_s=sum(t["seconds"] - t["ttft_s"] for t in turns),
                    reuse=sum(t.get("reused_prompt_tokens", 0) for t in turns),
                )
            )
    totals = {}
    for side in sides:
        grp = [r for r in rows if r["side"] == side]
        totals[side] = {
            k: sum(r[k] for r in grp)
            for k in (
                "version",
                "size",
                "measured",
                "complete",
                "grounded",
                "first_s",
                "wall_s",
                "generated",
                "reuse",
            )
        }
        totals[side].update(
            peak=max(r["peak"] for r in grp),
            prefill_tps=sum(r["initial_tokens"] for r in grp)
            / sum(r["prefill_s"] for r in grp),
            decode_tps=sum(r["decode_tokens"] for r in grp)
            / sum(r["decode_s"] for r in grp),
        )
    a, b = (totals[k] for k in sides)
    gates = {"accuracy": a["grounded"] == 2 and b["grounded"] < 2}
    gates.update({k: a[k] < b[k] for k in ("wall_s", "first_s", "peak")})
    gates.update({k: a[k] > b[k] for k in ("prefill_tps", "decode_tps")})
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "totals": totals,
        "rows": rows,
    }


def verify(root, decision):
    spec = registered(NAME)
    data = material()
    assert spec == read(root / "registration.json") and data == read(spec["material"])
    if decision.get("status") == "interrupted":
        assert decision["reason"] and not decision["passed"]
        return
    runs = unpack(root / "requests.json.gz")
    assert decide(spec, data, runs) == decision
    tok = tokenizer(spec)
    for row, pair in zip(data["rows"], runs, strict=True):
        original = gzip.decompress(
            (SOURCE / (row["revision"] + ".lock.gz")).read_bytes()
        )
        with TemporaryDirectory() as tmp, chdir(tmp):
            Path("uv.lock").write_bytes(original)
            callbacks = file_tools(["uv.lock"])
            for side, arm in (("recurrent", "candidate"), ("base", "control")):
                (q,) = pair["runs"][side]
                replay(
                    tok,
                    row,
                    callbacks,
                    [{"role": "user", "content": request(original.decode())}],
                    {
                        "trajectory": q,
                        "wall_s": q["request_seconds"],
                        "first_s": q["first_token_seconds"],
                        "peak": q["peak_mlx_bytes"],
                    },
                    spec,
                    arm,
                )


def execute(s, row, work, site, budget):
    raw = gzip.decompress((SOURCE / (row["revision"] + ".lock.gz")).read_bytes())
    (work / "uv.lock").write_bytes(raw)
    (work / "request.txt").write_text(request(raw.decode()))
    args = [
        sys.executable,
        "-I",
        "-c",
        GUARD + CODE.replace("main()\n", HOOK + "\nmain()\n"),
        str(int(s["limits"]["peak_gib"] * 2**30)),
        str(site),
        "ask",
        "--checkpoint",
        str(ROOT / s["checkpoint"]),
        "--trace",
        "--compare-base",
        "--cache-state-prefill",
        "--retain-conversation-prefix",
        "--tool-format",
        "qwen",
        "--file",
        "uv.lock",
        "--prompt-file",
        "request.txt",
        "--max-tokens",
        str(s["max_tokens"]),
        "--max-turns",
        str(s["max_turns"]),
    ]
    lease = child_lease()
    cache = work / "empty-cache"
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
                budget.check()
                try:
                    out, err = child.communicate(timeout=0.7)
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()
    if child.returncode:
        detail = err.decode()[-4000:]
        if "MemoryError: Long-context child MLX peak" in detail:
            raise MemoryError(detail.splitlines()[-1])
        raise RuntimeError(detail)
    return {**json.loads(out), "paired_cold_s": perf_counter() - start}


def run():
    s = registered(NAME)
    assert (
        digest(Path(s["requires"]) / "checksums.json") == s["prerequisite_sha256"]
        and read(Path(s["requires"]) / "decision.json")["passed"]
    )
    data = material()
    assert data == read(s["material"])
    wheel = ROOT / s["wheel"]
    assert digest(wheel) == s["wheel_sha256"]
    root = ROOT / "outputs/research" / NAME
    root.mkdir(exist_ok=True)
    if (root / "requests.json.gz").exists() or (root / "decision.json").exists():
        raise ValueError("Measurements already exist")
    save(root / "registration.json", s)
    capture(
        root,
        (
            "scripts/locked_context_study.py",
            "scripts/locked_context_material.py",
            "scripts/verify_runtime_wheel.py",
        ),
    )
    budget = Budget(NAME, s["limits"])
    try:
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            site = work / "installed"
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--target",
                    str(site),
                    "--offline",
                    "--no-deps",
                    str(wheel),
                ],
                check=True,
            )
            runs = []
            for row in data["rows"]:
                print(
                    "Opening complete revision",
                    row["revision"],
                    row["prompt_tokens"],
                    flush=True,
                )
                runs.append(execute(s, row, work, site, budget))
                packed(root / "requests.json.gz", runs)
            decision = decide(s, data, runs)
    except (TimeoutError, MemoryError, InterruptedError) as e:
        decision = {"passed": False, "status": "interrupted", "reason": str(e)}
    finally:
        budget.close()
    save(root / "decision.json", decision)
    verify(root, decision)


if __name__ == "__main__":
    run()
