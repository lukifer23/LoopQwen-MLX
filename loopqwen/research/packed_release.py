"""Audit a deployable packed-register policy on full books, exact replay and live tools."""

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from mlx_lm.utils import load_tokenizer

from ..evaluation.answers import numeric
from ..evaluation.language7 import interpret
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..model_loader import local_snapshot
from ..worker import child_lease
from .core import Budget
from .register_allocation import packed_evidence
from .registered import registered

SIDES = ("candidate", "control")
SITES = {(i, 0) for i in (3, 7, 11, 15, 19, 23)}
REGISTER = SITES | {(11, 1), (15, 1)}


def outcome(data, mat, spec):
    from scripts.tool_release import field_score, tokens

    tok = load_tokenizer(local_snapshot())
    end_ids = {*tok.eos_token_ids, tok.convert_tokens_to_ids("<|im_end|>")}
    evidence = True
    short = tool = correct = exact_books = 0
    summary = {
        s: {
            "register_seconds": 0,
            "register_ttft": 0,
            "register_peak": 0,
            "book_seconds": 0,
            "book_peak": 0,
            "book_nll": [],
            "book_top1": [],
            "cold": 0,
        }
        for s in SIDES
    }
    for i, pair in enumerate(zip(*(data[s] for s in SIDES), strict=True)):
        expected = mat["register"]["panel"][i]
        books = [r["book"] for r in pair]
        exact_books += (
            books[0]["nll"] == books[1]["nll"]
            and books[0]["predicted"] == books[1]["predicted"]
        )
        package = ("mlx", "numpy")[i]
        tool += (
            all(
                field_score(r["tool"], package, spec["packages"][package]) for r in pair
            )
            and tokens(pair[0]["tool"]) == tokens(pair[1]["tool"])
            and pair[0]["tool"]["status"] == pair[1]["tool"]["status"]
        )
        for s, r in zip(SIDES, pair, strict=True):
            total = summary[s]
            total["cold"] += r["cold"]
            b = r["book"]
            fixed = mat["books"][i]
            n = len(fixed["prompt_ids"])
            assert (b["id"], b["input_sha256"], b["tokens"]) == (
                fixed["id"],
                fixed["sha256"],
                n,
            )
            assert (
                b["route"] == "ordinary" and len(b["nll"]) == len(b["predicted"]) == 256
            )
            assert all(math.isfinite(x) and x >= 0 for x in b["nll"])
            assert {tuple(x[:2]) for x in b["retained"]} == SITES and all(
                x[2:] == [n - 1, n - 1] for x in b["retained"]
            )
            total["book_nll"].append(sum(b["nll"]) / 256)
            total["book_top1"].append(
                sum(
                    x == y
                    for x, y in zip(
                        b["predicted"], fixed["prompt_ids"][-256:], strict=True
                    )
                )
            )
            total["book_seconds"] += b["seconds"]
            total["book_peak"] = max(b["peak"], total["book_peak"])
            z = r["register"]
            ids = z["token_ids"]
            assert z["input_sha256"] == fingerprint(expected["prompt_ids"]) and z[
                "prompt_tokens"
            ] == len(expected["prompt_ids"])
            consumed = z["prompt_tokens"] + len(ids) - 1
            section = z["attention_memory"]
            evidence &= (
                z["context_route"] == "register"
                and section["start"]
                <= expected["gold_start"]
                < expected["gold_stop"]
                <= section["stop"]
            )
            evidence &= {
                (v["layer"], v["turn"]) for v in z["kv_history"]
            } == REGISTER and all(
                v["retained"] == v["offset"] == consumed for v in z["kv_history"]
            )
            if s == "candidate":
                sites = z.get("page_sites", [])
                evidence &= (
                    z.get("prefill_policy") == "direct-packed-register-v1"
                    and {(v["layer"], v["turn"]) for v in sites} == REGISTER
                )
                evidence &= all(
                    v["sparse_blocks"] > 0
                    and v["possible"] == consumed * (consumed + 1) // 2
                    and 0 < v["attended"] < v["possible"]
                    and v["offset"] == v["retained"] == consumed
                    and packed_evidence(v, consumed, capacity=z["prompt_tokens"] + 95)
                    for v in sites
                )
            else:
                evidence &= "prefill_policy" not in z and "page_sites" not in z
            assert 0 < z["ttft_s"] <= z["elapsed_s"] and 0 < len(ids) <= 96
            total["register_seconds"] += z["elapsed_s"]
            total["register_ttft"] += z["ttft_s"]
            total["register_peak"] = max(z["peak_bytes"], total["register_peak"])
        for x, y, w in zip(
            pair[0]["short"],
            pair[1]["short"],
            mat["short"][i * 6 : (i + 1) * 6],
            strict=True,
        ):
            assert x["id"] == y["id"] == w["id"] and x["input_sha256"] == y[
                "input_sha256"
            ] == fingerprint(w["prompt_ids"])
            short += (
                x["token_ids"] == y["token_ids"]
                and x["context_route"] == y["context_route"]
                and "prefill_policy" not in x
            )
        z, y = (r["register"] for r in pair)
        parsed = interpret(
            z["token_ids"],
            tok.convert_tokens_to_ids("<think>"),
            tok.convert_tokens_to_ids("</think>"),
            end_ids,
            tok,
        )
        correct += (
            z["token_ids"] == y["token_ids"]
            and z["token_ids"][-1] in end_ids
            and numeric(parsed["visible"]) == expected["gold"]
        )
    a, b = (summary[s] for s in SIDES)
    return {
        "passed": bool(
            evidence
            and exact_books == 2
            and short == 12
            and tool == 2
            and correct == 2
            and a["register_seconds"] <= spec["time_ratio"] * b["register_seconds"]
            and a["register_peak"] < b["register_peak"]
        ),
        "evidence": bool(evidence),
        "books_exact": exact_books,
        "short_exact": short,
        "tool_exact": tool,
        "register_exact_correct": correct,
        "metrics": summary,
    }


def verify(root, decision):
    spec = read(root / "registration.json")
    mat = read(spec["material"])
    assert all(fingerprint(w["prompt_ids"]) == w["sha256"] for w in mat["books"])
    data = {s: unpack(root / (s + ".json.gz")) for s in SIDES}
    assert all(len(rs) == 2 for rs in data.values())
    parent = read(Path(spec["parent"]) / "export-files.json")
    inventory = read(root / "export-files.json")
    changed = {
        k for k in inventory.keys() | parent.keys() if inventory.get(k) != parent.get(k)
    }
    assert "config.json" in changed and changed <= {
        "README.md",
        "config.json",
        "runtime/page_prefill.py",
        "runtime/export_runtime.py",
        "runtime/evaluation/generation_eval.py",
    }
    assert outcome(data, mat, spec) == decision


def run(name):
    from scripts.export_packed_register import build

    spec = registered(name)
    mat = read(spec["material"])
    out = Path("outputs/research") / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "decision.json").exists():
        raise FileExistsError(out)
    full, parent = (Path(spec[k]) for k in ("full", "parent"))
    reg = read("research/registry.json")["experiments"]
    for p in (full, parent):
        if (
            digest(p / "checksums.json") != reg[p.name]["archive"]
            or digest(p / "registration.json") != spec["requirements"][p.name]
            or not read(p / "decision.json")["passed"]
        ):
            raise ValueError("Unqualified prerequisite: " + p.name)
    baseline = Path(read(parent / "registration.json")["destination"])
    staged = Path(spec["staging"])
    destination = Path(spec["destination"])
    if destination.exists():
        ready_path = out / "release-ready.json"
        if not ready_path.exists():
            raise FileExistsError(destination)
        ready = read(ready_path)
        inventory = read(out / "export-files.json")
        data = {s: unpack(out / (s + ".json.gz")) for s in SIDES}
        if (
            ready != outcome(data, mat, spec)
            or not ready["passed"]
            or inventory != read(destination / "tool-export-files.json")
            or any(digest(destination / k) != h for k, h in inventory.items())
        ):
            raise ValueError(
                "Existing destination differs from the sealed passing crossover"
            )
        save(out / "decision.json", ready)
        return
    staged.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    budget = Budget(name, spec["limits"])
    try:
        inventory = build(baseline, parent, staged, full)
        save(out / "export-files.json", inventory)
        runs = {s: [] for s in SIDES}
        for i in range(2):
            for s in SIDES if i == 0 else reversed(SIDES):
                result = out / (s + f"-{i}.json")
                if not result.is_file():
                    bundle = staged if s == "candidate" else baseline
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp = Path(tmp)
                        opt = child_lease()
                        opt["env"].update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                        cmd = [
                            sys.executable,
                            "-I",
                            str(Path("scripts/packed_release_worker.py").resolve()),
                            str(bundle.resolve()),
                            str(Path(spec["material"]).resolve()),
                            str(i),
                            str(tmp / "result.json"),
                            str(spec["limits"]["peak_gib"]),
                        ]
                        with subprocess.Popen(cmd, cwd=tmp, **opt) as child:
                            try:
                                while child.poll() is None:
                                    try:
                                        child.wait(timeout=1)
                                    except subprocess.TimeoutExpired:
                                        budget.check()
                            except BaseException:
                                child.terminate()
                                raise
                            if child.returncode:
                                raise subprocess.CalledProcessError(
                                    child.returncode, cmd
                                )
                        save(result, read(tmp / "result.json"))
                runs[s].append(read(result))
                print(s, i, runs[s][-1]["register"]["elapsed_s"], flush=True)
        for s in SIDES:
            packed(out / (s + ".json.gz"), runs[s])
        decision = outcome(runs, mat, spec)
        if decision["passed"]:
            save(out / "release-ready.json", decision)
            staged.rename(destination)
        save(out / "decision.json", decision)
    finally:
        budget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    run(parser.parse_args().name)
