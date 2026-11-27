"""Measure extending the native per-request snapshot through actual observed function returns."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.io import digest, read, save
from loopqwen.research.core import Budget
from loopqwen.research.file_evidence_environment import capture
from loopqwen.research.registered import registered
from scripts.cache_state_conversations import execute, fixtures, summary
from scripts.verify_runtime_wheel import ROOT, build, trace

NAME = "extended-tool-prefix-01"


def decision(s, rounds):
    assert [r["mode"] for r in rounds] == s["order"]
    original = unpack(Path(s["requires"]) / "rounds.json.gz")[0]["items"]
    totals = {
        m: summary([item for r in rounds if r["mode"] == m for item in r["items"]])
        for m in set(s["order"])
    }
    parity = blocks = True
    saved = {}
    for block in rounds:
        assert len(original) == len(block["items"]) == len(s["cases"])
        for first, item in zip(original, block["items"], strict=True):
            assert (
                first["id"] == item["id"] and 0 < item["cold_first_s"] < item["cold_s"]
            )
            for before, now in zip(first["exchanges"], item["exchanges"], strict=True):
                parity &= trace(now) == trace(before)
                for x, y in zip(before["turns"], now["turns"], strict=True):
                    n = y.get("reused_prompt_tokens", 0)
                    prior = x.get("reused_prompt_tokens", 0)
                    blocks &= (
                        y["cache_state_blocks"] == (len(y["prompt_ids"]) - n - 1) // 128
                    )
                    if block["mode"] == "fixed":
                        blocks &= n == prior
                    else:
                        blocks &= n >= prior
                        saved[item["id"]] = saved.get(item["id"], 0) + n - prior
    paired = []
    for a, b in zip(rounds[::2], rounds[1::2], strict=True):
        vals = {v["mode"]: summary(v["items"]) for v in (a, b)}
        paired.append(
            {
                k: vals["extended"][k] / vals["fixed"][k]
                for k in ("cold_s", "turn_open_s", "cold_first_s", "peak", "decode_tps")
            }
        )
    a, b = totals["extended"], totals["fixed"]
    gates = {
        "streams": bool(parity and blocks),
        "exercised": sum(saved.values()) >= 1024
        and sum(n > 0 for n in saved.values()) >= 2,
        "turn_open": a["turn_open_s"] <= 0.98 * b["turn_open_s"]
        and sum(r["turn_open_s"] < 1 for r in paired) >= 3,
        "cold_response": a["cold_s"] < b["cold_s"],
        "first": a["cold_first_s"] < b["cold_first_s"],
        "peak": a["peak"] <= b["peak"],
        "decode": a["decode_tps"] >= b["decode_tps"],
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "totals": totals,
        "paired_ratios": paired,
        "additional_tokens_by_case": saved,
    }


def verify(root, d):
    s = registered(NAME)
    assert (
        s == read(root / "registration.json")
        and digest(Path(s["requires"]) / "checksums.json") == s["prerequisite_sha256"]
    )
    if d.get("status") == "interrupted":
        assert d["reason"] and not d["passed"]
        return
    assert decision(s, unpack(root / "rounds.json.gz")) == d
    assert read(root / "wheel.json")["source"] == s["wheel_revision"]


def run():
    s = registered(NAME)
    assert (
        read(Path(s["requires"]) / "decision.json")["passed"]
        and digest(Path(s["requires"]) / "checksums.json") == s["prerequisite_sha256"]
    )
    root = Path("outputs/research") / NAME
    root.mkdir(parents=True, exist_ok=True)
    if (root / "decision.json").exists() or (root / "rounds.json.gz").exists():
        raise RuntimeError("An installed measurement already exists")
    save(root / "registration.json", s)
    capture(
        root,
        (
            "scripts/extended_tool_prefix.py",
            "scripts/cache_state_conversations.py",
            "scripts/verify_runtime_wheel.py",
        ),
    )
    budget = Budget(NAME, s["limits"])
    try:
        home = Path("exports/extended-prefix-runtime-01").resolve()
        home.mkdir(exist_ok=True)
        wheel = build(home, s["wheel_revision"])
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            assert digest(wheel) == digest(build(work, s["wheel_revision"]))
            save(
                root / "wheel.json",
                {
                    "source": s["wheel_revision"],
                    "sha256": digest(wheel),
                    "bytes": wheel.stat().st_size,
                },
            )
            site = work / "site"
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--offline",
                    "--no-deps",
                    "--target",
                    str(site),
                    str(wheel),
                ],
                check=True,
            )
            fixtures(s, work)
            rounds = []
            for i, m in enumerate(s["order"]):
                items = []
                for original in s["cases"]:
                    case = dict(original)
                    if m == "extended":
                        case["flags"] = [*case["flags"], "--extend-tool-prefix"]
                    items.append(execute(s, case, "state", site, work, budget))
                rounds.append({"mode": m, "items": items})
                packed(root / "rounds.json.gz", rounds)
                print("Installed round", i + 1, m, flush=True)
            d = decision(s, rounds)
    except (TimeoutError, MemoryError, InterruptedError) as exc:
        d = {"passed": False, "status": "interrupted", "reason": str(exc)}
    finally:
        budget.close()
    save(root / "decision.json", d)
    verify(root, d)


if __name__ == "__main__":
    run()
