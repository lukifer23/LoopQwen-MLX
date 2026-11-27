"""One direct product comparison with intact natural tables, original instructions and retention."""

import gc
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from scripts.conditional_panel import FETA, NAME
from scripts.grounded_table_sentences import in_text, norm

from ..conditional_reentry import ConditionalAssistant
from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from ..relational_tools import tools
from ..tool_compare import DenseAssistant
from . import table_policy_primary as earlier
from .conditional_features import CODES
from .conditional_fit import NAME as FIT
from .core import Budget
from .cross_view_study import tokenizer
from .file_evidence_environment import capture
from .native_function_study import evaluate, executions, replay
from .registered import registered
from .table_reentry_fit import root_for
from .table_worlds import FILENAME, mount

FRESH = {"natural_mounted", "natural_inline", "instruction"}
ARMS = ("candidate", "control")


def providers(w, folder):
    if w["family"] == "natural_mounted":
        return tools([folder / w["source"] / FILENAME])
    if w["family"] in FRESH:
        return []
    return earlier.providers(w, folder)


def configure(panel, folder):
    from scripts.program_language_material import PANEL

    assert panel["original_panel_sha256"] == digest(PANEL)
    earlier.mounted(read(PANEL), folder)
    for r in read(FETA / "originals.json")["test"]:
        mount(folder / ("feta-" + str(r["id"])), r["rows"])


def score(tok, w, i, tr):
    if w["family"] not in FRESH:
        return earlier.score(tok, w, i, tr)
    completed = tr["status"] == "answered"
    if w["family"] == "instruction":
        return {"completed": completed, "quality": "masked"}
    ans = norm(tr["answer"] or "")
    reads = [
        o["observation"]
        for o in executions(tr)
        if o["call"]["name"] == "database_query"
        and o["observation"].get("referenced_tables") == ["records"]
        and not o["observation"].get("truncated", True)
        and o["observation"].get("rows")
    ]
    returned = [
        norm(str(value))
        for obj in reads
        for row in obj["rows"]
        for value in row
        if value is not None
    ]
    supported = sum(
        any(in_text(atom, cell) for cell in returned) for atom in w["attested"]
    )
    n = sum(in_text(v, ans) for v in w["attested"])
    return {
        "completed": completed,
        "read": bool(reads),
        "observed_attested": supported,
        "supported_read": supported >= 2,
        "attested_mentions": n,
        "attested_total": len(w["attested"]),
        "all_attested": n == len(w["attested"]),
        "quality": "masked",
    }


def costs(allq):
    turns = [t for q in allq for t in q["trajectory"]["turns"]]
    return {
        "first_s": sum(q["first_s"] for q in allq),
        "wall_s": sum(q["wall_s"] for q in allq),
        "peak": max(q["peak"] for q in allq),
        "output": sum(len(t["token_ids"]) for t in turns),
        "opening_tps": sum(
            len(t["prompt_ids"]) - t.get("reused_prompt_tokens", 0) for t in turns
        )
        / sum(t["ttft_s"] for t in turns),
        "decode_tps": sum(len(t["token_ids"]) - 1 for t in turns)
        / sum(t["seconds"] - t["ttft_s"] for t in turns),
    }


def outcome(panel, runs):
    rows = panel["rows"]
    legacy = [w for w in rows if w["family"] not in FRESH]
    old = {a: [r for r in rr if r["family"] not in FRESH] for a, rr in runs.items()}
    ret = earlier.decision({"rows": legacy}, old)
    assert len(legacy) == 268
    aggregate = {}
    fresh = {}
    for arm in ARMS:
        rr = runs[arm]
        assert [r["id"] for r in rr] == [w["id"] for w in rows]
        allq = [r["requests"][0] for r in rr]
        aggregate[arm] = costs(allq)
        summary = {}
        for f in sorted(FRESH):
            q = [r["requests"][0]["score"] for r in rr if r["family"] == f]
            assert len(q) == 24
            summary[f] = {"completed": sum(row["completed"] for row in q)}
            if f != "instruction":
                summary[f].update(
                    {
                        k: sum(r[k] for r in q)
                        for k in (
                            "attested_mentions",
                            "attested_total",
                            "all_attested",
                            "observed_attested",
                            "supported_read",
                            "read",
                        )
                    }
                )
        fresh[arm] = summary
    a, b = (aggregate[k] for k in ARMS)
    gates = {
        "retained_268": ret["passed"],
        "fresh_completion": all(
            fresh["candidate"][f]["completed"] >= fresh["control"][f]["completed"]
            for f in FRESH
        ),
        "mounted_read": fresh["candidate"]["natural_mounted"]["read"]
        > fresh["control"]["natural_mounted"]["read"],
    }
    gates.update({k: a[k] < b[k] for k in ("first_s", "wall_s", "peak")})
    gates.update({k: a[k] > b[k] for k in ("opening_tps", "decode_tps")})
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "retention": ret,
        "cost": aggregate,
        "fresh": fresh,
        "additional_prerequisite": "Sealed masked natural-language, database, changed-table and instruction quality.",
    }


def verify(root, d):
    if d.get("status") == "interrupted":
        assert not d["passed"] and d["reason"]
        return
    s = registered(NAME)
    assert s == read(root / "registration.json")
    parent = Path("research/results") / FIT
    assert read(root / "training-parent.json") == {
        "weights": digest(parent / "readout.safetensors"),
        "archive": digest(parent / "checksums.json"),
    }
    tok = tokenizer(s)
    panel = read(s["material"])
    runs = {}
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        configure(panel, folder)
        for side in ARMS:
            runs[side] = unpack(root / (side + ".json.gz"))
            for w, line in zip(panel["rows"], runs[side], strict=True):
                (q,) = line["requests"]
                assert q["trajectory"]["turns"][0]["prompt_ids"] == w["initial_ids"]
                replay(
                    tok,
                    w,
                    providers(w, folder),
                    [{"role": "user", "content": w["questions"][0]}],
                    q,
                    s,
                    side,
                )
                assert score(tok, w, 0, q["trajectory"]) == q["score"]
    assert outcome(panel, runs) == d


def run():
    s = registered(NAME)
    parent = Path("research/results") / FIT
    if not read(parent / "decision.json")["passed"]:
        raise ValueError("The whole-response and SQL prerequisite missed its gate")
    root = root_for(s)
    capture(
        root,
        (
            *CODES,
            "loopqwen/research/conditional_fit.py",
            "loopqwen/research/conditional_primary.py",
            "loopqwen/research/native_function_study.py",
            "loopqwen/evaluation/generation_eval.py",
            "scripts/conditional_panel.py",
        ),
    )
    guard = Budget(NAME, s["limits"])
    try:
        tok = tokenizer(s)
        panel = read(s["material"])
        runs = {}
        save(
            root / "training-parent.json",
            {
                "weights": digest(parent / "readout.safetensors"),
                "archive": digest(parent / "checksums.json"),
            },
        )

        def control():
            actor = DenseAssistant(True, "qwen", s["checkpoint"])
            actor.native_plain = True
            return actor

        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            configure(panel, folder)
            for side, factory in (
                ("candidate", lambda: ConditionalAssistant(s["checkpoint"], parent)),
                ("control", control),
            ):
                path = root / (side + ".json.gz")
                if not path.exists():
                    gc.collect()
                    mx.clear_cache()
                    packed(
                        path,
                        evaluate(
                            s,
                            panel,
                            side,
                            guard,
                            factory,
                            lambda w: providers(w, folder),
                            scorer=lambda w, i, tr: score(tok, w, i, tr),
                        ),
                    )
                runs[side] = unpack(path)
        d = outcome(panel, runs)
    except (TimeoutError, MemoryError, InterruptedError) as e:
        d = {"passed": False, "status": "interrupted", "reason": str(e)}
    finally:
        guard.close()
    save(root / "decision.json", d)


if __name__ == "__main__":
    run()
