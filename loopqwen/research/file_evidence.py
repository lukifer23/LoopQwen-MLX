"""Test whether a freely selected lookup and a counterfactual file can change the answer."""

import gc
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from ..csv_tools import tools
from ..evaluation.mechanism import packed, unpack
from ..evaluation.relations import conclusion
from ..evaluation.statistics import exact_mcnemar
from ..file_evidence import FileEvidenceAssistant, model_for
from ..io import digest, read, save
from ..tool_compare import DenseAssistant
from .core import Budget
from .native_function_study import evaluate, executions, fit_messages, replay, summarize
from .registered import registered

NAME = "file-evidence-01"


def catalog(row, folder):
    if row["family"] == "inline":
        return []
    path = Path(folder) / row["id"] / "clippings.csv"
    return tools([path], keyed=True)


def write_files(rows, root):
    for row in rows:
        if row["family"] == "file":
            folder = Path(root) / row["id"]
            folder.mkdir()
            (folder / "clippings.csv").write_bytes(row["file"].encode())


def grade(row, i, tr):
    assert i == 0
    done = tr["status"] == "answered"
    choice = conclusion(tr["answer"] or "") if done else None
    events = executions(tr)
    correct = choice == row["gold"]
    observed = any(
        e["call"] in row["calls"] and e["observation"] == {"value": row["evidence"]}
        for e in events
    )
    return {
        "completed": done,
        "answer": choice,
        "correct_answer": correct,
        "observed": observed,
        "success": correct and (row["family"] == "inline" or observed),
    }


def aggregate(rows, runs):
    groups = {}
    pairs = {}
    for side, records in runs.items():
        assert [r["id"] for r in records] == [r["id"] for r in rows]
        groups[side] = {
            k: summarize([r for r in records if k == "all" or r["family"] == k])
            for k in ("all", "inline", "file")
        }
        times = [
            t for r in records for q in r["requests"] for t in q["trajectory"]["turns"]
        ]
        groups[side]["opening_tps"] = (
            sum(len(w["initial_ids"]) for w in rows) / groups[side]["all"]["first_s"]
        )
        groups[side]["decode_tps"] = sum(len(t["token_ids"]) - 1 for t in times) / sum(
            t["seconds"] - t["ttft_s"] for t in times
        )
        pairs[side] = {}
        for family in ("inline", "file"):
            bits = {}
            polarities = {"yes": 0, "no": 0}
            for w, r in zip(rows, records, strict=True):
                if w["family"] == family:
                    score = r["requests"][0]["score"]
                    bits.setdefault(w["cluster"], []).append(score["success"])
                    polarities[w["gold"]] += int(score["correct_answer"])
            assert all(len(v) == 2 for v in bits.values())
            groups[side][family]["polarity_correct"] = polarities
            pairs[side][family] = {k for k, v in bits.items() if all(v)}
    c, b = groups["candidate"], groups["control"]
    matched = {}
    for family in ("file", "inline"):
        a, z = (pairs[s][family] for s in ("candidate", "control"))
        wins, losses = len(a - z), len(z - a)
        matched[family] = {
            "candidate": len(a),
            "base": len(z),
            "wins": wins,
            "losses": losses,
            "p": exact_mcnemar(wins, losses),
        }
    f = matched["file"]
    gates = {
        "file_pairs": f["wins"] - f["losses"] >= 5 and f["p"] <= 0.05,
        "inline_pairs": matched["inline"]["candidate"] > matched["inline"]["base"],
        "polarities": all(
            c[k]["polarity_correct"][g] > b[k]["polarity_correct"][g]
            for k in ("file", "inline")
            for g in ("yes", "no")
        ),
        "completion": c["all"]["completed"] >= b["all"]["completed"],
    }
    for k in ("wall_s", "first_s", "peak"):
        gates[k] = c["all"][k] < b["all"][k]
    for k in ("opening_tps", "decode_tps"):
        gates[k] = c[k] > b[k]
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "pairs": matched,
        "groups": groups,
    }


def verify(root, d):
    spec = registered(NAME)
    assert spec == read(root / "registration.json")
    if d.get("status") == "interrupted":
        assert not d["passed"] and d["reason"]
        return
    fit = read(root / "fit.json")
    assert (
        fit["updates"] == spec["updates"]
        and fit["passes"] == 2
        and fit["weights"] == digest(root / "tool-adapters.safetensors")
    )
    assert fit["training"] == digest(Path(spec["training"])) and fit["turns"] == len(
        read(spec["training"])["examples"]
    )
    from ..export_runtime import load_export_tokenizer

    release = Path(spec["checkpoint"])
    tok = load_export_tokenizer(
        release,
        eos_token_ids=[read(release / "config.json")["text_config"]["eos_token_id"]],
    )
    rows = read(spec["material"])["rows"]
    runs = {s: unpack(root / (s + ".json.gz")) for s in ("candidate", "control")}
    with TemporaryDirectory() as tmp:
        write_files(rows, tmp)
        for arm, records in runs.items():
            for w, r in zip(rows, records, strict=True):
                q = r["requests"][0]
                fs = catalog(w, tmp)
                history = [{"role": "user", "content": w["questions"][0]}]
                replay(tok, w, fs, history, q, spec, arm)
                assert q["trajectory"]["turns"][0]["prompt_ids"] == w["initial_ids"]
                assert grade(w, 0, q["trajectory"]) == q["score"]
    assert d == aggregate(rows, runs)


def run():
    s = registered(NAME)
    release = Path(s["requires"])
    assert (
        digest(release / "checksums.json") == s["prerequisite_sha256"]
        and read(release / "decision.json")["passed"]
    )
    root = Path("outputs/research") / NAME
    root.mkdir(parents=True, exist_ok=False)
    save(root / "registration.json", s)
    data = read(s["material"])
    b = Budget(NAME, s["limits"])
    try:
        m, t = model_for(s["checkpoint"], s)
        fit_messages(m, t, s, root, b)
        del m, t
        mx.synchronize()
        gc.collect()
        mx.clear_cache()

        def control():
            model = DenseAssistant(True, "qwen", s["checkpoint"])
            model.native_plain = True
            return model

        with TemporaryDirectory() as tmp:
            write_files(data["rows"], tmp)
            runs = {}
            for side in ("candidate", "control"):
                gc.collect()
                mx.clear_cache()
                actor = (
                    (lambda: FileEvidenceAssistant(s["checkpoint"], root))
                    if side == "candidate"
                    else control
                )
                runs[side] = evaluate(
                    s, data, side, b, actor, lambda w: catalog(w, tmp), scorer=grade
                )
                packed(root / (side + ".json.gz"), runs[side])
        d = aggregate(data["rows"], runs)
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        d = {"status": "interrupted", "passed": False, "reason": str(exc)}
    finally:
        b.close()
    save(root / "decision.json", d)
    print(d, flush=True)


if __name__ == "__main__":
    run()
