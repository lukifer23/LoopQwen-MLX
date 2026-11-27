"""Direct native comparison of a readout tied to a real second visit and untouched ordinary Base."""

import gc
import re
import shutil
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
from packaging.specifiers import SpecifierSet

from .. import checkpoint
from ..evaluation.extractive import score as extract
from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from ..tool_compare import DenseAssistant
from ..visit_residual import VisitResidualAssistant, from_release
from ..wheel_tools import tools as wheel_tools
from .core import Budget
from .cross_view_study import tokenizer, write_documents
from .file_evidence_environment import capture
from .followup_study import ranges
from .linked_rollin import exact_sign_flip
from .linked_rollin_data import providers as retained_tools
from .linked_rollin_data import score as retained_score
from .native_function_study import evaluate, executions, replay, summarize
from .registered import registered
from .visit_residual_fit import fit, stream

NAME = "visit-residual-01"
NEW = ("effects", "dependency", "same_version")
ARMS = ("candidate", "control")


def tools_for(w, folder):
    if w["family"] == "effects":
        return []
    if "wheels" in w:
        from scripts.wheel_archive_sources import CACHE

        return wheel_tools([CACHE / "wheels" / n for n in w["wheels"]])
    return retained_tools(w, folder)


def score(tok, w, i, tr):
    family = w["family"]
    if family not in NEW:
        return retained_score(tok, w, i, tr)
    done = tr["status"] == "answered"
    text = (tr["answer"] or "") if done else ""
    if family == "effects":
        exact, f1 = extract(text, w["answers"])
        return {
            "success": done and exact,
            "completed": done,
            "correct_answer": done and exact,
            "f1": f1 if done else 0.0,
        }
    expected = w["current"]
    args = expected["call"]["arguments"]
    observed = any(
        e["call"].get("name") == "wheel_metadata"
        and e["call"]["arguments"].get("wheel") == args["wheel"]
        and e["call"]["arguments"].get("field", "").casefold() == "requires-dist"
        and e["observation"] == expected["observation"]
        for e in executions({"turns": tr["turns"][:-1]})
    )
    correct_range = ranges(text) == {str(SpecifierSet(w["requirement"]))}
    verdict = set(re.findall(r"\b(?:yes|no)\b", text.casefold())) == {
        "yes" if w["allowed"] else "no"
    }
    return {
        "success": done and correct_range and verdict and observed,
        "completed": done,
        "correct_answer": done and correct_range and verdict,
        "observed": observed,
        "range": correct_range,
        "verdict": verdict,
    }


def outcome(panel, runs):
    arms = {}
    byid = {}
    for arm in ARMS:
        records = runs[arm]
        assert [w["id"] for w in panel["rows"]] == [w["id"] for w in records]
        fam = defaultdict(list)
        whole = defaultdict(lambda: defaultdict(list))
        byid[arm] = {}
        for w, r in zip(panel["rows"], records, strict=True):
            q = r["requests"][0]
            s = q["score"]
            byid[arm][w["id"]] = s
            fam[w["family"]].append(r)
            whole[w["family"]][w["cluster"]].append(s["success"])
        turns = [
            t for r in records for q in r["requests"] for t in q["trajectory"]["turns"]
        ]
        arms[arm] = {
            "all": summarize(records),
            "families": {k: summarize(v) for k, v in fam.items()},
            "whole": {
                k: sum(all(bits) for bits in vv.values()) for k, vv in whole.items()
            },
            "effects_f1": sum(r["requests"][0]["score"]["f1"] for r in fam["effects"])
            / len(fam["effects"]),
            "opening_tps": sum(
                len(t["prompt_ids"]) - t.get("reused_prompt_tokens", 0) for t in turns
            )
            / sum(t["ttft_s"] for t in turns),
            "decode_tps": sum(len(t["token_ids"]) - 1 for t in turns)
            / sum(t["seconds"] - t["ttft_s"] for t in turns),
        }
    a, b = [arms[k] for k in ARMS]
    losses = []
    diff = defaultdict(int)
    for w in panel["rows"]:
        x, y = [int(byid[n][w["id"]]["success"]) for n in ARMS]
        if w["family"] == "effects":
            diff[w["cluster"]] += x - y
        elif w["family"] not in NEW and y > x:
            losses.append(w["id"])
    p = exact_sign_flip(list(diff.values()))
    gates = {
        "human_questions": sum(diff.values()) >= 8 and p <= 0.05,
        "whole_situations": a["whole"]["effects"] >= max(6, b["whole"]["effects"] + 2),
        "human_f1": a["effects_f1"] >= b["effects_f1"] + 0.08,
        "dependency": a["whole"]["dependency"] >= max(4, b["whole"]["dependency"] + 1),
        "same_version": a["whole"]["same_version"]
        >= max(2, b["whole"]["same_version"] + 1),
        "prior_base": not losses,
        "prior_pairs": all(
            a["whole"][k] >= max(2, b["whole"][k] + 1) for k in ("wheel", "agenda")
        ),
        "completion": all(
            a["families"][k]["completed"] >= b["families"][k]["completed"]
            for k in a["families"]
        ),
    }
    gates.update({k: a["all"][k] < b["all"][k] for k in ("first_s", "wall_s", "peak")})
    gates.update({k: a[k] > b[k] for k in ("opening_tps", "decode_tps")})
    return {
        "passed": all(gates.values()),
        "arms": arms,
        "gates": gates,
        "old_base_lost": losses,
        "human_source_differences": dict(diff),
        "paired_p": p,
    }


def verify(root, d):
    s = registered(NAME)
    assert read(root / "registration.json") == s
    if d.get("status") == "interrupted":
        assert not d["passed"] and d["reason"]
        return
    f = read(root / "fit.json")
    views = read(s["training"])["examples"]
    panel = read(s["material"])
    tok = tokenizer(s)
    assert (
        f["trainable"] == 3 * 1024 * s["residual_rank"]
        and f["zero_without_change"]
        and f["training"] == digest(Path(s["training"]))
    )
    assert (
        f["weights"] == digest(root / "visit-residual.safetensors")
        and digest(root / "training-features.json.gz") == f["feature_sha256"]
    )
    features = unpack(root / "training-features.json.gz")
    steps = unpack(root / "updates.json.gz")
    assert [v["id"] for v in views] == [v["id"] for v in features] and len(steps) == f[
        "updates"
    ] == s["updates"]
    draw = stream(views, s["fit_seed"])
    assert sum(z["sampled"] for z in steps) == f["targets"]
    for step, record in enumerate(steps, 1):
        indices = next(draw)
        assert record["views"] == indices and record["step"] == step
        assert record["sampled"] == sum(
            min(features[i]["targets"], s["positions_per_view"]) for i in indices
        )
        assert record["contrasts"] == sum(
            features[i]["contrast"] is not None for i in indices
        )
    parity = read(root / "native-parity.json")
    assert (
        parity["all_logits_exact"]
        and parity["vocabulary"] == 248320
        and parity["different_visits"]
    )
    assert read(root / "optimizer-state.json")["step"] == s["updates"]
    runs = {a: unpack(root / (a + ".json.gz")) for a in ARMS}
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        write_documents(panel, folder)
        for a, records in runs.items():
            for w, r in zip(panel["rows"], records, strict=True):
                q = r["requests"][0]
                tr = q["trajectory"]
                assert tr["turns"][0]["prompt_ids"] == w["initial_ids"]
                replay(
                    tok,
                    w,
                    tools_for(w, folder),
                    [{"role": "user", "content": w["questions"][0]}],
                    q,
                    s,
                    a,
                )
                assert score(tok, w, 0, tr) == q["score"]
    assert outcome(panel, runs) == d


def run():
    from scripts.visit_residual_material import build
    from scripts.visit_residual_oracle import audit

    s = registered(NAME)
    root = Path("outputs/research") / NAME
    root.mkdir(parents=True, exist_ok=True)
    assert (
        read(Path(s["requires"]) / "decision.json")["passed"]
        and digest(Path(s["requires"]) / "checksums.json") == s["prerequisite_sha256"]
    )
    previous = read(Path("research/results") / s["deferred_after"] / "decision.json")
    if previous["passed"] or "arms" not in previous:
        raise RuntimeError(
            "The previous quality primary has its own unresolved prerequisite"
        )
    if (root / "decision.json").exists():
        raise FileExistsError("Primary is closed")
    save(root / "registration.json", s)
    capture(
        root,
        (
            "loopqwen/visit_residual.py",
            "loopqwen/research/visit_residual_fit.py",
            "loopqwen/research/visit_residual_study.py",
            "scripts/visit_residual_material.py",
            "scripts/visit_residual_oracle.py",
            "loopqwen/research/native_function_study.py",
            "loopqwen/tool_model.py",
            "loopqwen/tool_session.py",
            "loopqwen/looped_model.py",
            "pyproject.toml",
            "uv.lock",
        ),
    )
    budget = Budget(NAME, s["limits"])
    try:
        build()
        audit()
        from scripts.visit_residual_oracle import RECEIPT

        assert digest(Path(RECEIPT)) == s["source_oracle_sha256"]
        shutil.copyfile(RECEIPT, root / "source-qualification.json")
        budget.check()
        tok = tokenizer(s)
        panel = read(s["material"])
        if not (root / "fit.json").exists():
            net = from_release(s["checkpoint"], s)
            fit(net, tok, s, root, budget)
            del net
        latest = checkpoint.location(Path("checkpoints/research") / NAME)
        shutil.copyfile(
            latest / "optimizer.safetensors", root / "optimizer.safetensors"
        )
        shutil.copyfile(latest / "state.json", root / "optimizer-state.json")
        gc.collect()
        mx.clear_cache()
        budget.check()

        def base():
            actor = DenseAssistant(True, "qwen", s["checkpoint"])
            actor.native_plain = True
            return actor

        runs = {}
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_documents(panel, folder)
            for arm, maker in (
                ("candidate", lambda: VisitResidualAssistant(s["checkpoint"], root)),
                ("control", base),
            ):
                path = root / (arm + ".json.gz")
                if not path.exists():
                    gc.collect()
                    mx.clear_cache()
                    rows = evaluate(
                        s,
                        panel,
                        arm,
                        budget,
                        maker,
                        lambda w: tools_for(w, folder),
                        scorer=lambda w, i, tr: score(tok, w, i, tr),
                    )
                    packed(path, rows)
                runs[arm] = unpack(path)
        d = outcome(panel, runs)
    except (TimeoutError, MemoryError, InterruptedError) as e:
        d = {"passed": False, "status": "interrupted", "reason": str(e)}
    finally:
        budget.close()
    save(root / "decision.json", d)
    print("Closed second-visit comparison", flush=True)


if __name__ == "__main__":
    run()
