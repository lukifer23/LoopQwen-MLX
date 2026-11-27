"""Evaluate one shared-depth model on disjoint human judgments and actual dependencies."""

import gc
import itertools
import re
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from ..cross_view import CrossViewAssistant, compile_layer
from ..evaluation.mechanism import packed, unpack
from ..io import digest, read, save
from ..tool_compare import DenseAssistant
from ..tool_model import bind
from .core import Budget
from .cross_view_study import grade as table_grade
from .cross_view_study import providers, tokenizer, write_documents
from .file_evidence_environment import capture
from .native_function_study import evaluate, executions, fit_messages, replay, summarize
from .reasoned_evidence_fit import LABELS, fit, parse_teacher, teach, turns
from .registered import registered

NAME = "reasoned-evidence-01"


def verdict(value):
    clean = value.lower().replace("**", "").strip()
    labels = r"(true|false|uncertain|unknown|entailed|contradicted|undetermined)"
    form = (
        r"(?:the\s+)?(?:(?:final\s+)?(?:verdict|answer|judgment)(?:\s+is)?\s*[:=]?\s*|(?:this|the)\s+claim\s+is\s+)?"
        + labels
        + r"[.!\s]*"
    )
    # Accept a bare verdict or explicitly identified verdict at the end of a full reply.
    m = re.fullmatch(form, clean)
    if not m:
        m = re.search(r"(?:^|[\n.!?]\s*)" + form + "$", clean)
    if not m:
        return None
    label = m.group(1)
    return {
        "unknown": "uncertain",
        "undetermined": "uncertain",
        "entailed": "true",
        "contradicted": "false",
    }.get(label, label)


def score(tok, w, i, tr):
    if w["family"] == "retention":
        return table_grade(w, i, tr)
    assert i == 0
    completed = tr["status"] == "answered"
    answer = verdict(tr["answer"]) if completed else None
    observed = any(
        e["call"] in w["calls"] and e["observation"] == w["observation"]
        for e in executions(tr)
    )
    ids = tr["turns"][-1]["token_ids"]
    close = tok.convert_tokens_to_ids("</think>")
    words = (
        len(re.findall(r"\b\w+\b", tok.decode(ids[: ids.index(close)])))
        if close in ids
        else 0
    )
    return {
        "completed": completed,
        "verdict": answer,
        "correct_answer": completed and answer == w["gold"],
        "observed": observed,
        "success": completed
        and answer == w["gold"]
        and (w["document"] is None or observed),
        "reasoning_words": words,
    }


def outcome(panel, arms):
    results = {}
    cases = {}
    for side, rows in arms.items():
        assert [r["id"] for r in rows] == [w["id"] for w in panel["rows"]]
        groups = {
            k: summarize([r for r in rows if k == "all" or r["family"] == k])
            for k in ("all", "inline", "file", "wheel", "retention")
        }
        stories = defaultdict(list)
        wheels = defaultdict(list)
        counts = Counter()
        reasoned = Counter()
        retained = set()
        for w, r in zip(panel["rows"], rows, strict=True):
            q = r["requests"][0]["score"]
            f = w["family"]
            if f in ("file", "inline"):
                stories[w["cluster"]].append(q["success"])
                counts[w["gold"]] += q["success"]
                reasoned["completed"] += q["completed"]
                reasoned["nontrivial"] += q["completed"] and q["reasoning_words"] >= 4
            elif f == "wheel":
                wheels[w["cluster"]].append(q["success"])
            elif f == "retention" and q["success"]:
                retained.add(w["id"])
        assert (
            len(stories) == 12
            and all(len(a) == 6 for a in stories.values())
            and len(wheels) == 8
            and all(len(a) == 2 for a in wheels.values())
        )
        ts = [t for r in rows for t in r["requests"][0]["trajectory"]["turns"]]
        group = groups["all"]
        results[side] = {
            "groups": groups,
            "whole_stories": sum(all(a) for a in stories.values()),
            "wheel_pairs": sum(all(a) for a in wheels.values()),
            "by_label": dict(counts),
            "reasoning": dict(reasoned),
            "opening_tps": sum(len(w["initial_ids"]) for w in panel["rows"])
            / group["first_s"],
            "decode_tps": sum(len(t["token_ids"]) - 1 for t in ts)
            / sum(t["seconds"] - t["ttft_s"] for t in ts),
        }
        cases[side] = (stories, retained)
    a, b = results["candidate"], results["control"]
    la, lb = a["groups"], b["groups"]
    left, ra = cases["candidate"]
    right, rb = cases["control"]
    delta = [sum(left[k]) - sum(right[k]) for k in sorted(left)]
    net = sum(delta)
    obs = abs(net)
    p = sum(
        abs(sum(x * y for x, y in zip(delta, signs, strict=True))) >= obs
        for signs in itertools.product((-1, 1), repeat=len(delta))
    ) / 2 ** len(delta)
    gates = {
        "story_accuracy": net >= 12 and p <= 0.05,
        "whole_stories": a["whole_stories"] > b["whole_stories"],
        "inline": la["inline"]["grounded"] > lb["inline"]["grounded"],
        "file": la["file"]["grounded"] > lb["file"]["grounded"],
        "each_label": all(
            a["by_label"].get(k, 0) >= b["by_label"].get(k, 0) for k in LABELS
        ),
        "user_wheels": a["wheel_pairs"] >= 4 and a["wheel_pairs"] > b["wheel_pairs"],
        "table_retention": rb <= ra,
        "reasoning": a["reasoning"]["nontrivial"]
        >= max(1, a["reasoning"]["completed"] * 0.5),
        "completion": all(
            la[k]["completed"] >= lb[k]["completed"]
            for k in ("file", "inline", "wheel", "retention")
        ),
    }
    gates.update(
        {k: la["all"][k] < lb["all"][k] for k in ("wall_s", "first_s", "peak")}
    )
    gates.update({k: a[k] > b[k] for k in ("opening_tps", "decode_tps")})
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "arms": results,
        "paired_story_differences": delta,
        "net": net,
        "exact_story_p": p,
    }


def verify(
    root,
    d,
    *,
    name=NAME,
    make_views=turns,
    decision_rule=outcome,
    score_fn=score,
    tool_catalog=providers,
):
    s = registered(name)
    assert read(root / "registration.json") == s
    if (root / "teacher-fit.json").exists():
        report = read(root / "teacher-fit.json")
        generated = unpack(root / "teacher-generations.json.gz")
        examples = read(s["training"])["teacher_pool"]
        accepted = Counter()
        for w, r in zip(examples, generated, strict=True):
            assert w["id"] == r["id"]
            parsed = parse_teacher(r["text"], len(w["evidence"].splitlines()))
            assert parsed == r["parsed"] and r["accepted"] == bool(
                len(r["generated"]) < s["teacher_cap"]
                and parsed
                and parsed["label"] == w["label"]
            )
            if r["accepted"]:
                accepted[w["label"]] += 1
        assert report["accepted"] == dict(accepted) and report["targets"] == digest(
            root / "teacher-generations.json.gz"
        )
    if d.get("status") == "interrupted":
        assert not d["passed"] and d["reason"]
        return
    f = read(root / "fit.json")
    cp = read(root / "compiled.json")
    p = read(s["material"])
    tok = tokenizer(s)
    assert (
        f["weights"] == digest(root / "tool-adapters.safetensors")
        and f["updates"] == s["updates"]
        and f["explanation_targets"] > 0
    )
    assert f["views"] == digest(root / "training-views.json.gz") and unpack(
        root / "training-views.json.gz"
    ) == make_views(s, root)
    assert (
        cp["weights"] == digest(root / "cross-view.safetensors")
        and cp["adapter"] == f["weights"]
    )
    arms = {k: unpack(root / (k + ".json.gz")) for k in ("candidate", "control")}
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        write_documents(p, folder)
        for side, rows in arms.items():
            for w, r in zip(p["rows"], rows, strict=True):
                q = r["requests"][0]
                replay(
                    tok,
                    w,
                    tool_catalog(w, folder),
                    [{"role": "user", "content": w["questions"][0]}],
                    q,
                    s,
                    side,
                )
                assert q["trajectory"]["turns"][0]["prompt_ids"] == w[
                    "initial_ids"
                ] and q["score"] == score_fn(tok, w, 0, q["trajectory"])
    assert decision_rule(p, arms) == d


def run(
    *,
    name=NAME,
    qualify=teach,
    trainer=fit,
    compiler=compile_layer,
    after_compile=None,
    extra_sources=(),
    decision_rule=outcome,
    score_fn=score,
    tool_catalog=providers,
):
    s = registered(name)
    root = Path("outputs/research") / name
    root.mkdir(exist_ok=True, parents=True)
    assert (
        read(Path(s["requires"]) / "decision.json")["passed"]
        and digest(Path(s["requires"]) / "checksums.json") == s["prerequisite_sha256"]
    )
    if (root / "decision.json").exists():
        raise FileExistsError("The direct comparison is closed")
    if (root / "registration.json").exists():
        assert read(root / "registration.json") == s
    else:
        save(root / "registration.json", s)
    capture(
        root,
        (
            "loopqwen/research/reasoned_evidence.py",
            "loopqwen/research/reasoned_evidence_fit.py",
            "loopqwen/research/reasoned_evidence_data.py",
            "scripts/reasoned_evidence_material.py",
            "loopqwen/research/cross_view_study.py",
            "loopqwen/research/native_function_study.py",
            "loopqwen/research/native_function_data.py",
            "loopqwen/research/file_evidence_fit.py",
            "loopqwen/cross_view.py",
            "loopqwen/interpolated_instruction.py",
            "loopqwen/looped_model.py",
            "loopqwen/tool_session.py",
            "loopqwen/tool_model.py",
            "pyproject.toml",
            "uv.lock",
            *extra_sources,
        ),
    )
    guard = Budget(name, s["limits"])
    try:
        tok = tokenizer(s)
        qualify(s, tok, root, guard)
        guard.check()
        mx.reset_peak_memory()
        if not (root / "fit.json").exists():
            net = bind(None, Path(s["checkpoint"]) / "tool")
            fit_messages(net, tok, s, root, guard, trainer=trainer)
            del net
            gc.collect()
            mx.clear_cache()
        guard.check()
        mx.reset_peak_memory()
        if not (root / "compiled.json").exists():
            compiler(s["checkpoint"], root)
        if after_compile is not None:
            after_compile(s["checkpoint"], root)
        gc.collect()
        mx.clear_cache()
        guard.check()

        def base():
            actor = DenseAssistant(True, "qwen", s["checkpoint"])
            actor.native_plain = True
            return actor

        p = read(s["material"])
        arms = {}
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_documents(p, folder)
            for side, make in (
                ("candidate", lambda: CrossViewAssistant(s["checkpoint"], root)),
                ("control", base),
            ):
                path = root / (side + ".json.gz")
                if not path.exists():
                    gc.collect()
                    mx.clear_cache()
                    records = evaluate(
                        s,
                        p,
                        side,
                        guard,
                        make,
                        lambda w: tool_catalog(w, folder),
                        scorer=lambda w, i, tr: score_fn(tok, w, i, tr),
                    )
                    packed(path, records)
                arms[side] = unpack(path)
        d = decision_rule(p, arms)
    except (MemoryError, TimeoutError, InterruptedError) as exc:
        d = {"passed": False, "status": "interrupted", "reason": str(exc)}
    finally:
        guard.close()
    save(root / "decision.json", d)
    print("Closed direct reasoning comparison", flush=True)


if __name__ == "__main__":
    run()
