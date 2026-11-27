"""Compare one protected, two-visit decoder against the pinned Base on matched current evidence."""

import math
import re
from collections import defaultdict
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet

from ..evaluation.extractive import normalized
from ..evaluation.extractive import score as extract
from ..evaluation.mechanism import unpack
from ..io import digest, read, save
from ..wheel_tools import tools as wheel_tools
from . import reasoned_evidence as comparison
from .followup_fit import batches, fit
from .linked_rollin import exact_sign_flip
from .linked_rollin_data import providers as earlier_tools
from .linked_rollin_data import score as earlier_score
from .native_function_study import executions, summarize
from .registered import registered

NAME = "followup-protected-01"
# PEP 440 clauses, including comma-separated conjunctions; unknown prose is never executed.
CLAUSE = r"(?:~=|==|!=|<=|>=|<|>|===)\s*[0-9][0-9a-zA-Z.*+!_-]*"
RANGE = re.compile(CLAUSE + r"(?:\s*,\s*" + CLAUSE + r")*")


def refs(labels):
    if labels.count("CANNOTANSWER") * 2 >= len(labels):
        return ["CANNOTANSWER"]
    return [z for z in labels if z != "CANNOTANSWER"]


def agreement(text, labels):
    # Convert the exact user-facing abstention to the publisher’s special label.
    if text == "Cannot answer":
        text = "CANNOTANSWER"
    if labels == ["CANNOTANSWER"]:
        return float(text == "CANNOTANSWER")
    f = [extract(text, [r])[1] for r in labels]
    return (
        f[0]
        if len(f) == 1
        else sum(max(f[:j] + f[j + 1 :]) for j in range(len(f))) / len(f)
    )


def human_ceiling(labels):
    if len(labels) == 1:
        return 1.0
    return sum(
        max(extract(a, [b])[1] for j, b in enumerate(labels) if j != i)
        for i, a in enumerate(labels)
    ) / len(labels)


def providers(w, folder):
    if "wheels" in w:
        from scripts.wheel_archive_sources import CACHE

        return wheel_tools([CACHE / "wheels" / n for n in w["wheels"]])
    return [] if w["family"].startswith("human_") else earlier_tools(w, folder)


def ranges(text):
    found = set()
    for match in RANGE.finditer(text):
        try:
            found.add(str(SpecifierSet(match.group().rstrip("."))))
        except InvalidSpecifier:
            continue
    return found


def score(tok, w, i, tr):
    done = tr["status"] == "answered"
    text = (tr["answer"] or "") if done else ""
    if w["family"].startswith("human_"):
        labels = refs(w["annotations"])
        f = agreement(text, labels) if done else 0.0
        ceiling = human_ceiling(labels)
        same = bool(
            w["previous_answer"]
            and normalized(text) == normalized(w["previous_answer"])
        )
        return {
            "completed": done,
            "success": done and f >= 0.5,
            "correct_answer": done and f >= 0.5,
            "f1": f,
            "human_agreement": ceiling,
            "human_equivalent": done and f >= ceiling,
            "previous_repeated": same,
        }
    if "wheels" in w:
        constraints = ranges(text)
        current = str(SpecifierSet(w["requirement"]))
        mentions = constraints == {current}
        yn = set(re.findall(r"\b(?:yes|no)\b", text.casefold()))
        verdict = yn == {"yes" if w["allowed"] else "no"}
        measured = any(
            e["call"].get("name") == "wheel_metadata"
            and e["call"].get("arguments", {}).get("wheel")
            == w["current"]["call"]["arguments"]["wheel"]
            and e["call"]["arguments"].get("field", "").casefold() == "requires-python"
            and e["observation"] == w["current"]["observation"]
            for e in executions({"turns": tr["turns"][:-1]})
        )
        return {
            "completed": done,
            "success": done and mentions and verdict and measured,
            "correct_answer": done and mentions and verdict,
            "measured": measured,
            "verdict": verdict,
            "constraint": mentions,
            "used_older_constraint": str(SpecifierSet(w["prior_requirement"]))
            in constraints,
        }
    return earlier_score(tok, w, i, tr)


def outcome(panel, runs):
    arms = {}
    scores = {}
    for side, data in runs.items():
        assert [w["id"] for w in panel["rows"]] == [r["id"] for r in data]
        by = defaultdict(list)
        for w, r in zip(panel["rows"], data, strict=True):
            by[w["family"]].append(r)
        fam = {k: summarize(v) for k, v in by.items()}
        scores[side] = {r["id"]: r["requests"][0]["score"] for r in data}
        a = scores[side]
        whole = defaultdict(lambda: defaultdict(list))
        for w in panel["rows"]:
            kind = "human" if w["family"].startswith("human_") else w["family"]
            whole[kind][w["cluster"]].append(a[w["id"]]["success"])
        ts = [t for r in data for q in r["requests"] for t in q["trajectory"]["turns"]]
        f1 = {
            f: sum(r["requests"][0]["score"]["f1"] for r in by[f]) / len(by[f])
            for f in ("human_first", "human_followup")
        }
        arms[side] = {
            "all": summarize(data),
            "families": fam,
            "whole": {k: sum(all(v) for v in gr.values()) for k, gr in whole.items()},
            "mean_f1": f1,
            "opening_tps": sum(
                len(t["prompt_ids"]) - t.get("reused_prompt_tokens", 0) for t in ts
            )
            / sum(t["ttft_s"] for t in ts),
            "decode_tps": sum(len(t["token_ids"]) - 1 for t in ts)
            / sum(t["seconds"] - t["ttft_s"] for t in ts),
        }
    a, b = arms["candidate"], arms["control"]
    x, y = scores["candidate"], scores["control"]
    differences = defaultdict(int)
    lost = []
    for w in panel["rows"]:
        key = w["id"]
        if w["family"].startswith("human_"):
            differences[w["cluster"]] += int(x[key]["success"]) - int(y[key]["success"])
        elif w["family"] != "archive" and y[key]["success"] and not x[key]["success"]:
            lost.append(key)
    p = exact_sign_flip(list(differences.values()))
    gates = {
        "human_net": sum(differences.values()) >= 8 and p <= 0.05,
        "followup_gain": a["mean_f1"]["human_followup"]
        >= b["mean_f1"]["human_followup"] + 0.10,
        "first_gain": a["mean_f1"]["human_first"] > b["mean_f1"]["human_first"],
        "whole_human": a["whole"]["human"] >= max(4, b["whole"]["human"] + 1),
        "archive_pairs": a["whole"]["archive"] >= max(4, b["whole"]["archive"] + 1),
        "old_cases": not lost,
        "old_pairs": all(
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
        "gates": gates,
        "arms": arms,
        "lost": lost,
        "source_differences": dict(differences),
        "paired_p": p,
    }


def qualify(s, tok, root, guard):
    from scripts.followup_material import build
    from scripts.followup_oracle import RECEIPT, audit
    from scripts.followup_sources import enroll

    prior = read(Path("research/results/linked-rollin-02") / "decision.json")
    if prior.get("passed") or "arms" not in prior:
        raise InterruptedError("The previous direct primary requires its own follow-up")
    enroll()
    build()
    receipt = audit()
    assert digest(RECEIPT) == s["source_oracle_sha256"]
    save(root / "source-qualification.json", receipt)
    guard.check()


def verify(root, d):
    s = registered(NAME)
    comparison.verify(
        root,
        d,
        name=NAME,
        make_views=lambda spec, _: read(spec["training"])["examples"],
        decision_rule=outcome,
        score_fn=score,
        tool_catalog=providers,
    )
    if d.get("status") == "interrupted":
        return
    views = unpack(root / "training-views.json.gz")
    records = read(root / "optimizer-directions.json")
    fitlog = read(root / "fit.json")
    from .cross_view_study import tokenizer
    from .followup_fit import semantic
    from .native_function_data import token_pair

    tok = tokenizer(s)
    contrasts = {}
    for i, w in enumerate(views):
        if w["alternative"] is not None:
            _, bad, _ = semantic(tok, w)
            if bad:
                j, foil = bad
                contrasts[i] = (j, token_pair(tok, w)[1][j], foil)
    draw = batches(views, s["fit_seed"])
    assert len(records) == fitlog["updates"] == s["updates"]
    for i, r in enumerate(records, 1):
        active, ref = next(draw)
        assert r["active"] == active and r["protected"] == ref and r["step"] == i
        assert (
            r["factor"] in (1, 0.5, 0.25, 0.125, 0.0625, 0)
            and r["after"] <= r["before"] + s["reference_tolerance"]
        )
        expected = [i for i in active if i in contrasts]
        assert [m["view"] for m in r["margins"]] == expected
        for m in r["margins"]:
            assert (m["position"], m["target"], m["foil"]) == contrasts[m["view"]]
            assert math.isfinite(m["first"]) and math.isfinite(m["second"])
        assert r["projected"] == (r["raw_directional"] > 0)
        assert all(
            math.isfinite(r[k]) and r[k] >= 0
            for k in ("proposed_step_l2", "applied_step_l2")
        )
        if r["factor"] == 0:
            assert r["applied_step_l2"] == 0 and r["after"] == r["before"]
    assert fitlog["applied"] == sum(r["factor"] > 0 for r in records)
    assert fitlog["projected"] == sum(r["projected"] for r in records)


if __name__ == "__main__":
    comparison.run(
        name=NAME,
        qualify=qualify,
        trainer=fit,
        decision_rule=outcome,
        score_fn=score,
        tool_catalog=providers,
        extra_sources=(
            "loopqwen/research/followup_fit.py",
            "loopqwen/research/followup_study.py",
            "loopqwen/research/linked_rollin_fit.py",
            "scripts/followup_sources.py",
            "scripts/followup_material.py",
            "scripts/followup_oracle.py",
            "loopqwen/wheel_tools.py",
        ),
    )
