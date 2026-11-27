"""Test whether corrections at naturally visited decision boundaries improve open-answer tool use."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from ..evaluation.mechanism import unpack
from ..io import digest, read, save
from . import reasoned_evidence as pair
from .linked_rollin_data import providers, score
from .linked_rollin_fit import fit, turns
from .native_function_study import summarize

NAME = "linked-rollin-01"
TEXT_DISJOINT = "linked-rollin-02"


def qualify(s, tok, root, guard):
    from scripts.linked_rollin_material import build
    from scripts.linked_table_archive import audit
    from scripts.linked_table_material import run
    from scripts.linked_table_sources import DEST

    upstream = read("research/results/valid-return-01/decision.json")
    if "arms" not in upstream or upstream["passed"]:
        raise InterruptedError(
            "The previous direct pair has priority for its own staged follow-up"
        )
    audit(read(DEST / "enrollment.json"))
    corrected = s["id"] == TEXT_DISJOINT
    callbacks = Path("docs/measurements/linked-table-callbacks.json")
    if corrected:
        from scripts.linked_text_repair import audit as audit_text, RECEIPT

        stopped = Path("research/results") / NAME
        assert read(stopped / "decision.json").get("status") == "interrupted"
        assert not any(
            (stopped / n).exists()
            for n in ("candidate.json.gz", "control.json.gz", "fit.json")
        )
        assert digest(stopped / "checksums.json") == s["stopped_source_study_sha256"]
        audit_text()
        assert digest(RECEIPT) == s["sources"]["bidirectional_text_audit"]
        callbacks = callbacks.with_name("linked-text-callbacks.json")
    run(text_disjoint=corrected)
    build(text_disjoint=corrected)
    assert digest(callbacks) == s["sources"]["callbacks"]
    train = read(s["training"])
    panel = read(s["material"])
    tr = {w["cluster"] for w in train["rows"]}
    held = {w["cluster"] for w in panel["rows"] if "files" in w}
    assert len(tr) == 128 and len(held) == 24 and not tr & held
    assert Counter(w["family"] for w in panel["rows"]) == {
        "linked_file": 24,
        "linked_inline": 24,
        "file": 6,
        "inline": 6,
        "agenda": 8,
        "wheel": 8,
        "retention": 8,
    }
    save(
        root / "source-qualification.json",
        {
            "training_sha256": digest(Path(s["training"])),
            "panel_sha256": digest(Path(s["material"])),
            "training_sources": len(tr),
            "fresh_sources": len(held),
            "requests": len(panel["rows"]),
            "offline_sources": digest(
                Path("docs/measurements/linked-table-source-audit.json")
            ),
            "callbacks": digest(callbacks),
            **(
                {"bidirectional_text_audit": s["sources"]["bidirectional_text_audit"]}
                if corrected
                else {}
            ),
        },
    )
    guard.check()


def exact_sign_flip(deltas):
    paths = {0: 1}
    for x in deltas:
        step = Counter()
        for value, n in paths.items():
            step[value + x] += n
            step[value - x] += n
        paths = step
    return sum(n for v, n in paths.items() if abs(v) >= abs(sum(deltas))) / (
        2 ** len(deltas)
    )


def decision(p, arms):
    sides = {}
    outcomes = {}
    for arm, items in arms.items():
        assert [r["id"] for r in items] == [w["id"] for w in p["rows"]]
        fam = {
            f: summarize([r for r in items if r["family"] == f])
            for f in sorted({w["family"] for w in p["rows"]})
        }
        whole = defaultdict(lambda: defaultdict(list))
        retention = set()
        fs = []
        traces = []
        for w, r in zip(p["rows"], items, strict=True):
            q = r["requests"][0]
            s = q["score"]
            traces.extend(q["trajectory"]["turns"])
            if "files" in w:
                whole["linked"][w["cluster"]].append(s["success"])
                fs.append(s["f1"])
            else:
                if s["success"]:
                    retention.add(w["id"])
                whole[w["family"]][w["cluster"]].append(s["success"])
        all_group = summarize(items)
        assert len(fs) == 48 and len(whole["linked"]) == 24
        sides[arm] = {
            "families": fam,
            "all": all_group,
            "mean_open_f1": sum(fs) / len(fs),
            "complete": {
                k: sum(all(v) for v in group.values()) for k, group in whole.items()
            },
            "opening_tps": sum(len(q["prompt_ids"]) for q in traces)
            / sum(q["ttft_s"] for q in traces),
            "decode_tps": sum(len(q["token_ids"]) - 1 for q in traces)
            / sum(q["seconds"] - q["ttft_s"] for q in traces),
        }
        outcomes[arm] = (whole["linked"], retention)
    a, b = [sides[k] for k in ("candidate", "control")]
    left, ar = outcomes["candidate"]
    right, br = outcomes["control"]
    delta = [sum(left[k]) - sum(right[k]) for k in sorted(left)]
    prob = exact_sign_flip(delta)
    new = sum(map(sum, left.values()))
    gates = {
        "human_open_answers": new >= 12 and sum(delta) >= 8 and prob <= 0.05,
        "both_presentations": all(
            a["families"][k]["grounded"] > b["families"][k]["grounded"]
            for k in ("linked_file", "linked_inline")
        ),
        "whole_questions": a["complete"]["linked"] >= 3
        and a["complete"]["linked"] > b["complete"]["linked"],
        "f1": a["mean_open_f1"] > b["mean_open_f1"],
        "earlier_requests": br <= ar,
        "practical_tools": all(
            a["complete"][k] >= 2 and a["complete"][k] > b["complete"][k]
            for k in ("wheel", "agenda")
        ),
        "completion": all(
            a["families"][k]["completed"] >= b["families"][k]["completed"]
            for k in a["families"]
        ),
        **{k: a["all"][k] < b["all"][k] for k in ("first_s", "wall_s", "peak")},
        **{k: a[k] > b[k] for k in ("opening_tps", "decode_tps")},
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "arms": sides,
        "paired_source_differences": delta,
        "net": sum(delta),
        "exact_p": prob,
        "earlier_base_successes_lost": sorted(br - ar),
    }


def verify(root, d):
    pair.verify(
        root,
        d,
        name=read(root / "registration.json")["id"],
        make_views=turns,
        decision_rule=decision,
        score_fn=score,
        tool_catalog=providers,
    )
    if d.get("status"):
        return
    f = read(root / "fit.json")
    assert sum(f["draws"].values()) == 4 * f["updates"] and f["contrast_draws"] > 0
    for phase, x in enumerate(f["phases"]):
        assert (
            x["phase"] == phase
            and 0 < x["with_real_alternative_token"] <= x["corrections"]
        )
        source = unpack(root / f"student-round-{phase}.json.gz")
        assert len(source) == 64


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-disjoint", action="store_true")
    args = parser.parse_args()
    pair.run(
        name=TEXT_DISJOINT if args.text_disjoint else NAME,
        qualify=qualify,
        trainer=fit,
        decision_rule=decision,
        score_fn=score,
        tool_catalog=providers,
        extra_sources=(
            "loopqwen/research/linked_rollin.py",
            "loopqwen/research/linked_rollin_data.py",
            "loopqwen/research/linked_rollin_fit.py",
            "scripts/linked_rollin_material.py",
            "scripts/linked_table_material.py",
            "scripts/linked_table_archive.py",
            "scripts/linked_table_sources.py",
            "loopqwen/research/valid_return_data.py",
            "scripts/valid_return_material.py",
            "loopqwen/evaluation/extractive.py",
            "loopqwen/evaluation/observed_chain.py",
            "loopqwen/evaluation/generation_eval.py",
            "loopqwen/qwen_tools.py",
            *(
                ("scripts/linked_text_repair.py", "scripts/linked_source_overlap.py")
                if args.text_disjoint
                else ()
            ),
        ),
    )
