"""Review complete open-ended replies without their arm or human label, then reconstruct paired outcomes."""

import argparse
from collections import defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import NAME


def review(root, blind):
    s = read(root / "registration.json")
    p = read(s["material"])
    by_id = {q["id"]: q for a in p["articles"] for q in a["questions"]}
    entries = {}
    arms = {}
    for arm in ("candidate", "control"):
        rr = unpack(root / (arm + ".json.gz"))
        output = []
        for w, r in zip(p["rows"], rr, strict=True):
            if w["family"] == "retention":
                continue
            q = r["requests"][0]
            g = q["score"]
            tr = q["trajectory"]
            key = None
            if g["completed"] and not g["exact"]:
                source = {
                    "question": by_id[w["question_id"]]["question"],
                    "passage": w["source"],
                    "reply": tr["answer"],
                }
                key = fingerprint(source)[:16]
                if key in entries:
                    assert entries[key] == source
                entries[key] = source
            output.append((w, g, key))
        arms[arm] = output
    if blind:
        import json

        print(json.dumps(entries, indent=2, sort_keys=True, ensure_ascii=False))
        return
    labels = read("research/diagnostics/cross-view-answers.json")
    assert labels.keys() == entries.keys() and set(labels.values()) <= {
        "correct",
        "incorrect",
        "unresolved",
    }
    results = {}
    for arm, rows in arms.items():
        stats = defaultdict(lambda: defaultdict(int))
        articles = defaultdict(lambda: defaultdict(list))
        for w, g, key in rows:
            f = w["family"]
            valid = g["completed"] and (g["exact"] or labels.get(key) == "correct")
            grounded = valid and (f == "inline" or g["observed"])
            stats[f]["requests"] += 1
            stats[f]["semantic_answers"] += valid
            stats[f]["grounded"] += grounded
            stats[f]["original_grounded"] += g["success"]
            stats[f]["unresolved"] += labels.get(key) == "unresolved"
            articles[w["cluster"]][f].append(grounded)
        results[arm] = {
            "by_family": dict(stats),
            "both_file": sum(all(a["file"]) for a in articles.values()),
            "both_inline": sum(all(a["inline"]) for a in articles.values()),
            "all_four": sum(
                all(a["file"]) and all(a["inline"]) for a in articles.values()
            ),
        }
    result = {
        "distinct_reviewed": len(entries),
        "labels_sha256": digest(Path("research/diagnostics/cross-view-answers.json")),
        "arms": results,
    }
    save("docs/measurements/cross-view-answer-reading.json", result)
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blind", action="store_true")
    p.add_argument("--root", type=Path, default=Path("research/results") / NAME)
    args = p.parse_args()
    review(args.root, args.blind)
