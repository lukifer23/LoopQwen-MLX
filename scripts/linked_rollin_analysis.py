"""Trace each open answer back to a visible source, actual callback and complete user request."""

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from loopqwen.evaluation.extractive import score as extract
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.linked_rollin import exact_sign_flip
from loopqwen.research.linked_rollin_data import matched, reached
from loopqwen.research.native_function_study import executions

NAME = "linked-rollin-02"
ARMS = ("candidate", "control")
ROOT = Path("research/results") / NAME
DOC = Path("docs/measurements")
PLAN = Path("research/diagnostics/linked-rollin-source-review.json")
BLIND = DOC / "linked-rollin-blinded-answers.json"
RATINGS = DOC / "linked-rollin-blinded-ratings.json"


def inspect(root):
    spec = read(root / "registration.json")
    plan = read(PLAN)
    assert digest(Path(spec["material"])) == plan["panel_sha256"]
    notes = {w["cluster"]: w for w in plan["items"]}
    p = read(spec["material"])
    arms = {a: unpack(root / (a + ".json.gz")) for a in ARMS}
    records, blind = [], {}
    for i, w in enumerate(p["rows"]):
        linked = "files" in w
        note = notes[w["cluster"]] if linked else None
        row = {k: w[k] for k in ("id", "cluster", "family", "gold")}
        row["question"] = w["questions"][0]
        row["source_support"] = note["label"] if linked else ""
        for arm in ARMS:
            case = arms[arm][i]
            assert case["id"] == row["id"]
            (q,) = case["requests"]
            tr = q["trajectory"]
            turns = tr["turns"]
            acts = [executions({"turns": [t]}) for t in turns]
            seq = [e for group in acts for e in group]
            ans = tr["answer"] or ""
            score = q["score"]
            fields = {
                "answer": ans,
                "status": tr["status"],
                "completed": score["completed"],
                "success": score["success"],
                "first_s": q["first_s"],
                "wall_s": q["wall_s"],
                "peak": q["peak"],
                "turns": len(turns),
                "output_tokens": sum(len(t["token_ids"]) for t in turns),
                "callbacks": len(seq),
                "actions_and_observations": json.dumps(
                    acts, ensure_ascii=False, separators=(",", ":")
                ),
            }
            if linked:
                needed = len(w["events"])
                pos = reached(w, acts)
                assert score["milestones"] == pos
                answer_ok, f1 = extract(ans, [w["gold"], *note["supported_variants"]])
                first_right = bool(acts[0] and any(matched(w, 0, e) for e in acts[0]))
                direct = any(matched(w, needed - 1, e) for e in seq)
                wrong_return = bool(
                    acts[0]
                    and not first_right
                    and any("value" in e["observation"] for e in acts[0])
                )
                fields.update(
                    official_exact=score["correct_answer"],
                    official_f1=score["f1"],
                    variant_grounded=score["completed"] and answer_ok and pos == needed,
                    variant_f1=f1 if score["completed"] else 0,
                    first_required=first_right,
                    waited_for_source=pos == needed,
                    article_received=direct,
                    milestones=pos,
                    required=needed,
                    early_article=needed == 2
                    and any(matched(w, 1, e) for e in acts[0]),
                    wrong_successful_open=wrong_return,
                    recovered_wrong_open=wrong_return and pos == needed,
                )
                if w["id"] == note["blind_sample"]:
                    ob = {
                        "question": note["question"],
                        "published_answer": w["gold"],
                        "source_assessment": note["source_note"],
                        "linked_cell": w["cell"],
                        "source_passage": w["events"][-1]["observation"]["value"],
                        "visible_answer": ans,
                    }
                    # Identical responses to identical questions receive the same sealed assessment.
                    key = fingerprint(ob)[:20]
                    assert key not in blind or blind[key] == ob
                    blind[key] = ob
                    fields["blind_key"] = key
            row.update({arm + "_" + k: v for k, v in fields.items()})
        records.append(row)
    assert len(records) == 84 and sum("candidate_blind_key" in r for r in records) == 24
    return records, dict(sorted(blind.items()))


def intervals(rows):
    # Keep both input presentations or both opposing practical requests in the same source.
    strata = defaultdict(lambda: defaultdict(list))
    for r in rows:
        f = r["family"]
        family = (
            "human"
            if f in ("inline", "file")
            else "linked"
            if f.startswith("linked_")
            else f
        )
        strata[family][r["cluster"]].append(r)
    rng = np.random.default_rng(670032)
    total = np.zeros((20000, 2, 2))
    for g in strata.values():
        v = np.array(
            [
                [
                    [sum(r[a + "_" + k] for r in unit) for k in ("first_s", "wall_s")]
                    for a in ARMS
                ]
                for unit in g.values()
            ]
        )
        total += v[rng.integers(0, len(v), size=(len(total), len(v)))].sum(axis=1)
    ratio = total[:, 0, :] / total[:, 1, :]
    return dict(
        zip(
            ("first_ratio", "wall_ratio"),
            np.quantile(ratio, [0.025, 0.975], axis=0).T.tolist(),
            strict=True,
        )
    )


def compare(rows):
    result = {
        "timing_intervals": intervals(rows),
        "blinded": {},
        "sources": {},
        "families": {},
    }
    ratings = read(RATINGS)
    assert set(ratings) == {
        r[a + "_blind_key"] for r in rows for a in ARMS if a + "_blind_key" in r
    }
    for a in ARMS:
        counts = Counter()
        claims = grounded_semantic = 0
        for r in rows:
            if a + "_blind_key" in r:
                review = ratings[r[a + "_blind_key"]]
                assert review["verdict"] in (
                    "equivalent",
                    "incomplete",
                    "contradicted",
                    "appropriately qualified",
                    "absent",
                )
                assert type(review["invented_factual_claim"]) is bool
                counts[review["verdict"]] += 1
                claims += int(review["invented_factual_claim"])
                grounded_semantic += int(
                    r[a + "_waited_for_source"]
                    and review["verdict"] in ("equivalent", "appropriately qualified")
                )
        result["blinded"][a] = {
            "counts": dict(counts),
            "invented_claims": claims,
            "source_observed_and_supported": grounded_semantic,
        }
    for subset, check in (
        ("all", lambda r: True),
        ("direct_support", lambda r: r["source_support"] == "supported"),
    ):
        selected = [r for r in rows if r["family"].startswith("linked_") and check(r)]
        for measure in (
            "success",
            "variant_grounded",
            "official_exact",
            "waited_for_source",
        ):
            g = defaultdict(list)
            for r in selected:
                g[r["cluster"]].append(
                    int(r["candidate_" + measure]) - int(r["control_" + measure])
                )
            d = [sum(x) for x in g.values()]
            result["sources"][subset + "_" + measure] = {
                "requests": len(selected),
                "clusters": len(d),
                **{a: sum(r[a + "_" + measure] for r in selected) for a in ARMS},
                "net": sum(d),
                "paired_p": exact_sign_flip(d),
            }
    for f in sorted({r["family"] for r in rows}):
        group = [r for r in rows if r["family"] == f]
        metrics = ["success", "completed", "callbacks", "turns", "output_tokens"]
        if f.startswith("linked_"):
            metrics += [
                "article_received",
                "waited_for_source",
                "first_required",
                "early_article",
                "wrong_successful_open",
                "recovered_wrong_open",
                "official_exact",
                "variant_grounded",
            ]
        result["families"][f] = {
            a: {k: sum(r[a + "_" + k] for r in group) for k in metrics} for a in ARMS
        }
    gained = [r for r in rows if r["candidate_success"] and not r["control_success"]]
    result["gains"] = [r["id"] for r in gained]
    result["losses"] = [
        r["id"] for r in rows if not r["candidate_success"] and r["control_success"]
    ]
    result["joint_gains"] = [
        r["id"]
        for r in gained
        if all(
            r["candidate_" + k] < r["control_" + k]
            for k in ("first_s", "wall_s", "peak")
        )
    ]
    return result


def persist(path, data):
    if path.exists():
        assert read(path) == data
    else:
        save(path, data)


def run(root, mode):
    rows, blind = inspect(root)
    persist(BLIND, blind)
    if mode == "blind":
        print(len(blind), "unique blinded responses from 48 appearances")
        return
    output = compare(rows)
    persist(DOC / "linked-rollin-analysis.json", output)
    stream = io.StringIO(newline="")
    keys = list(
        dict.fromkeys(k for r in rows for k in r if not k.endswith("_blind_key"))
    )
    out = csv.DictWriter(
        stream, fieldnames=keys, extrasaction="ignore", lineterminator="\n"
    )
    out.writeheader()
    out.writerows(rows)
    csv_path = Path("paper/linked-rollin-requests.csv")
    data = stream.getvalue().encode()
    if csv_path.exists():
        assert csv_path.read_bytes() == data
    else:
        csv_path.write_bytes(data)
    print(
        "Recounted all 168 native model trajectories and 48 masked answer appearances."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("blind", "report"))
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    run(args.root, args.mode)
