"""Check that only a real file cell changed, then recount the use of the returned text."""

import argparse
import csv
import hashlib
import io
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.native_function_study import executions
from scripts.revision_answer_reading import normalize

NAME = "file-evidence-01"
PANEL = Path("research/data/file-evidence-panel.json")
SOURCE = Path("docs/measurements/file-evidence-cells.json")


def sources():
    rows = read(PANEL)["rows"]
    pairs = defaultdict(list)
    for w in rows:
        if w["family"] == "file":
            pairs[w["cluster"]].append(w)
    results = []
    for case, pair in pairs.items():
        a, b = pair
        assert (
            a["questions"] == b["questions"]
            and a["initial_ids"] == b["initial_ids"]
            and a["calls"] == b["calls"]
        )
        versions = [list(csv.reader(io.StringIO(w["file"], newline=""))) for w in pair]
        left, right = versions
        assert (
            left[0] == right[0] == ["article", "excerpt"]
            and len(left) == len(right) == 4
        )
        changed = [
            (i, j)
            for i, (u, v) in enumerate(zip(left, right, strict=True))
            for j, (t, s) in enumerate(zip(u, v, strict=True))
            if t != s
        ]
        assert len(changed) == 1
        idx, col = changed[0]
        assert (
            idx > 0
            and col == 1
            and left[idx] == [a["page"], a["evidence"]]
            and right[idx] == [b["page"], b["evidence"]]
        )
        assert all(
            w["calls"]
            == [
                {
                    "name": "csv_lookup",
                    "arguments": {
                        "document": "clippings.csv",
                        "key_column": "article",
                        "key": w["page"],
                        "value_column": "excerpt",
                    },
                }
            ]
            for w in pair
        )
        results.append(
            {
                "article": a["page"],
                "case": case,
                "revision": a["revision"],
                "row_after_header": idx,
                "changed_column": "excerpt",
                "input_tokens": len(a["initial_ids"]),
                "input_sha256": fingerprint(a["initial_ids"]),
                "files": [
                    {
                        "id": w["id"],
                        "sha256": hashlib.sha256(w["file"].encode()).hexdigest(),
                        "gold": w["gold"],
                    }
                    for w in pair
                ],
            }
        )
    acquired = read("docs/measurements/file-evidence-wikipedia-sources.json")
    assert len(pairs) == 16 and {r["revision"] for r in rows} == {
        str(x["revision"]) for x in acquired["articles"]
    }
    result = {
        "physical_cells_verified": len(results),
        "distinct_source_edits": acquired["edits"],
        "questions": results,
    }
    if SOURCE.exists():
        assert read(SOURCE) == result
    else:
        save(SOURCE, result)
    print(
        len(results), "pairs with one changed cell and identical user-facing token IDs"
    )


def review(root, blind):
    rows = read(PANEL)["rows"]
    displays = {}
    arms = {}
    for arm in ("candidate", "control"):
        output = []
        for w, r in zip(rows, unpack(root / (arm + ".json.gz")), strict=True):
            assert w["id"] == r["id"]
            q = r["requests"][0]
            tr = q["trajectory"]
            actual = tr["answer"] or ""
            s = q["score"]
            ident = None
            if s["completed"] and normalize(actual) not in ("", "yes", "no", "maybe"):
                body = {"claim": w["claim"], "evidence": w["evidence"], "reply": actual}
                ident = fingerprint(body)[:16]
                displays[ident] = body
            output.append((w, s, tr, ident))
        arms[arm] = output
    if blind:
        for k, v in sorted(displays.items()):
            print(k, v)
        return
    labels = read("research/diagnostics/file-evidence-answer-reading.json")
    assert labels.keys() == displays.keys() and set(labels.values()) <= {
        "yes",
        "no",
        "conflict",
        "unclear",
    }
    notes = {
        r["case"]: r["rating"]
        for r in read("research/diagnostics/file-evidence-source-reading.json")
    }
    overlap = read("docs/measurements/file-evidence-source-overlap.json")["fields"]
    near = {
        row["id"]
        for source in overlap.values()
        for row in source["items"]
        if row["normalized_exact"] or row["five_word_jaccard"] >= 0.6
    }
    template_cases = {
        w["cluster"] for w in rows if w["id"] in near or w["cluster"] in near
    }
    out = {}
    for arm, records in arms.items():
        bins = {}
        for family in ("inline", "file"):
            totals = Counter()
            pairs = defaultdict(list)
            for w, s, tr, ident in records:
                if w["family"] != family:
                    continue
                choice = s["answer"] if ident is None else labels[ident]
                correct = s["completed"] and choice == w["gold"]
                observations = executions(tr)
                grounded = family == "inline" or any(
                    e["call"] in w["calls"]
                    and e["observation"] == {"value": w["evidence"]}
                    for e in observations
                )
                assert family == "inline" or grounded == s["observed"]
                totals["requests"] += 1
                totals["correct_conclusion"] += int(correct)
                totals["correct_grounded"] += int(correct and grounded)
                totals["correct_lookup"] += int(family == "file" and grounded)
                totals["attempted_lookup"] += int(
                    any(e["call"]["name"] == "csv_lookup" for e in observations)
                )
                pairs[w["cluster"]].append((correct and grounded, choice, grounded))
            totals["complete_pairs"] = sum(all(s[0] for s in p) for p in pairs.values())
            totals["direct_pairs"] = sum(notes[c] == "direct" for c in pairs)
            totals["correct_direct_pairs"] = sum(
                notes[c] == "direct" and all(s[0] for s in p) for c, p in pairs.items()
            )
            totals["distinct_wording_pairs"] = sum(
                c not in template_cases for c in pairs
            )
            totals["correct_distinct_wording_pairs"] = sum(
                c not in template_cases and all(s[0] for s in p)
                for c, p in pairs.items()
            )
            totals["two_returned"] = (
                sum(all(s[2] for s in p) for p in pairs.values())
                if family == "file"
                else 0
            )
            totals["same_parsed_judgment"] = sum(
                p[0][1] is not None and p[0][1] == p[1][1] for p in pairs.values()
            )
            bins[family] = dict(totals)
        out[arm] = bins
    out.update(distinct_reviewed=len(displays), panel_sha256=digest(PANEL))
    save("docs/measurements/file-evidence-answer-reading.json", out)
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blind", action="store_true")
    p.add_argument("--sources", action="store_true")
    p.add_argument("--root", type=Path, default=Path("research/results") / NAME)
    args = p.parse_args()
    if args.sources:
        sources()
    else:
        review(args.root, args.blind)
