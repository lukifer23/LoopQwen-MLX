"""Blindly review complete noncanonical replies and reconstruct article-level sensitivity."""

import argparse
from collections import defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from scripts.revision_material import NAME, PANEL

LABELS = Path("research/diagnostics/revision-answer-reading.json")


def normalize(s):
    return s.strip().strip(" .!*\n\t").lower()


def review(root, blind=False):
    rows = read(PANEL)
    displays = {}
    grouped = defaultdict(dict)
    for arm in ("base", "candidate"):
        for row, r in zip(rows, unpack(root / (arm + ".json.gz")), strict=True):
            assert row["id"] == r["id"]
            s = r["score"]
            text = s["visible"]
            literal = normalize(text)
            if s["complete"] and literal and literal not in ("yes", "no", "maybe"):
                body = {k: row[k] for k in ("evidence", "claim")}
                body["reply"] = text
                ident = fingerprint(body)
                displays[ident] = body
            else:
                ident = None
            grouped[arm].setdefault(row["case"], []).append((row["label"], s, ident))
    if blind:
        for ident, item in sorted(displays.items()):
            print(ident[:16], item)
        return
    labels = read(LABELS)
    assert labels.keys() == {ident[:16] for ident in displays}
    assert set(labels.values()) <= {"yes", "no", "unclear", "conflict"}
    result = {}
    for arm, bycase in grouped.items():
        pairs = []
        for slots in bycase.values():
            halves = []
            for label, score, ident in slots:
                choice = labels[ident[:16]] if ident is not None else score["answer"]
                halves.append(
                    score["complete"]
                    and choice == {"SUPPORTS": "yes", "REFUTES": "no"}[label]
                )
            pairs.append(halves)
        result[arm] = {
            "items": sum(sum(x) for x in pairs),
            "pairs": sum(all(x) for x in pairs),
        }
    result["distinct_reviewed"] = len(displays)
    save("docs/measurements/revision-answer-reading.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("research/results") / NAME)
    args = parser.parse_args()
    review(args.root, args.blind)
