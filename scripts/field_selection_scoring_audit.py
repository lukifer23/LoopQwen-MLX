"""Independently test how much the grounded judgment result depends on answer formatting."""

import re
from collections import Counter
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, save

LABELS = {
    "true": "true",
    "entailed": "true",
    "false": "false",
    "contradicted": "false",
    "uncertain": "uncertain",
    "undetermined": "uncertain",
    "unknown": "uncertain",
}
TERM = r"\b(" + "|".join(LABELS) + r")\b"


def strict(text):
    clean = text.strip().casefold()
    return clean if clean in ("true", "false", "uncertain") else None


def tail_judgment(text):
    plain = text.casefold().replace("**", "").replace("`", "").strip()
    if not plain:
        return None
    # Only the last nonempty sentence or line can determine the sensitivity label.
    bits = [s.strip() for s in re.split(r"[.!?\n]+", plain) if s.strip()]
    if not bits:
        return None
    end = bits[-1]
    tokens = re.findall(TERM, end)
    if (
        len(set(tokens)) != 1
        or len(end.split()) > 22
        or re.search(r"\b(?:not|never|isn’t|isn't)\s+(?:clearly\s+)?" + TERM, end)
    ):
        return None
    return LABELS[tokens[-1]]


def run():
    root = Path("research/results/field-selection-01")
    policy = read(root / "registration.json")
    panel = read(policy["material"])
    report = {}
    for arm in ("candidate", "control"):
        traces = unpack(root / (arm + ".json.gz"))
        by_family = {}
        disagreements = []
        for w, t in zip(panel["rows"], traces, strict=True):
            assert t["id"] == w["id"]
            if w["family"] == "retention":
                continue
            (request,) = t["requests"]
            trajectory = request["trajectory"]
            completed = trajectory["status"] == "answered"
            answer = trajectory["answer"] if completed else ""
            found = False
            for turn in trajectory["turns"]:
                calls = list(
                    zip(
                        turn.get("calls", []), turn.get("observations", []), strict=True
                    )
                )
                if "observation" in turn:
                    calls.append((turn["call"], turn["observation"]))
                found |= any(
                    c in w["calls"] and o == w["observation"] for c, o in calls
                )
            assert found == request["score"]["observed"]
            eligible = completed and (w["document"] is None or found)
            strict_label = strict(answer)
            tail = tail_judgment(answer)
            registered = request["score"]["verdict"]
            assert request["score"]["success"] == bool(
                eligible and registered == w["gold"]
            )
            counts = by_family.setdefault(w["family"], Counter())
            for key, v in (
                ("bare", strict_label),
                ("registered", registered),
                ("final_sentence", tail),
            ):
                counts[key] += bool(eligible and v == w["gold"])
            if eligible and len({strict_label, tail, registered}) > 1:
                disagreements.append(
                    {
                        "id": w["id"],
                        "gold": w["gold"],
                        "bare": strict_label,
                        "registered": registered,
                        "final_sentence": tail,
                        "actual_answer": answer,
                    }
                )
        report[arm] = {
            "families": {k: dict(v) for k, v in by_family.items()},
            "format_disagreements": disagreements,
        }
    save("docs/measurements/field-selection-scorer-sensitivity.json", report)
    print({k: v["families"] for k, v in report.items()})


if __name__ == "__main__":
    run()
