"""Inventory the complete registered program, including missed gates and recovered executions."""

import csv
import json
from collections import Counter
from pathlib import Path

from loopqwen.io import digest, read, save
from scripts.research_archive import checksums

REGISTRY = Path("research/registry.json")
PAPER = Path("paper")


def value(x):
    if x is None:
        return ""
    return (
        x
        if isinstance(x, str)
        else json.dumps(x, sort_keys=True, separators=(",", ":"))
    )


def inventory():
    registry = read(REGISTRY)
    rows = []
    counts = Counter()
    sources = {}
    for name, record in registry["experiments"].items():
        registration = Path("research/experiments") / (name + ".json")
        assert digest(registration) == record["registration"]
        sources[name] = read(registration)
    retired = {
        s["supersedes"]: name for name, s in sources.items() if "supersedes" in s
    }
    paths = [
        (n, Path("research/results") / n, v, "primary")
        for n, v in registry["experiments"].items()
    ]
    paths.extend(
        (Path(n).name, Path("research/sessions") / n, v, "recovered execution")
        for n, v in registry["session_results"].items()
    )
    for name, folder, entry, kind in paths:
        s = sources[name]
        root = folder / "checksums.json"
        assert root.exists() == bool(entry["archive"])
        d, payloads = {}, {}
        if root.exists():
            assert digest(root) == entry["archive"]
            payloads = checksums(folder)
            assert (
                read(root) == payloads
                and payloads["registration.json"] == entry["registration"]
            )
            d = read(folder / "decision.json")
            if type(d.get("passed")) is bool:
                verdict = "passed" if d["passed"] else "missed"
            else:
                verdict = "archived without boolean gate"
            counts["payloads"] += len(payloads)
            counts["bytes"] += sum((folder / p).stat().st_size for p in payloads)
        elif name in retired:
            verdict = "superseded"
        else:
            verdict = "registered, no archive"
        counts[kind + "/" + verdict] += 1
        rows.append(
            {
                "study": name,
                "record": kind,
                "decision": verdict,
                "original_status": value(d.get("status")),
                "question_or_primary": value(
                    s.get("question") or s.get("primary") or s.get("comparison")
                ),
                "registered_gate": value(
                    s.get("decision") or s.get("gate") or s.get("metric")
                ),
                "registered_prerequisite": value(s.get("requires")),
                "superseded_by": retired.get(name, ""),
                "payloads": len(payloads),
                "archive_directory": str(folder) if payloads else "",
                "registration_sha256": entry["registration"],
                "inventory_sha256": entry["archive"] or "",
            }
        )
    with (PAPER / "research-scope.csv").open("w", newline="") as stream:
        out = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        out.writeheader()
        out.writerows(rows)
    save(
        PAPER / "research-scope.json",
        {
            "registry": digest(REGISTRY),
            "registrations": len(sources),
            "recovered": len(registry["session_results"]),
            "totals": dict(counts),
            "index": digest(PAPER / "research-scope.csv"),
        },
    )
    print(len(sources), "registrations;", counts["payloads"], "archived files checked")


if __name__ == "__main__":
    inventory()
