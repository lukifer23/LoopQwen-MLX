"""Inventory the complete registered program, including missed gates and recovered executions."""

import csv
import json
import math
from collections import Counter, defaultdict
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


def execution_windows():
    """Account once for each completed job in the closed native-worker windows."""
    records = []
    by_study = defaultdict(list)
    paths = sorted(Path("research/sessions").glob("window-*.json"))
    for path in paths:
        w = read(path)
        expected = w.get("native_seconds", w.get("total_native_seconds"))
        assert math.isclose(
            sum(j["seconds"] for j in w["jobs"]), expected, abs_tol=0.01
        )
        for j in w["jobs"]:
            assert j["seconds"] > 0 and j["peak_bytes"] >= 0
            row = {
                "ledger": str(path),
                "ledger_sha256": digest(path),
                "study": j["job"],
                "start_unix_s": j["start"],
                "elapsed_s": j["seconds"],
                "mlx_peak_bytes": j["peak_bytes"],
            }
            records.append(row)
            by_study[j["job"]].append(row)
    records.sort(key=lambda r: r["start_unix_s"])
    assert all(
        a["start_unix_s"] + a["elapsed_s"] <= b["start_unix_s"] + 0.01
        for a, b in zip(records, records[1:])
    )
    target = PAPER / "research-worker-windows.csv"
    with target.open("w", newline="") as stream:
        out = csv.DictWriter(stream, records[0].keys(), lineterminator="\n")
        out.writeheader()
        out.writerows(records)
    return by_study, {
        "windows": len(paths),
        "jobs": len(records),
        "study_names": len(by_study),
        "worker_hours": round(sum(r["elapsed_s"] for r in records) / 3600, 3),
        "index": digest(target),
    }


def inventory():
    registry = read(REGISTRY)
    rows = []
    jobs, worker = execution_windows()
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
                    s.get("decision")
                    or s.get("gate")
                    or s.get("metric")
                    or s.get("protocol")
                    or s.get("evaluation")
                    or s.get("parity")
                ),
                "registered_prerequisite": value(s.get("requires")),
                "superseded_by": retired.get(name, ""),
                "registered_limits": value(s["limits"]),
                "recorded_jobs_for_study_name": len(jobs[name])
                if kind == "primary"
                else "",
                "recorded_worker_seconds_for_study_name": sum(
                    r["elapsed_s"] for r in jobs[name]
                )
                if kind == "primary" and jobs[name]
                else "",
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
            "closed_worker_windows": worker,
            "totals": dict(counts),
            "index": digest(PAPER / "research-scope.csv"),
        },
    )
    print(len(sources), "registrations;", counts["payloads"], "archived files checked")


if __name__ == "__main__":
    inventory()
