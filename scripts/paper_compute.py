"""Summarize closed native research windows without treating lease time as GPU utilization."""

import csv
from collections import Counter
from datetime import datetime, timezone

from loopqwen.io import digest, read


def make(root, paper):
    rows = []
    for path in sorted((root / "research/sessions").glob("window-*.json")):
        record = read(path)
        if not isinstance(record.get("jobs"), list):
            continue
        total = record.get("native_seconds", record.get("total_native_seconds"))
        if total is None:
            continue
        jobs = record["jobs"]
        if not all(isinstance(j, dict) and "job" in j and "seconds" in j for j in jobs):
            continue
        if abs(total - sum(j["seconds"] for j in jobs)) > 0.1:
            raise ValueError("Window budget and attempt durations differ")
        day = datetime.fromtimestamp(record["start"], timezone.utc).isoformat()
        counted = Counter()
        for job in jobs:
            name = job["job"]
            counted[name] += 1
            rows.append(
                (
                    day,
                    path.name,
                    digest(path),
                    name,
                    counted[name],
                    job["seconds"],
                    record["wall_seconds"],
                    total,
                )
            )
    with (paper / "native-compute.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "window_start_utc",
                "sealed_ledger",
                "ledger_sha256",
                "study",
                "attempt_in_window",
                "charged_seconds",
                "window_wall_seconds",
                "window_charged_seconds",
            ]
        )
        w.writerows(rows)
    recent = [r for r in rows if r[1] >= "window-1795703976.json"]
    windows = {r[1]: r for r in recent}
    if recent:
        times = sum(r[5] for r in recent) / 3600
        wall = sum(r[6] for r in windows.values()) / 3600
        count = len({r[3] for r in recent})
        message = (
            f"The closed November 26 extension windows cover {count} studies and {len(recent)} charged attempts. "
            f"The registered machine lease was occupied for {times:.2f} hours during {wall:.2f} hours of elapsed window time. "
            "These durations include native model work, CPU preparation and scoring inside the measured job. The supplement lists every charged attempt, including interrupted attempts. The figures exclude earlier phases, open windows and manuscript preparation.\n"
        )
        (paper / "compute-results.typ").write_text(message)
