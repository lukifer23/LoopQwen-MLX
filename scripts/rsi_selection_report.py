"""Reconstruct candidate priorities from archived learner predictions and verify chosen IDs."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from rsi_archive import read


def report(root):
    def salted(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    spec = read(root / "registration.json")
    pools = read(root / "pools.json.gz")
    result = {}
    for path in sorted(root.glob("adaptive*-diagnosis.json.gz")):
        name = path.name.split("-")[0]
        n = int(name[-1])
        pairs = defaultdict(list)
        for record in read(path)["records"]:
            pairs[record["map_id"]].append(record)
        choices = {}
        for arm in ("adaptive", "static"):
            chosen = read(root / f"{arm}{n}-curriculum.json")["selected"]
            choices[arm] = {r["id"]: r["reason"] for r in chosen}
        rows = []
        for item in pools[f"pool{n}"]:
            key = item["map_id"]
            a, b = pairs[key]
            dist = (a["true_p"][-1] + b["true_p"][-1]) / 2
            disagreement = a["pred"] != b["pred"]
            if a["prefix"] == 6 and b["prefix"] == 6:
                category, priority = "mastered", dist
            elif disagreement or a["prefix"] in (4, 5) or b["prefix"] in (4, 5):
                category, priority = "boundary", abs(dist - 0.5)
            else:
                category, priority = "failure", -max(a["prefix"], b["prefix"])
            rows.append(
                dict(
                    map_id=key,
                    stratum=item["stratum"],
                    category=category,
                    rank_value=priority,
                    order_disagrees=disagreement,
                    static_priority=salted(["static", spec["seed"], n, key]),
                    adaptive_selection=choices["adaptive"].get(key),
                    static_selected=key in choices["static"],
                )
            )
        for stratum in {r["stratum"] for r in rows}:
            group = [r for r in rows if r["stratum"] == stratum]
            selected = []
            extra = []
            for cat, quota in [("boundary", 5), ("failure", 2), ("mastered", 1)]:
                ranked = sorted(
                    (r for r in group if r["category"] == cat),
                    key=lambda r: (r["rank_value"], r["map_id"]),
                )
                selected.extend(ranked[:quota])
                extra.extend(ranked[quota:])
            selected += extra[: 8 - len(selected)]
            if {r["map_id"] for r in selected} != {
                r["map_id"] for r in group if r["adaptive_selection"]
            }:
                raise ValueError("Adaptive rank audit")
            if {
                r["map_id"]
                for r in sorted(group, key=lambda r: r["static_priority"])[:8]
            } != {r["map_id"] for r in group if r["static_selected"]}:
                raise ValueError("Static rank audit")
        result[name] = rows
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    record = report(a.archive)
    if a.report.exists():
        if read(a.report) != record:
            raise ValueError("Saved selection report differs")
    else:
        a.report.write_bytes(
            gzip.compress(json.dumps(record, separators=(",", ":")).encode(), mtime=0)
        )
    print("Verified", sum(map(len, record.values())), "candidate ranks")
