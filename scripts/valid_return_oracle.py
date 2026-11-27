"""Independently replay the enrolled package metadata, calendar dates and file lookups."""

import argparse
import calendar
import csv
import io
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loopqwen.io import digest, read, save
from scripts.agenda_sources import events
from scripts.field_selection_oracle import select, wheel


def run(acquire=False):
    spec = read("research/experiments/valid-return-01.json")
    p = read(spec["material"])
    with ThreadPoolExecutor(max_workers=3) as pool:
        wheels = list(pool.map(lambda w: wheel(w, acquire), p["wheels"]))
    original = {w["id"]: w for w in events(2025)}
    for w in p["agenda"]:
        src = original[w["id"]]
        assert (
            all(src[k] == w[k] for k in ("uid", "room", "start", "end"))
            and src["title"].strip() == w["title"]
        )
    counters = Counter()
    for w in p["rows"]:
        for e in w.get("ordered_events", []):
            name = e["call"]["name"]
            a = e["call"]["arguments"]
            if name == "csv_lookup":
                raw = list(csv.reader(io.StringIO(p["documents"][a["document"]])))
                assert select(raw, a["key_column"], a["key"], a["value_column"]) == [
                    e["observation"]["value"]
                ]
            else:
                assert name == "weekday_number"
                year, month, day = map(int, a["date"].split("-"))
                assert e["observation"] == {
                    "value": calendar.weekday(year, month, day) + 1
                }
            counters[w["family"] + "." + name] += 1
    assert counters == {
        "file.csv_lookup": 36,
        "wheel.csv_lookup": 16,
        "agenda.csv_lookup": 16,
        "agenda.weekday_number": 8,
    }
    out = {
        "material": digest(Path(spec["material"])),
        "wheels": wheels,
        "reference_callbacks": dict(counters),
        "organizer_events_replayed": len(p["agenda"]),
    }
    path = Path("docs/measurements/valid-return-oracle.json")
    if path.exists():
        assert read(path) == out
    else:
        save(path, out)
    print(
        len(wheels),
        "embedded distribution metadata files;",
        sum(counters.values()),
        "actual callbacks verified",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true")
    run(parser.parse_args().acquire)
