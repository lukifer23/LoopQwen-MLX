"""Cross-check public conference times in the organizer's independent XML and calendar exports."""

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen
from xml.etree.ElementTree import parse
from zoneinfo import ZoneInfo

from loopqwen.io import digest

CACHE = Path("data/research_external/fosdem")
FILES = {
    2023: {
        "xml": "36003f37f7a218bd0c15df3e36d1c1dad168db33ed89cd2160ba1906fc986a29",
        "ical": "f2499a48ebb705e63232eca5288ab2622da6d565b181d769533686dc5df8576f",
    },
    2024: {
        "xml": "db4c47980d93ab4a613cd99d5ac8416fa9bec41dba582223c45023543508674e",
        "ical": "0297d02f5aaaf342856db4bf60c4eda279090a8a141ea8d86060d6afdc1893fe",
    },
    2025: {
        "xml": "958c70712f080ba4e55a35f9eb6bbec9e00c7e4482f7e2724d894ccc3cb45d94",
        "ical": "f6daf12a6dbac3a1d921cc1ace5e451a6b59f40df6092c5de232b3d5e212cbfc",
    },
}


def source(year, fmt, acquire):
    path = CACHE / f"{year}.{fmt}"
    if not path.exists() and acquire:
        CACHE.mkdir(parents=True, exist_ok=True)
        trial = path.with_suffix(path.suffix + ".partial")
        with urlopen(
            f"https://archive.fosdem.org/{year}/schedule/{fmt}", timeout=45
        ) as response:
            trial.write_bytes(response.read())
        assert digest(trial) == FILES[year][fmt]
        trial.replace(path)
    assert digest(path) == FILES[year][fmt]
    return path


def events(year, acquire=False):
    xml = source(year, "xml", acquire)
    calendar = source(year, "ical", acquire)
    # The organizer emits local times in Brussels and identifies the same event by
    # its decimal conference ID before the first @ in UID. Unfold RFC 5545 lines.
    raw = re.sub(rb"\r?\n[ \t]", b"", calendar.read_bytes()).decode("utf-8")
    scheduled = {}
    for event in raw.split("BEGIN:VEVENT")[1:]:
        section = event.split("END:VEVENT", 1)[0].strip()
        values = dict(
            line.split(":", 1) for line in section.splitlines() if ":" in line
        )
        key = values["UID"].split("@", 1)[0]
        assert key not in scheduled
        scheduled[key] = values
    output = []
    doc = parse(xml)
    zone = ZoneInfo("Europe/Brussels")
    for day in doc.findall("day"):
        for e in day.findall("room/event"):
            i = e.get("id")
            ref = scheduled[i]
            start = datetime.fromisoformat(
                day.get("date") + "T" + e.findtext("start")
            ).replace(tzinfo=zone)
            h, m = map(int, e.findtext("duration").split(":"))
            end = start + timedelta(hours=h, minutes=m)
            assert (
                start.strftime("%Y%m%dT%H%M%S") == ref["DTSTART"]
                and end.strftime("%Y%m%dT%H%M%S") == ref["DTEND"]
            )
            assert e.findtext("room") == ref["LOCATION"] and start < end
            if e.findtext("date"):
                assert datetime.fromisoformat(e.findtext("date")) == start
            output.append(
                {
                    "id": f"{year}-{i}",
                    "title": e.findtext("title"),
                    "room": ref["LOCATION"],
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "uid": ref["UID"],
                    "track": e.findtext("track"),
                    "url": f"https://archive.fosdem.org/{year}/schedule/event/{e.findtext('slug')}/",
                }
            )
    assert len(output) == len(scheduled) == len({w["id"] for w in output})
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acquire", action="store_true")
    args = p.parse_args()
    for year in FILES:
        rows = events(year, args.acquire)
        print(
            year, len(rows), "cross-checked event IDs, start times, end times and rooms"
        )
