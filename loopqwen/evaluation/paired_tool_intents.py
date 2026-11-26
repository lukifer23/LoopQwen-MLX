"""Score changes of requested property on actual calendar, time-zone, Unicode and package data."""

import re
import unicodedata
from datetime import date, datetime
from importlib.metadata import distribution, distributions
from zoneinfo import ZoneInfo

from ..tool_session import Tool
from . import executable_tools, program_questions
from .statistics import exact_mcnemar

# Every operation in a pair accepts the same arguments. The caller supplies the
# exposed names and order. Values come from the live provider during inference.
OPERATIONS = {
    "calendar": (
        ("weekday_number", "ISO weekday number from Monday (1) to Sunday (7)."),
        ("day_of_year", "One-based ordinal day within the calendar year."),
    ),
    "zone": (
        (
            "utc_offset_minutes",
            "Time zone's total offset east of UTC in minutes at local noon.",
        ),
        ("daylight_minutes", "Minutes contributed by daylight saving at local noon."),
    ),
    "unicode": (
        ("character_name", "Official Unicode name of the character."),
        ("general_category", "Two-letter Unicode general category of the character."),
    ),
    "distribution": (
        ("installed_version", "Installed software distribution version string."),
        (
            "number_of_files",
            "Number of entries in the installed distribution's file inventory.",
        ),
    ),
}
ARGUMENTS = {
    "calendar": {"date": "Local date in ISO YYYY-MM-DD format."},
    "zone": {
        "zone": "IANA time zone identifier.",
        "date": "Local date in ISO YYYY-MM-DD format.",
    },
    "unicode": {"character": "A single Unicode character."},
    "distribution": {"distribution": "Installed Python distribution name."},
}


def installed_inventory():
    from packaging.utils import canonicalize_name

    names = {
        canonicalize_name(d.metadata["Name"])
        for d in distributions()
        if d.metadata.get("Name")
    }
    # Resolve duplicates exactly as inference resolves a request in this interpreter.
    result = {}
    for name in sorted(names):
        package = distribution(name)
        result[name] = [package.version, len(package.files or ())]
    return result


def read_value(family, index, kwargs):
    if family == "calendar":
        day = date.fromisoformat(kwargs["date"])
        return day.isoweekday() if index == 0 else day.timetuple().tm_yday
    if family == "zone":
        day = date.fromisoformat(kwargs["date"])
        value = datetime(
            day.year, day.month, day.day, 12, tzinfo=ZoneInfo(kwargs["zone"])
        )
        delta = value.utcoffset() if index == 0 else value.dst()
        return int(delta.total_seconds() / 60)
    if family == "unicode":
        character = kwargs["character"]
        return (
            unicodedata.name(character)
            if index == 0
            else unicodedata.category(character)
        )
    if family == "distribution":
        package = distribution(kwargs["distribution"])
        return package.version if index == 0 else len(package.files or ())
    raise ValueError("Unknown resource family")


def same_resource(family, supplied, expected):
    from packaging.utils import canonicalize_name

    if supplied.keys() != expected.keys():
        return False
    if family == "distribution":
        return canonicalize_name(supplied["distribution"]) == canonicalize_name(
            expected["distribution"]
        )
    if family in ("zone", "calendar"):
        try:
            return date.fromisoformat(supplied["date"]) == date.fromisoformat(
                expected["date"]
            ) and supplied.get("zone") == expected.get("zone")
        except (TypeError, ValueError):
            return False
    return supplied == expected


def tools_for(row, replay):
    family = row["family"]
    properties = {
        k: {"type": "string", "description": v} for k, v in ARGUMENTS[family].items()
    }
    schema = dict(
        type="object",
        properties=properties,
        required=list(properties),
        additionalProperties=False,
    )

    def callback(index):
        def lookup(**kwargs):
            if replay and family == "distribution":
                # Preserve the full enrolled environment, including a model's wrong lookup.
                from importlib.metadata import PackageNotFoundError

                from packaging.utils import canonicalize_name

                name = canonicalize_name(kwargs["distribution"])
                if name not in replay:
                    raise PackageNotFoundError(name)
                value = replay[name][index]
            else:
                value = read_value(family, index, kwargs)
            return {"value": value}

        return lookup

    return [
        Tool(row["names"][i], OPERATIONS[family][i][1], schema, callback(i))
        for i in row["order"]
    ]


def workload(material, replay=False):
    items = material["rows"]
    if not replay and installed_inventory() != material["environment"]:
        raise ValueError(
            "Installed distributions differ from the enrolled resource inventory"
        )

    def catalog(row):
        if row["provider"] == "bfcl_exec":
            return executable_tools.tools_for(row)
        if row["provider"] == "arithmetic":
            return program_questions.tools_for(row)
        return tools_for(row, material["environment"] if replay else None)

    def grade(row, trajectory):
        if row["provider"] in ("bfcl_exec", "arithmetic"):
            scorer = (
                executable_tools
                if row["provider"] == "bfcl_exec"
                else program_questions
            )
            report = scorer.score(row, trajectory)
            return {
                k: report[k] for k in ("success", "path", "correct_answer", "completed")
            }
        turns = [t for t in trajectory["turns"] if "observation" in t]
        value = str(row["gold"])
        correct = bool(
            re.search(
                r"(?<![\w.\-])" + re.escape(value) + r"(?!\w|\.\d)",
                trajectory["answer"] or "",
                re.I,
            )
        )
        selected = any(
            t["call"]["name"] == row["names"][row["index"]]
            and t["observation"] == {"value": row["gold"]}
            and same_resource(row["family"], t["call"]["arguments"], row["args"])
            for t in turns
        )
        done = trajectory["status"] == "answered"
        path = selected and len(turns) == 1
        return dict(
            success=path and correct and done,
            path=path,
            correct_answer=correct and done,
            completed=done,
        )

    return items, catalog, grade


def decision(records, spec, metrics):
    from pathlib import Path

    from ..io import read
    from .mechanism import unpack

    rows = read(spec["material"])["rows"]
    groups = {}
    pair_scores = {}
    aliases = {
        row["pair"]: row["opaque"] for row in rows if row["provider"] == "resource"
    }
    multiple_calls = {}
    for side, arm in records.items():
        groups[side] = {}
        pairs = {}
        multiple_calls[side] = 0
        for row, report in zip(rows, arm, strict=True):
            assert (
                row["id"] == report["id"]
                and row["initial_ids"] == report["trajectory"]["turns"][0]["prompt_ids"]
            )
            family = row.get("family", row.get("source", row["provider"]))
            cell = groups[side].setdefault(
                family, dict(n=0, strict=0, answer=0, seconds=0.0)
            )
            cell["n"] += 1
            cell["strict"] += report["score"]["success"]
            cell["answer"] += report["score"]["correct_answer"]
            cell["seconds"] += report["trajectory"]["seconds"]
            if row["provider"] == "resource":
                multiple_calls[side] += (
                    sum("observation" in turn for turn in report["trajectory"]["turns"])
                    > 1
                )
                pairs.setdefault(row["pair"], {})[row["index"]] = report["score"][
                    "success"
                ]
        assert len(pairs) == 16 and all(
            set(group) == {0, 1} for group in pairs.values()
        )
        pair_scores[side] = {k: all(v.values()) for k, v in pairs.items()}
    a, b = (pair_scores[s] for s in ("candidate", "control"))
    wins = sum(v and not b[k] for k, v in a.items())
    losses = sum(v and not a[k] for k, v in b.items())
    p = exact_mcnemar(wins, losses)
    c, t = (groups[s] for s in ("candidate", "control"))
    previous = unpack(Path(spec["initial_adapter"]) / "candidate.json.gz")
    previous_success = {r["id"] for r in previous if r["score"]["success"]}
    public = {
        r["id"]: r["score"]["success"]
        for w, r in zip(rows, records["candidate"], strict=True)
        if w["provider"] == "bfcl_exec"
    }
    assert len(previous_success) == 17 and set(public) == {r["id"] for r in previous}
    regressions = sorted(k for k in previous_success if not public[k])
    passed = (
        wins - losses >= spec["pair_gain"]
        and p <= 0.05
        and all(
            c[k]["strict"] >= 4 and c[k]["strict"] > t[k]["strict"] for k in OPERATIONS
        )
        and all(c[k]["answer"] >= t[k]["answer"] for k in ("ASDiv", "SVAMP"))
        and not regressions
        and metrics["candidate"]["peak_bytes"] <= metrics["control"]["peak_bytes"]
    )
    return dict(
        paired_correct={s: sum(v.values()) for s, v in pair_scores.items()},
        gains=wins,
        losses=losses,
        paired_p=p,
        groups=groups,
        public_regressions=regressions,
        resource_multiple_calls=multiple_calls,
        correct_pairs_by_name={
            s: {
                label: sum(v for k, v in results.items() if aliases[k] == opaque)
                for label, opaque in (("descriptive", False), ("opaque", True))
            }
            for s, results in pair_scores.items()
        },
    ), passed
