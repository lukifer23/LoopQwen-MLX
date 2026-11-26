"""Score changes of requested property on actual calendar, time-zone, Unicode and package data."""

import re
from datetime import date
from importlib.metadata import PackageNotFoundError, distributions

from ..resource_tools import (
    OPERATIONS,
    installed_distribution,
    read_value,
    site_packages,
)
from ..resource_tools import tools as resource_tools
from . import executable_tools, program_questions
from .statistics import exact_mcnemar


def installed_inventory():
    from packaging.utils import canonicalize_name

    names = {
        canonicalize_name(d.metadata["Name"])
        for d in distributions(path=site_packages())
        if d.metadata.get("Name")
    }
    # Resolve duplicates exactly as inference resolves a request in this interpreter.
    result = {}
    for name in sorted(names):
        package = installed_distribution(name)
        result[name] = [package.version, len(package.files or ())]
    return result


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
    def lookup(family, index, kwargs):
        if replay and family == "distribution":
            from packaging.utils import canonicalize_name

            name = canonicalize_name(kwargs["distribution"])
            if name not in replay:
                raise PackageNotFoundError(name)
            return replay[name][index]
        return read_value(family, index, kwargs)

    return resource_tools(
        row["family"], names=row["names"], order=row["order"], lookup=lookup
    )


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
