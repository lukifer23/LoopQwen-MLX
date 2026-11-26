"""Freeze paired requests over real Python resources, fresh human arithmetic and independent public functions."""

import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from mlx_lm.utils import load_tokenizer
from program_reasoning_material import ASDIV_SHA, ASDIV_URL, former, norm, sha, source

from loopqwen.evaluation import executable_tools, program_questions
from loopqwen.evaluation import paired_tool_intents as intents
from loopqwen.evaluation.answers import numeric
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import initial_messages, prompt

DATA = Path("research/data")
PRIOR = "69f643c"
INPUTS = {
    "calendar": [
        {"date": s} for s in ("2024-02-29", "2025-07-14", "2026-11-26", "2023-12-31")
    ],
    "zone": [
        {"zone": z, "date": d}
        for z, d in (
            ("Europe/Paris", "2024-07-14"),
            ("Asia/Kathmandu", "2025-05-19"),
            ("Australia/Adelaide", "2024-01-17"),
            ("America/St_Johns", "2025-07-17"),
        )
    ],
    "unicode": [{"character": c} for c in ("Ω", "½", "π", "€")],
    "distribution": [
        {"distribution": s} for s in ("regex", "tqdm", "filelock", "truststore")
    ],
}
# Distinct phrasings per pair avoid making one repeated prompt determine an entire family.
FORMS = {
    "calendar": [
        (
            "Which ISO weekday number did {date} fall on?",
            "How many days into the year was {date}, counting January 1 as day one?",
        ),
        (
            "For {date}, give the day of the week as a number with Monday equal to 1.",
            "I'm indexing all dates in the year starting at 1. What index does {date} get?",
        ),
        (
            "Look up the ISO day of the week for {date}. I need the numeric code.",
            "What was the ordinal day within the year for {date}?",
        ),
        (
            "If Sunday is day seven, what weekday number is {date}?",
            "Counting days since the start of January inclusively, where does {date} land?",
        ),
    ],
    "zone": [
        (
            "At noon on {date} in {zone}, what's the total offset from UTC in minutes?",
            "For local noon on {date} in {zone}, how many minutes of the offset came from daylight saving?",
        ),
        (
            "How many minutes ahead of UTC was {zone} at noon on {date}?",
            "At noon in {zone} on {date}, what was the daylight-saving adjustment in minutes?",
        ),
        (
            "I need the UTC offset in minutes for local noon in {zone} on {date}.",
            "Within {zone} on {date}, how much extra time in minutes did summer time contribute at noon?",
        ),
        (
            "Give the signed offset east of UTC at local noon on {date} in {zone}. Use minutes.",
            "How many daylight-saving minutes applied in {zone} on {date} at local noon?",
        ),
    ],
    "unicode": [
        (
            "What's the official Unicode name for {character}?",
            "Give the two-letter Unicode general category of {character}.",
        ),
        (
            "How does the Unicode database spell the name of {character}?",
            "What general-category code classifies the character {character}?",
        ),
        (
            "Find the standardized character name for {character}.",
            "I need the two-letter character class for {character} from Unicode.",
        ),
        (
            "Look up {character} in Unicode and return the character's name.",
            "If I group Unicode characters by general category, which abbreviation applies to {character}?",
        ),
    ],
    "distribution": [
        (
            "Check which release of {distribution} is installed here. Give its version.",
            "How many files are listed by the installed {distribution} distribution?",
        ),
        (
            "What version number does the locally installed {distribution} report?",
            "Count the file entries in the local distribution metadata for {distribution}.",
        ),
        (
            "Find the installed version string for {distribution}.",
            "How many individual files does the installed package inventory list for {distribution}?",
        ),
        (
            "Read the version of {distribution} from this Python installation.",
            "Check {distribution}'s record of installed files. How many entries are there?",
        ),
    ],
}


def main():
    rng = random.Random(610931)
    environment = intents.installed_inventory()
    rows = []
    for family, entries in INPUTS.items():
        for j, args in enumerate(entries):
            names = (
                [n for n, _ in intents.OPERATIONS[family]]
                if j % 2 == 0
                else [f"read_{rng.getrandbits(24):06x}" for _ in range(2)]
            )
            order = rng.sample([0, 1], 2)
            pair = f"{family}-{j}"
            vals = [intents.read_value(family, i, args) for i in range(2)]
            assert vals[0] != vals[1]
            for i in range(2):
                question = FORMS[family][j][i].format(**args)
                rows.append(
                    dict(
                        id=f"{pair}-{i}",
                        provider="resource",
                        pair=pair,
                        family=family,
                        index=i,
                        args=args,
                        names=names,
                        order=order,
                        opaque=bool(j % 2),
                        question=question,
                        user_request=question,
                        gold=vals[i],
                    )
                )
    sv = read("research/experiments/checked-rationale-01.json")["sources"]["svamp"]
    ad = source("asdiv.xml", ASDIV_URL, ASDIV_SHA)
    sw = source(sv["file"], sv["url"], sv["sha256"])
    groups = {
        "ASDiv": [
            (
                " ".join(
                    (r.findtext("Body", "") + " " + r.findtext("Question", "")).split()
                ),
                numeric(re.sub(r"\s*\([^()]*\)\s*$", "", r.findtext("Answer", ""))),
            )
            for r in ET.parse(ad).iter("Problem")
        ],
        "SVAMP": [
            (r["Body"].strip() + " " + r["Question"].strip(), numeric(str(r["Answer"])))
            for r in read(sw)
        ],
    }
    old = former({q for group in groups.values() for q, _ in group}, prior=PRIOR)
    sv_openings = {tuple(norm(q)[:8]) for q, _ in groups["SVAMP"]}
    seen = set()
    for src, group in groups.items():
        selected = []
        for question, gold in sorted(
            group, key=lambda item: sha("semantic-resource-610931:" + item[0])
        ):
            opening = tuple(norm(question)[:8])
            if (
                not gold
                or question in old
                or opening in seen
                or (src == "ASDiv" and opening in sv_openings)
            ):
                continue
            seen.add(opening)
            selected.append(
                dict(
                    id=sha(question),
                    provider="arithmetic",
                    source=src,
                    question=question,
                    user_request=question + program_questions.INSTRUCTION,
                    gold=gold,
                )
            )
            if len(selected) == 12:
                break
        assert len(selected) == 12
        rows.extend(selected)
    for row in read(DATA / "tool-exec-bfcl-01.json")["rows"]:
        rows.append(
            dict(
                row,
                provider="bfcl_exec",
                user_request=row["question"][0][0]["content"]
                + executable_tools.INSTRUCTION,
            )
        )
    rng.shuffle(rows)
    tok = load_tokenizer(local_snapshot())
    panel = dict(
        rows=rows,
        prior=PRIOR,
        environment=environment,
        asdiv_sha256=digest(ad),
        svamp_sha256=digest(sw),
    )
    _, catalog, _ = intents.workload(panel, True)
    for row in rows:
        tools = catalog(row)
        row["initial_ids"] = tok.encode(
            prompt(tok, initial_messages(row["user_request"], tools)),
            add_special_tokens=False,
        )
        if row["provider"] == "resource":
            tool = next(t for t in tools if t.name == row["names"][row["index"]])
            assert tool.callback(**row["args"]) == {"value": row["gold"]}
    assert len(rows) == len({w["id"] for w in rows}) == 78
    save(DATA / "paired-intent-panel.json", panel)


if __name__ == "__main__":
    main()
