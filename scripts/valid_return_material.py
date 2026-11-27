"""Freeze human judgments and genuinely different successful returns on identical records."""

import itertools
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from loopqwen import qwen_tools
from loopqwen.csv_tools import tools as csv_tools
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.reasoned_evidence_data import choose, table, text
from loopqwen.resource_tools import tools as local_tools
from scripts.agenda_sources import events
from scripts.field_selection_material import catalog, observed
from scripts.field_selection_sources import take_tables

SEED = 210764
TRAIN = Path("research/data/valid-return-training.json")
PANEL = TRAIN.with_name("valid-return-panel.json")
WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
LABELS = ("true", "false", "uncertain")


def agenda(year, n, blocked):
    rows = choose(events(year), SEED, lambda w: w["id"])
    selected = []
    for w in rows:
        name = w["title"].strip()
        if (
            not 8 <= len(name) <= 90
            or name.casefold() in blocked
            or any(c in name for c in "\r\n<>")
        ):
            continue
        blocked.add(name.casefold())
        selected.append({**w, "title": name, "date": w["start"][:10]})
        if len(selected) == n:
            break
    assert len(selected) == n
    return selected


def agenda_doc(items):
    return table(
        ["Session", "Room", "Date", "Start"],
        [(w["title"], w["room"], w["date"], w["start"][11:16]) for w in items],
    )


def meeting(w, peers, kind, truth):
    name = w["title"]
    when = date.fromisoformat(w["date"])
    if kind == "weekday":
        day = when.weekday()
        proposed = WEEKDAYS[day if truth else (day + 1) % 7]
        return (
            text(
                f'Use the agenda file and calendar service to check the scheduled weekday for "{name}".',
                f"This session was scheduled on {proposed}.",
            ),
            "Date",
            w["date"],
        )
    alt = next(a["room"] for a in peers if a["room"] != w["room"])
    room = w["room"] if truth else alt
    return (
        text(
            f'Check the venue entry for "{name}" in the agenda.',
            f"The listed room was {room}.",
        ),
        "Room",
        w["room"],
    )


def tool_set(w, folder):
    if not w["document"]:
        return []
    offered = csv_tools([folder / w["document"]], keyed=True)
    if w["family"] == "agenda":
        offered.extend(local_tools("calendar"))
    return offered


def weekday_event(iso):
    tool = local_tools("calendar")[0]
    call = {"name": tool.name, "arguments": {"date": iso}}
    obs = tool.callback(**call["arguments"])
    assert obs == {"value": date.fromisoformat(iso).isoweekday()}
    return {"call": call, "observation": obs}


def build():
    old_training = read("research/data/field-selection-training.json")
    old_panel = read("research/data/field-selection-panel.json")
    human = take_tables(held_split="test_alpha1", sample_seed=SEED)["test_alpha1"]
    old = old_training["tables"] + old_panel["tables"]
    assert not {w["id"] for w in human} & {w["id"] for w in old}
    assert not {w["fields"]["title"].casefold() for w in human} & {
        w["fields"]["title"].casefold() for w in old
    }
    assert not {q["claim"].strip().casefold() for w in human for q in w["claims"]} & {
        q["claim"].strip().casefold() for w in old for q in w["claims"]
    }
    excluded = {
        w["package"].casefold() for w in old_panel["wheels"] + old_training["packages"]
    }
    pool = read("research/sources/field-selection/inventory.json")
    inventory = [w for w in pool if w["package"].casefold() not in excluded]
    patches = [
        "3.5.10",
        "3.6.15",
        "3.7.17",
        "3.8.20",
        "3.9.20",
        "3.10.15",
        "3.11.10",
        "3.12.7",
    ]

    def guards(w):
        spec = SpecifierSet(w["requirement"])
        return [
            (a, b)
            for a, b in itertools.pairwise(patches)
            if Version(a) not in spec and Version(b) in spec
        ]

    wheels = choose(
        [w for w in inventory if len(guards(w)) == 1], SEED, lambda w: w["package"]
    )[:8]
    assert len(wheels) == 8
    blocked = set()
    held_agenda = agenda(2025, 16, blocked)
    train_agenda = agenda(2023, 32, blocked) + agenda(2024, 32, blocked)
    train = {
        "tables": choose(old_training["tables"], SEED, lambda w: w["id"])[:64],
        "packages": old_training["packages"],
        "agenda": train_agenda,
        "explanations": [
            w
            for label in LABELS
            for w in choose(
                [
                    r
                    for r in read("research/data/proof-curriculum-training.json")[
                        "human"
                    ]
                    if r["label"] == label
                ],
                SEED,
                lambda w: w["id"],
            )[:32]
        ],
    }
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    rows = []
    docs = {}

    def emit(ident, entity, family, question, label, name=None, expected=None):
        row = {
            "id": ident,
            "cluster": entity,
            "family": family,
            "questions": [question],
            "gold": label,
            "document": name,
            "calls": [] if not expected else [expected[0]["call"]],
            "observation": None if not expected else expected[0]["observation"],
            "ordered_events": expected or [],
        }
        tools = tool_set(row, tmp)
        row["initial_ids"] = tok.encode(
            qwen_tools.render(
                tok,
                [{"role": "user", "content": question}],
                qwen_tools.declarations(tools),
            ),
            add_special_tokens=False,
        )
        rows.append(row)

    with TemporaryDirectory() as work:
        tmp = Path(work)
        for w in human:
            f = w["fields"]
            name = w["id"] + "-catalog.csv"
            raw = catalog(w, human)
            docs[name] = raw
            for q in w["claims"]:
                rid = fingerprint([w["id"], q["claim"]])[:16]
                emit(
                    rid + "-inline",
                    w["id"],
                    "inline",
                    text("\n".join(f"{k}: {v}" for k, v in f.items()), q["claim"]),
                    q["label"],
                )
                prompt = text(
                    f'Consult the attached table for "{f["title"]}".', q["claim"]
                )
                _, call, obs, _ = observed(
                    tmp,
                    name,
                    raw,
                    "title",
                    f["title"],
                    q["field"],
                    f[q["field"]],
                    prompt,
                )
                emit(
                    rid + "-file",
                    w["id"],
                    "file",
                    prompt,
                    q["label"],
                    name,
                    [{"call": call, "observation": obs}],
                )
        name = "release-manifest.csv"
        raw = table(
            ["Distribution", "Published version", "Requires-Python"],
            [(w["package"], w["version"], w["requirement"]) for w in inventory],
        )
        docs[name] = raw
        for w in wheels:
            for ver in guards(w)[0]:
                prompt = text(
                    f'Consult the published interpreter constraint for "{w["package"]}" in the release manifest.',
                    f"Python {ver} is permitted by {w['package']} {w['version']}.",
                )
                _, call, obs, _ = observed(
                    tmp,
                    name,
                    raw,
                    "Distribution",
                    w["package"],
                    "Requires-Python",
                    w["requirement"],
                    prompt,
                )
                label = (
                    "true"
                    if Version(ver) in SpecifierSet(w["requirement"])
                    else "false"
                )
                emit(
                    w["package"] + "-" + ver,
                    w["package"],
                    "wheel",
                    prompt,
                    label,
                    name,
                    [{"call": call, "observation": obs}],
                )
        for j, w in enumerate(held_agenda[:8]):
            peers = held_agenda[j // 4 * 4 : j // 4 * 4 + 4] + held_agenda[8:12]
            raw = agenda_doc(peers)
            name = f"brussels-program-{j // 4}.csv"
            docs[name] = raw
            for kind in ("weekday", "room"):
                truth = (j % 2 == 0) == (kind == "weekday")
                prompt, field, val = meeting(w, peers, kind, truth)
                _, c, o, _ = observed(
                    tmp, name, raw, "Session", w["title"], field, val, prompt
                )
                ev = [{"call": c, "observation": o}]
                if kind == "weekday":
                    ev.append(weekday_event(w["date"]))
                emit(
                    w["id"] + "-" + kind,
                    w["id"],
                    "agenda",
                    prompt,
                    "true" if truth else "false",
                    name,
                    ev,
                )
        for w in old_panel["rows"]:
            if w["family"] == "retention":
                rows.append(w)
                docs[w["document"]] = old_panel["documents"][w["document"]]
    assert len(rows) == 112 and len({w["id"] for w in rows}) == 112
    for dest, material in (
        (TRAIN, train),
        (
            PANEL,
            {
                "rows": choose(rows, SEED, lambda w: w["id"]),
                "documents": docs,
                "tables": human,
                "wheels": wheels,
                "agenda": held_agenda,
            },
        ),
    ):
        if dest.exists():
            assert read(dest) == material
        else:
            save(dest, material)
    print(
        "Frozen 112 direct requests, 64 human training tables and 64 actual schedule records",
        flush=True,
    )


if __name__ == "__main__":
    build()
