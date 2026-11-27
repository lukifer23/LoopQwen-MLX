"""Pair two successful calls to the same real row and teach recovery from an irrelevant value."""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from .. import qwen_tools
from ..csv_tools import tools as csv_tools
from ..io import read
from ..resource_tools import tools as local_tools
from ..tool_session import compact
from .reasoned_evidence_data import choose, table, text


def turns(spec, root):
    from scripts.field_selection_material import catalog
    from scripts.valid_return_material import (
        LABELS,
        SEED,
        WEEKDAYS,
        agenda_doc,
        meeting,
    )

    t = read(spec["training"])
    examples = []

    def add(uid, kind, src, msgs, defs, target, negative=None):
        for role, reply in [
            ("preferred", target),
            *([("rejected", negative)] if negative is not None else []),
        ]:
            examples.append(
                {
                    "id": uid + "-" + kind + "-" + role,
                    "pair": uid + "-" + kind,
                    "source": src,
                    "kind": kind,
                    "choice": role,
                    "messages": msgs,
                    "tools": defs,
                    "target": reply,
                }
            )

    def ans(label, reason):
        return {"role": "assistant", "content": label, "reasoning_content": reason}

    def value_message(obs):
        return {"role": "tool", "content": compact(obs)}

    def actions(
        path,
        raw,
        keycol,
        key,
        field,
        other,
        prompt,
        ident,
        src,
        expected,
        *,
        with_calendar=False,
    ):
        path.write_bytes(raw.encode())
        tool = csv_tools([path], keyed=True)[0]
        offered = [tool, *local_tools("calendar")] if with_calendar else [tool]
        defs = qwen_tools.declarations(offered)
        calls = []
        obs = []
        for k in (field, other):
            args = {
                "document": path.name,
                "key_column": keycol,
                "key": key,
                "value_column": k,
            }
            calls.append(
                qwen_tools.assistant_message([{"name": tool.name, "arguments": args}])
            )
            obs.append(tool.callback(**args))
        assert (
            obs[0] == {"value": expected}
            and obs[1] != obs[0]
            and all("value" in v for v in obs)
        )
        user = [{"role": "user", "content": prompt}]
        add(ident, "field", src, user, defs, calls[0], calls[1])
        add(
            ident,
            "repair",
            src,
            [*user, calls[1], value_message(obs[1])],
            defs,
            calls[0],
        )
        return defs, [*user, calls[0], value_message(obs[0])]

    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        for w in t["tables"]:
            fields = w["fields"]
            name = fields["title"]
            raw = catalog(w, t["tables"])
            for i, q in enumerate(w["claims"]):
                field = q["field"]
                alternatives = [
                    k
                    for k in fields
                    if k != "title"
                    and k != field
                    and fields[k]
                    and fields[k] != fields[field]
                ]
                alt = next(
                    (a["field"] for a in w["claims"] if a["field"] in alternatives),
                    choose(alternatives, SEED, str)[0],
                )
                source = text(f'Consult the table for "{name}".', q["claim"])
                ident = f"{w['id']}-{i}"
                defs, hist = actions(
                    tmp / "catalog.csv",
                    raw,
                    "title",
                    name,
                    field,
                    alt,
                    source,
                    ident,
                    "human",
                    fields[field],
                )
                reason = f"The {field} field for {name} gives {fields[field]}."
                truth = ans(q["label"], reason)
                other = ans(LABELS[(LABELS.index(q["label"]) + 1 + i % 2) % 3], reason)
                add(ident, "verdict", "human", hist, defs, truth, other)
                inline = text(
                    "\n".join(f"{k}: {v}" for k, v in fields.items()), q["claim"]
                )
                add(
                    ident + "-inline",
                    "verdict",
                    "human",
                    [{"role": "user", "content": inline}],
                    [],
                    truth,
                    other,
                )
        packages = t["packages"]
        unique = {w["package"]: w for w in packages}
        for i, w in enumerate(packages):
            keycol = ("Name", "package", "project", "Distribution name")[i % 4]
            release = ("Version", "Version", "Published release", "release")[i % 4]
            fields = [keycol, release, "Requires-Python"]
            raw = table(
                fields, [(k, v["release"], v["requirement"]) for k, v in unique.items()]
            )
            key = w["package"]
            # A successful return of the wrong field is a hard negative in each direction.
            for kind in ("requirement", "release"):
                if kind == "requirement":
                    prompt = text(
                        f'Check the allowed Python versions for "{key}" in the package catalog.',
                        w["claim"],
                    )
                    field, foil, value = fields[2], fields[1], w["requirement"]
                    truth = w["label"]
                    reason = w["reason"]
                else:
                    equal = i % 2 == 0
                    proposed = (
                        w["release"]
                        if equal
                        else next(
                            u["release"]
                            for u in unique.values()
                            if u["release"] != w["release"]
                        )
                    )
                    prompt = text(
                        f'Check the published release of "{key}" in the package catalog.',
                        f"The published version number is {proposed}.",
                    )
                    field, foil, value = fields[1], fields[2], w["release"]
                    truth = "true" if equal else "false"
                    reason = f"The release for {key} is {value}. The proposed version is {proposed}."
                ident = w["id"] + "-" + kind
                defs, hist = actions(
                    tmp / "versions.csv",
                    raw,
                    keycol,
                    key,
                    field,
                    foil,
                    prompt,
                    ident,
                    "package",
                    value,
                )
                add(
                    ident,
                    "verdict",
                    "package",
                    hist,
                    defs,
                    ans(truth, reason),
                    ans("false" if truth == "true" else "true", reason),
                )
        for i, w in enumerate(t["agenda"]):
            # Mix events across rooms in each authentic catalog page.
            peers = t["agenda"][i // 8 * 8 : i // 8 * 8 + 8]
            raw = agenda_doc(peers)
            for kind in ("weekday", "room"):
                truth = (i % 2 == 0) == (kind == "weekday")
                prompt, field, value = meeting(w, peers, kind, truth)
                ident = w["id"] + "-" + kind
                foil = "Room" if kind == "weekday" else "Date"
                defs, hist = actions(
                    tmp / "meeting-program.csv",
                    raw,
                    "Session",
                    w["title"],
                    field,
                    foil,
                    prompt,
                    ident,
                    "agenda",
                    value,
                    with_calendar=True,
                )
                if kind == "weekday":
                    cal, ordinal = local_tools("calendar")
                    args = {"date": value}
                    right = qwen_tools.assistant_message(
                        [{"name": cal.name, "arguments": args}]
                    )
                    wrong = qwen_tools.assistant_message(
                        [{"name": ordinal.name, "arguments": args}]
                    )
                    result = cal.callback(**args)
                    assert result != ordinal.callback(**args)
                    add(ident, "operation", "agenda", hist, defs, right, wrong)
                    hist = [*hist, right, value_message(result)]
                    reason = f"The event is dated {value}. ISO weekday {result['value']} is {WEEKDAYS[date.fromisoformat(value).weekday()]}."
                else:
                    reason = (
                        f"The room assigned to this session in the program is {value}."
                    )
                label = "true" if truth else "false"
                add(
                    ident,
                    "verdict",
                    "agenda",
                    hist,
                    defs,
                    ans(label, reason),
                    ans("false" if truth else "true", reason),
                )
    for i, w in enumerate(t["explanations"]):
        reason = w["reason"]
        y = w["label"]
        add(
            w["id"],
            "verdict",
            "rationale",
            [{"role": "user", "content": text(w["evidence"], w["claim"])}],
            [],
            ans(y, reason),
            ans(LABELS[(LABELS.index(y) + 1 + i % 2) % 3], reason),
        )
    for w in read("research/data/cross-view-training.json")["examples"]:
        if w["kind"] == "rehearsal":
            examples.append(
                {**w, "pair": w["id"], "source": "retention", "choice": "preferred"}
            )
    assert len({r["id"] for r in examples}) == len(examples)
    return examples
