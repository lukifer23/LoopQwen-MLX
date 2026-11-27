"""Freeze opposite-role lookups, same-table verdict triplets and a fresh direct comparison."""

import re
from email.parser import BytesParser
from pathlib import Path
from tempfile import TemporaryDirectory

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from loopqwen import qwen_tools
from loopqwen.csv_tools import tools
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.reasoned_evidence_data import choose, table, text
from loopqwen.tool_session import compact
from scripts.field_selection_sources import SEED, roles, take_tables

DIR = Path("research/data")
TRAIN = DIR / "field-selection-training.json"
PANEL = DIR / "field-selection-panel.json"


def observed(tmp, fname, raw, key_col, key, field, value, question):
    path = tmp / fname
    path.write_bytes(raw.encode())
    provider = tools([path], keyed=True)
    args = {"document": fname, "key_column": key_col, "key": key, "value_column": field}
    call = {"name": provider[0].name, "arguments": args}
    obs = provider[0].callback(**args)
    assert obs == {"value": value}
    user = [{"role": "user", "content": question}]
    history = [
        *user,
        qwen_tools.assistant_message([call]),
        {"role": "tool", "content": compact(obs)},
    ]
    return qwen_tools.declarations(provider), call, obs, history


def catalog(w, peers):
    fields = w["fields"]
    others = sorted(
        (p for p in peers if p["id"] != w["id"]),
        key=lambda p: (-len(fields.keys() & p["fields"].keys()), fingerprint(p["id"])),
    )
    records = choose([w, *others[:2]], SEED, lambda p: p["id"])
    return table(
        list(fields), [[r["fields"].get(k, "") for k in fields] for r in records]
    )


def training_views(train, tmp):
    examples = []

    def add(ident, group, kind, label, defs, history, target):
        examples.append(
            {
                "id": ident + "-" + kind,
                "group": group,
                "kind": kind,
                "label": label,
                "tools": defs,
                "messages": history,
                "target": target,
            }
        )

    def answer(label, reason):
        return {"role": "assistant", "content": label, "reasoning_content": reason}

    for w in train["tables"]:
        fields = w["fields"]
        name = fields["title"]
        raw = catalog(w, train["tables"])
        fname = "reference-table.csv"
        for i, q in enumerate(w["claims"]):
            label = q["label"]
            ident = w["id"] + "-" + str(i)
            field = q["field"]
            req = text(f'Consult the attached table for "{name}".', q["claim"])
            defs, call, _, history = observed(
                tmp, fname, raw, "title", name, field, fields[field], req
            )
            reason = f"The {field} field for {name} gives {fields[field]}."
            tgt = answer(label, reason)
            add(ident, w["id"], "file", label, defs, history, tgt)
            inline = text("\n".join(f"{k}: {v}" for k, v in fields.items()), q["claim"])
            add(
                ident,
                w["id"],
                "inline",
                label,
                [],
                [{"role": "user", "content": inline}],
                tgt,
            )
            add(
                ident,
                w["id"],
                "opening",
                label,
                defs,
                history[:1],
                qwen_tools.assistant_message([call]),
            )
    for w in train["roles"]:
        raw = table(w["header"], w["rows"])
        fname = "records.csv"
        for i, (a, b, x, y) in enumerate(
            ((w["a"], w["b"], w["x"], w["y"]), (w["b"], w["a"], w["y"], w["x"]))
        ):
            ident = "wiki-" + w["id"] + "-" + str(i)
            req = f'Find the {b} in the record whose {a} is "{x}".'
            defs, call, _, hist = observed(tmp, fname, raw, a, x, b, y, req)
            action = qwen_tools.assistant_message([call])
            add(ident, w["id"], "direction", "", defs, hist[:1], action)
            wrong = {
                "name": call["name"],
                "arguments": {**call["arguments"], "key_column": b, "value_column": a},
            }
            callback = tools([tmp / fname], keyed=True)[0].callback
            try:
                callback(**wrong["arguments"])
            except ValueError as exc:
                error = {
                    "role": "tool",
                    "content": compact({"error": type(exc).__name__}),
                }
            else:
                raise AssertionError("Swapped training call unexpectedly succeeded")
            add(
                ident,
                w["id"],
                "repair",
                "",
                defs,
                [*hist[:1], qwen_tools.assistant_message([wrong]), error],
                action,
            )
    packages = train["packages"]
    unique = {w["package"]: w for w in packages}
    for i, w in enumerate(packages):
        keycol = ("Name", "package", "project", "Distribution name")[i % 4]
        h = [keycol, "Version", "Requires-Python"]
        raw = table(h, [(k, v["release"], v["requirement"]) for k, v in unique.items()])
        req = text(
            f'Use the attached metadata to check the Python requirement for "{w["package"]}".',
            w["claim"],
        )
        defs, call, _, hist = observed(
            tmp,
            "package-versions.csv",
            raw,
            keycol,
            w["package"],
            h[2],
            w["requirement"],
            req,
        )
        add(
            w["id"],
            w["package"],
            "package_open",
            w["label"],
            defs,
            hist[:1],
            qwen_tools.assistant_message([call]),
        )
        add(
            w["id"],
            w["package"],
            "package_answer",
            w["label"],
            defs,
            hist,
            answer(w["label"], w["reason"]),
        )
    examples.extend(
        r
        for r in read(DIR / "cross-view-training.json")["examples"]
        if r["kind"] == "rehearsal"
    )
    assert len({w["id"] for w in examples}) == len(examples)
    return examples


def build():
    split = take_tables()
    packages = read(DIR / "proof-curriculum-training.json")["package"]
    train = {"tables": split["train"], "roles": roles(), "packages": packages}
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    rows = []
    documents = {}

    def emit(
        ident, cluster, family, q, gold, document=None, defs=(), call=None, obs=None
    ):
        rows.append(
            {
                "id": ident,
                "cluster": cluster,
                "family": family,
                "questions": [q],
                "gold": gold,
                "document": document,
                "calls": [call] if call else [],
                "observation": obs,
                "initial_ids": tok.encode(
                    qwen_tools.render(
                        tok, [{"role": "user", "content": q}], list(defs)
                    ),
                    add_special_tokens=False,
                ),
            }
        )

    with TemporaryDirectory() as directory:
        tmp = Path(directory)
        for w in split["test_alpha3"]:
            f = w["fields"]
            raw = catalog(w, split["test_alpha3"])
            fname = w["id"] + "-reference.csv"
            documents[fname] = raw
            for q in w["claims"]:
                ident = fingerprint([w["id"], q["claim"]])[:16]
                inline = text("\n".join(f"{k}: {v}" for k, v in f.items()), q["claim"])
                emit(ident + "-inline", w["id"], "inline", inline, q["label"])
                req = text(
                    f'Consult the attached table for "{f["title"]}".', q["claim"]
                )
                defs, call, obs, _ = observed(
                    tmp, fname, raw, "title", f["title"], q["field"], f[q["field"]], req
                )
                emit(
                    ident + "-file",
                    w["id"],
                    "file",
                    req,
                    q["label"],
                    fname,
                    defs,
                    call,
                    obs,
                )
        inventory = read("research/sources/field-selection/inventory.json")
        prior = {
            canonicalize_name(w["package"])
            for w in packages + read(DIR / "reasoned-evidence-panel.json")["wheels"]
        }
        for w in inventory:
            assert (
                canonicalize_name(w["package"]) not in prior
                and digest(Path(w["path"])) == w["metadata_sha256"]
            )
            h = BytesParser().parsebytes(Path(w["path"]).read_bytes())
            assert [h[k] for k in ("Name", "Version", "Requires-Python")] == [
                w[k] for k in ("package", "version", "requirement")
            ]
        selected = choose(
            [
                w
                for w in inventory
                if re.fullmatch(r">=\s*3\.(?:7|8|9|10)", w["requirement"])
            ],
            SEED,
            lambda w: w["package"],
        )[:8]
        patches = {6: "3.6.15", 7: "3.7.17", 8: "3.8.20", 9: "3.9.20", 10: "3.10.15"}
        fname = "published-metadata.csv"
        raw = table(
            ["Project", "Release", "Requires-Python"],
            [(w["package"], w["version"], w["requirement"]) for w in inventory],
        )
        documents[fname] = raw
        for w in selected:
            k = int(w["requirement"].split(".")[1])
            for minor in (k - 1, k):
                ver = patches[minor]
                req = text(
                    f'Look up the Python requirement for "{w["package"]}" in the release inventory.',
                    f"Python {ver} satisfies the requirement of {w['package']} {w['version']}.",
                )
                defs, call, obs, _ = observed(
                    tmp,
                    fname,
                    raw,
                    "Project",
                    w["package"],
                    "Requires-Python",
                    w["requirement"],
                    req,
                )
                label = "true" if ver in SpecifierSet(w["requirement"]) else "false"
                assert (minor == k) == (label == "true")
                emit(
                    w["package"] + "-" + ver,
                    w["package"],
                    "wheel",
                    req,
                    label,
                    fname,
                    defs,
                    call,
                    obs,
                )
    old = read(DIR / "cross-view-panel.json")
    for w in old["rows"]:
        if w["family"] == "retention":
            rows.append(w)
            documents[w["document"]] = old["documents"][w["document"]]
    assert len(rows) == 96 and len({r["id"] for r in rows}) == 96
    for path, data in (
        (TRAIN, train),
        (
            PANEL,
            {
                "rows": choose(rows, SEED, lambda r: r["id"]),
                "documents": documents,
                "tables": split["test_alpha3"],
                "wheels": selected,
            },
        ),
    ):
        if path.exists():
            assert read(path) == data
        else:
            save(path, data)
    print("Frozen", len(train["tables"]), len(train["roles"]), len(rows), flush=True)


if __name__ == "__main__":
    build()
