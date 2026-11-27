"""Build identical file prompts around two genuine historical versions of an article."""

import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen import qwen_tools
from loopqwen.csv_tools import tools
from loopqwen.export_runtime import load_export_tokenizer
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.tool_session import compact
from scripts.revision_material import PANEL as PREVIOUS

NAME = "file-evidence-01"
REG = Path("research/experiments") / (NAME + ".json")
DATA = Path("research/data")
SRC = Path("data/research_external/vitaminc")


def pairs(split, s, exclude, limit):
    from shutil import copyfileobj
    from urllib.request import urlopen

    path = SRC / (split + ".jsonl")
    if not path.exists():
        tmp = path.with_suffix(".download")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with (
            urlopen(s["source_base"] + path.name, timeout=90) as inp,
            tmp.open("wb") as out,
        ):
            copyfileobj(inp, out)
        if digest(tmp) != s["sources"][split]:
            raise ValueError("Unexpected publisher payload")
        tmp.replace(path)
    assert digest(path) == s["sources"][split]
    groups = defaultdict(list)
    with path.open() as f:
        for r in map(json.loads, f):
            if r["revision_type"] != "real" or r["page"].casefold() in exclude:
                continue
            groups[(r["page"], r["wiki_revision_id"], r["case_id"], r["claim"])].append(
                (r["unique_id"], r["label"], r["evidence"])
            )
    pages = set()
    result = []
    texts = set()
    for key in sorted(groups, key=lambda k: fingerprint([s["fit_seed"], split, k])):
        page, revision, case, claim = key
        rows = groups[key]
        if (
            page.casefold() in pages
            or len(rows) != 2
            or {r[1] for r in rows} != {"SUPPORTS", "REFUTES"}
            or len({r[2] for r in rows}) != 2
            or any(
                "\ufffd" in v or not 3 <= len(v) <= 820 or v.casefold() in texts
                for v in [claim, *[r[2] for r in rows]]
            )
        ):
            continue
        # Skip text that could impersonate a chat delimiter in the native protocol.
        if any(
            t in (claim + page + " ".join(r[2] for r in rows))
            for t in ("<|", "<think>", "</think>")
        ):
            continue
        result.append(
            {
                "page": page,
                "revision": revision,
                "case": case,
                "claim": claim,
                "rows": rows,
            }
        )
        pages.add(page.casefold())
        texts.update(v.casefold() for v in [claim, *[r[2] for r in rows]])
        if len(result) == limit:
            break
    assert len(result) == limit
    return result


def forms(group, mode, training):
    key = group["page"]
    claim = group["claim"]
    if mode == "file":
        if training:
            versions = [
                f'Read the clipping for "{key}" from the supplied file. Does it support this claim? {claim}\nAnswer yes or no.',
                f'Check the article "{key}" in my CSV. Claim: {claim}\nUsing its excerpt, answer yes or no.',
            ]
            return versions[int(fingerprint(key), 16) % len(versions)]
        return f'My file contains a clipping from "{key}". According to that clipping, is the following statement supported?\n{claim}\nAnswer yes or no.'
    return (
        (
            "Excerpt:\n{evidence}\n\nClaim:\n"
            if training
            else "Read this passage:\n{evidence}\n\nDoes the passage support the statement below?\n"
        )
        + claim
        + "\nAnswer yes or no."
    )


def table(key, evidence, distractors):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["article", "excerpt"])
    entries = [(key, evidence), *distractors]
    writer.writerows(sorted(entries, key=lambda x: fingerprint(x[0])))
    return stream.getvalue()


def material():
    s = read(REG)
    previous = read(PREVIOUS)
    excluded = {r["page"].casefold() for r in previous}
    fresh = pairs("test", s, excluded, s["file_pairs"] + s["inline_pairs"])
    excluded.update(g["page"].casefold() for g in fresh)
    training = pairs("train", s, excluded, s["train_pairs"])
    clean = lambda x: " ".join(x.lower().split())

    def texts(groups):
        return {
            clean(t) for g in groups for t in [g["claim"], *(r[2] for r in g["rows"])]
        }

    assert texts(fresh).isdisjoint(texts(training))
    prior_text = {clean(r[k]) for r in previous for k in ("claim", "evidence")}
    assert (texts(fresh) | texts(training)).isdisjoint(prior_text)
    release = Path(s["checkpoint"])
    tok = load_export_tokenizer(
        release,
        eos_token_ids=[read(release / "config.json")["text_config"]["eos_token_id"]],
    )
    train = []
    rows = []
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "clippings.csv"
        for is_train, groups in ((True, training), (False, fresh)):
            for i, g in enumerate(groups):
                distractors = [
                    (d["page"], d["rows"][0][2])
                    for d in (
                        groups[(i + 1) % len(groups)],
                        groups[(i + 2) % len(groups)],
                    )
                ]
                modes = (
                    ("file", "inline")
                    if is_train
                    else (("file",) if i < s["file_pairs"] else ("inline",))
                )
                for mode in modes:
                    for j, (ident, label, evidence) in enumerate(g["rows"]):
                        gold = {"SUPPORTS": "yes", "REFUTES": "no"}[label]
                        body = forms(g, mode, is_train).replace("{evidence}", evidence)
                        uid = mode + "-" + ident
                        messages = [{"role": "user", "content": body}]
                        dest = (
                            table(g["page"], evidence, distractors)
                            if mode == "file"
                            else ""
                        )
                        calls = []
                        funcs = []
                        if dest:
                            path.write_bytes(dest.encode())
                            funcs = tools([path], keyed=True)
                            args = {
                                "document": path.name,
                                "key_column": "article",
                                "key": g["page"],
                                "value_column": "excerpt",
                            }
                            call = {"name": "csv_lookup", "arguments": args}
                            returned = funcs[0].callback(**args)
                            assert returned == {"value": evidence}
                            calls = [call]
                        decl = qwen_tools.declarations(funcs)
                        if is_train:
                            if calls:
                                action = qwen_tools.assistant_message(calls)
                                if j == 0:
                                    train.append(
                                        {
                                            "id": uid + "-lookup",
                                            "tools": decl,
                                            "messages": messages.copy(),
                                            "target": action,
                                        }
                                    )
                                messages.extend(
                                    [
                                        action,
                                        {"role": "tool", "content": compact(returned)},
                                    ]
                                )
                            train.append(
                                {
                                    "id": uid,
                                    "tools": decl,
                                    "messages": messages,
                                    "target": {"role": "assistant", "content": gold},
                                    "pair": g["case"] + mode,
                                    "pair_order": j,
                                    "thinking": "",
                                }
                            )
                        else:
                            rows.append(
                                {
                                    "id": uid,
                                    "family": mode,
                                    "cluster": g["case"],
                                    "page": g["page"],
                                    "revision": g["revision"],
                                    "questions": [body],
                                    "claim": g["claim"],
                                    "gold": gold,
                                    "evidence": evidence,
                                    "file": dest,
                                    "calls": calls,
                                    "initial_ids": tok.encode(
                                        qwen_tools.render(tok, messages, decl),
                                        add_special_tokens=False,
                                    ),
                                }
                            )
    # Rehearse authentic human lookup/return messages on original WikiSQL training tables.
    prev = read(DATA / "role-binding-training.json")["examples"]
    sql = [r for r in prev if r["id"].startswith("wiki-train-") and r["tools"]]
    for r in sorted(sql, key=lambda r: fingerprint([s["fit_seed"], r["id"]]))[
        : s["rehearsal"]
    ]:
        train.append({k: r[k] for k in ("id", "tools", "messages", "target")})
    rows.sort(key=lambda r: fingerprint([s["fit_seed"], "queue", r["id"]]))
    for g in fresh[: s["file_pairs"]]:
        twins = [r for r in rows if r["cluster"] == g["case"]]
        assert (
            len(twins) == 2
            and twins[0]["initial_ids"] == twins[1]["initial_ids"]
            and twins[0]["file"] != twins[1]["file"]
        )
    trainset = {
        "source_articles": [
            {k: g[k] for k in ("case", "page", "revision", "claim")} for g in training
        ],
        "examples": train,
    }
    for name, value in ((s["material"], {"rows": rows}), (s["training"], trainset)):
        if not Path(name).exists():
            save(name, value)
        assert read(name) == value
    return len(train), len(rows)


if __name__ == "__main__":
    print(material())
