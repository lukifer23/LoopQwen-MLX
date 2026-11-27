"""Build source-balanced complete training turns, independent of model scores."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import monotonic

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from scripts.agenda_sources import events, source
from scripts.middle_depth_budget import BASE
from scripts.program_language_material import answer, user

DATA = Path("research/sources/middle-depth")
ROWS = DATA / "training.json.gz"
SEED = 4619
SOURCES = Path("research/sources")
CAP = 1280


def ordered(rows, key):
    return sorted(rows, key=lambda r: fingerprint([SEED, key(r)]))


def build():
    start = monotonic()
    tok = tokenizer({"checkpoint": str(BASE)})
    pools = defaultdict(list)
    identities = {}

    def load(name):
        path = SOURCES / name
        identities[str(path)] = digest(path)
        return unpack(path) if path.suffix == ".gz" else read(path)

    def add(
        family,
        group,
        source_id,
        name,
        prompt,
        final,
        reasoning="",
        foil=None,
        history=None,
    ):
        messages = [*(history or []), user(prompt)]
        p, y = token_pair(
            tok, dict(messages=messages, tools=[], target=answer(final, reasoning))
        )
        neg = []
        if foil:
            _, wrong = token_pair(
                tok, dict(messages=messages, tools=[], target=answer(foil, reasoning))
            )
            pos = next(i for i, (a, b) in enumerate(zip(y, wrong)) if a != b)
            neg = [[pos, wrong[pos]]]
        if len(p) + len(y) <= CAP:
            pools[family, group].append(
                dict(id=name, cluster=source_id, p=p, y=y, negatives=neg)
            )

    # Each published polarity pair contributes both answers.
    pairs = ordered(load("qualitative-pairs/training.json.gz"), lambda r: r["id"])
    selected = set()
    for pair in pairs:
        group = pair["source_background"]
        if group in selected:
            continue
        selected.add(group)
        for row in pair["members"]:
            other = next(v for v in row["choices"] if v != row["correct"])
            add(
                "reasoning",
                "polarity",
                group,
                row["id"],
                row["background"]
                + "\n\n"
                + row["question"]
                + " ("
                + " or ".join(row["choices"])
                + ")",
                row["correct"] + ".",
                foil=other + ".",
            )
        if len(selected) == 24:
            break

    bank = load("receipt-repair/targets.json.gz")
    for r in ordered(bank, lambda r: r["id"]):
        if r["split"] != "train" or len(r["p"]) + len(r["y"]) > CAP:
            continue
        if r["group"] in (
            "open",
            "deliver",
            "repair",
            "recovered",
            "reuse",
            "redirect",
        ):
            pools["tools", r["group"]].append(r)
        elif r["group"] == "reasoning":
            pools["reasoning", "comparison"].append(r)
    existing = {r["id"] for r in pools["reasoning", "comparison"]}
    for r in ordered(load("source-copy/training.json.gz"), lambda r: r["id"]):
        if (
            r["group"] == "reasoning"
            and r["id"] not in existing
            and len(r["p"]) + len(r["y"]) <= CAP
        ):
            pools["reasoning", "comparison"].append(dict(**r, negatives=[]))
    table = load("grounded-table-sentences/originals.json")["train"]
    for r in ordered(table, lambda r: r["id"]):
        table_text = "\n".join(" | ".join(row) for row in r["rows"])
        add(
            "documents",
            "complete_table",
            r["source_table"],
            "table/" + str(r["id"]),
            r["page"]
            + " — "
            + r["section"]
            + "\n"
            + table_text
            + "\n\n"
            + r["question"],
            r["reference"],
        )
        if len(pools["documents", "complete_table"]) == 64:
            break
    originals = load("human-assistance/originals.json")["splits"]["train"]["selected"]
    for r in ordered(originals, lambda r: r["tree"]):
        add(
            "chat",
            "conversation",
            r["tree"],
            "oasst/" + r["response_id"],
            r["question"],
            r["answer"],
        )
        if len(pools["chat", "conversation"]) == 48:
            break
    # This corpus contains attributed human summarization and rewriting.
    dolly = Path("data/research_external/dolly/original.jsonl")
    identities[str(dolly)] = digest(dolly)
    from scripts.program_language_material import DOLLY_SHA

    assert identities[str(dolly)] == DOLLY_SHA
    rows = [json.loads(line) for line in dolly.read_text().splitlines()]
    for r in ordered(
        (r for r in rows if r["category"] == "summarization"),
        lambda r: r["instruction"],
    ):
        name = fingerprint(r)
        add(
            "chat",
            "rewrite",
            name,
            "dolly/" + name,
            r["instruction"] + "\n\n" + r["context"],
            r["response"],
        )
        if len(pools["chat", "rewrite"]) == 48:
            break
    # The short schedule curriculum keeps two original, XML-verified records.
    # A documented time change alters the required comparison in the edited edition.
    year = 2023
    for fmt in ("ical", "xml"):
        path = source(year, fmt, False)
        identities[str(path)] = digest(path)
    rows = ordered(events(year), lambda r: r["id"])

    def minutes(r):
        return int(
            (
                datetime.fromisoformat(r["end"]) - datetime.fromisoformat(r["start"])
            ).total_seconds()
            / 60
        )

    used = set()
    for a in rows:
        if a["id"] in used:
            continue
        b = next(
            (
                b
                for b in rows
                if b["id"] != a["id"]
                and b["id"] not in used
                and 0 < minutes(a) - minutes(b) <= 35
                and a["title"] != b["title"]
            ),
            None,
        )
        if not b:
            continue
        used.update([a["id"], b["id"]])
        from datetime import timedelta

        group = a["id"] + "/" + b["id"]
        for kind, extra in (
            ("published", 0),
            ("revision", minutes(a) - minutes(b) + 5),
        ):
            end = datetime.fromisoformat(b["end"]) + timedelta(minutes=extra)
            p = (
                f"{a['title']} | {a['start']} to {a['end']}\n"
                f"{b['title']} | {b['start']} to {end.isoformat()}\n\n"
                "Which session has more scheduled minutes, and by how much? Show the two durations."
            )
            da, db = minutes(a), minutes(b) + extra
            win, lose = (a, b) if da > db else (b, a)
            response = f"{win['title']} is longer by {abs(da - db)} minutes. The first session lasts {da} minutes and the second lasts {db} minutes."
            add(
                "documents",
                "agenda",
                group,
                group + "/" + kind,
                p,
                response,
                foil=lose["title"] + " is longer.",
            )
        if len(used) >= 32:
            break
    original_edits = DATA / "conversation-edits.json"
    identities[str(original_edits)] = digest(original_edits)
    for i, (group, prompt, final, wrong) in enumerate(read(original_edits)):
        add(
            "chat",
            "edited_reply",
            "authored/" + group,
            "authored/" + str(i),
            prompt,
            final,
            foil=wrong,
        )
    # Equal turns per family. Numeric action/delivery/recovery and nonnumeric
    # database histories all receive a reserved share of the tool family's mass.
    quotas = {
        "reasoning": {"polarity": 48, "comparison": 24},
        "chat": {"conversation": 32, "rewrite": 32, "edited_reply": 8},
        "tools": {
            g: 12
            for g in ("open", "deliver", "repair", "recovered", "reuse", "redirect")
        },
        "documents": {"complete_table": 40, "agenda": 32},
    }
    decks = {}
    for f, groups in quotas.items():
        deck = []
        for group, quota in groups.items():
            values = pools[f, group]
            if len(values) < quota:
                raise ValueError((f, group, quota, len(values)))
            for r in values[:quota]:
                deck.append(
                    {
                        **{k: r[k] for k in ("id", "p", "y", "cluster", "negatives")},
                        "family": f,
                        "group": group,
                    }
                )
        decks[f] = ordered(deck, lambda r: r["id"])
    training = [decks[f][i] for i in range(72) for f in sorted(decks)]
    assert len({r["id"] for r in training}) == len(training)
    packed(ROWS, training)
    save(
        DATA / "inventory.json",
        dict(
            source_hashes=identities,
            builder=digest(Path(__file__)),
            seed=SEED,
            training=digest(ROWS),
            quotas=quotas,
            rows=len(training),
            input_positions=sum(len(r["p"]) for r in training),
            target_positions=sum(len(r["y"]) for r in training),
            seconds=monotonic() - start,
            teacher_draws=0,
            exposure="Two original agenda entries per training prompt; edited end times are counterfactual. FeTaQA includes the whole table. Full-length prefill and suffix adaptation use a separate native stage.",
        ),
    )
    print(len(training), digest(ROWS))


if __name__ == "__main__":
    build()
