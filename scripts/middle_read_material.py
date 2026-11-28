"""Prepare checked opening programs, live deliveries and unrestricted preservation rows."""

import csv
import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from loopqwen.arithmetic_actions import as_steps, mentions
from loopqwen.calculator_tool import tool
from loopqwen.csv_tools import tools as csv_tools
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.qwen_tools import declarations
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from scripts.agenda_sources import events, source
from scripts.middle_depth_budget import BASE
from scripts.middle_depth_material import ROWS
from scripts.program_language_material import answer, call, observed, user

DATA = Path("research/sources/middle-read")
MATH = Path("research/data/operand-planner-training.json")
EDITS = DATA / "evidence-edits.json"
BANK = DATA / "training.json.gz"
SPEC = Path("research/diagnostics/middle-depth/middle-read-fit.json")


def material():
    start = monotonic()
    tok = tokenizer({"checkpoint": str(BASE)})
    rows, provenance = [], []
    provider = tool()
    math = {r["id"]: r for r in read(MATH)["rows"]}

    def add(key, cluster, kind, messages, providers, target, foil=None):
        row = {"messages": messages, "tools": declarations(providers), "target": target}
        p, y = token_pair(tok, row)
        negatives = []
        if foil:
            _, crossed = token_pair(tok, row | {"target": foil})
            pos = next(i for i, (a, b) in enumerate(zip(y, crossed)) if a != b)
            negatives = [[pos, crossed[pos]]]
        rows.append(
            {
                "id": key,
                "cluster": cluster,
                "group": kind,
                "p": p,
                "y": y,
                "negatives": negatives,
            }
        )

    for edit in read(EDITS):
        item = math[edit["id"]]
        source_text = item["question"]
        before, after = edit["replace"]
        if source_text.count(before) != 1:
            raise ValueError("Ambiguous edition change")
        pair = []
        for i, q in enumerate((source_text, source_text.replace(before, after))):
            operands = mentions(q)
            steps = as_steps([v for a, b, v in operands], item["actions"])
            observation = provider.callback(steps=steps)
            value = Fraction(observation["value"])
            if value != edit["checked"][i] or (
                i == 0 and value != Fraction(item["gold"])
            ):
                raise ValueError(
                    f"Source formula and checked answer disagree on {item['id']}"
                )
            action = call(provider, steps=steps)
            action["reasoning_content"] = edit["reason"]
            pair.append(
                (
                    [user(q + " Please use the calculator to check your result.")],
                    action,
                    observation,
                    answer(edit["answer_template"].format(value)),
                )
            )
            provenance.append(
                {
                    "source": item["id"],
                    "corpus": item["source"],
                    "edition": i,
                    "input": q,
                    "roles": edit["reason"],
                    "mentions": operands,
                    "steps": steps,
                    "returned": observation,
                }
            )
        if pair[0][2]["value"] == pair[1][2]["value"]:
            raise ValueError("The relevant change must alter the result")
        for i, (msgs, action, observation, reply) in enumerate(pair):
            key = f"{item['id']}/{i}/"
            add(
                key + "call",
                item["id"],
                "initial_calculation",
                msgs,
                [provider],
                action,
                pair[1 - i][1],
            )
            add(
                key + "reply",
                item["id"],
                "verified_delivery",
                [*msgs, action, observed(observation)],
                [provider],
                reply,
                pair[1 - i][3],
            )
    agenda = sorted(events(2023), key=lambda r: fingerprint([7793, r["id"]]))
    chosen = []
    for e in agenda:
        if len(e["title"]) <= 85 and all(e["room"] != v["room"] for v in chosen):
            chosen.append(e)
            if len(chosen) == 6:
                break
    if len(chosen) != 6:
        raise ValueError("Insufficient distinct source rooms")
    with TemporaryDirectory() as tmp:
        csv_file = Path(tmp) / "conference.csv"
        with csv_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("session", "room"))
            writer.writerows((e["title"], e["room"]) for e in chosen)
        (lookup,) = csv_tools([csv_file], keyed=True)
        for e in chosen:
            msgs = [
                user(
                    f"Please check conference.csv and tell me which room is hosting {e['title']}."
                )
            ]
            kwargs = {
                "document": csv_file.name,
                "key_column": "session",
                "key": e["title"],
                "value_column": "room",
            }
            observation = lookup.callback(**kwargs)
            if observation != {"value": e["room"]}:
                raise ValueError("Schedule lookup differs from both organizer sources")
            action = call(lookup, **kwargs)
            add(e["id"] + "/open", e["id"], "initial_lookup", msgs, [lookup], action)
            add(
                e["id"] + "/deliver",
                e["id"],
                "verified_lookup",
                [*msgs, action, observed(observation)],
                [lookup],
                answer(f"It is in {e['room']}."),
            )
            provenance.append(
                {
                    "source": e,
                    "csv_sha256": digest(csv_file),
                    "actual": observation,
                    "arguments": kwargs,
                }
            )
    prior = unpack(ROWS)
    for family, n in (("chat", 12), ("reasoning", 6), ("documents", 6)):
        pool = sorted(
            (
                r
                for r in prior
                if r["family"] == family and len(r["p"]) + len(r["y"]) <= 500
            ),
            key=lambda r: fingerprint([7793, r["id"]]),
        )
        if len(pool) < n:
            raise ValueError("Insufficient unrestricted examples")
        for row in pool[:n]:
            rows.append(row | {"group": "preserve_" + family})
    if len(rows) != 84 or len({r["id"] for r in rows}) != 84:
        raise ValueError("Training enrollment changed")
    rows.sort(key=lambda r: fingerprint([7793, r["id"]]))
    if BANK.exists() and unpack(BANK) != rows:
        raise ValueError("Stored source examples have changed")
    if not BANK.exists():
        packed(BANK, rows)
    origins = DATA / "source-checks.json"
    if origins.exists() and read(origins) != json.loads(json.dumps(provenance)):
        raise ValueError("Stored source observations have changed")
    if not origins.exists():
        save(origins, provenance)
    spec = {
        "seed": 7793,
        "order": [r["id"] for r in rows],
        "data": digest(BANK),
        "sources": {
            str(p): digest(p)
            for p in (
                EDITS,
                MATH,
                ROWS,
                source(2023, "xml", False),
                source(2023, "ical", False),
                origins,
            )
        },
        "draws_per_epoch": len(rows),
        "epochs_max": 2,
        "early_step": len(rows),
        "batch": 1,
        "depths": [1, 2],
        "lr": 0.001,
        "warmup": 8,
        "max_positions": max(len(r["p"]) + len(r["y"]) - 1 for r in rows),
        "max_labels": max(len(r["y"]) for r in rows),
    }
    if SPEC.exists() and read(SPEC) != spec:
        raise ValueError("Frozen training material changed")
    first = not SPEC.exists()
    save(SPEC, spec)
    if first:
        save(
            DATA / "preparation.json",
            {
                "fit": digest(SPEC),
                "seconds": monotonic() - start,
                "accepted_math_sources": 12,
                "reviewed_editions": 24,
                "actual_calculator_observations": 24,
                "actual_csv_lookups": 6,
                "teacher_draws": 0,
                "gpu_seconds": 0,
            },
        )
    print(
        len(rows),
        "verified turns, maximum positions",
        spec["max_positions"],
        "seconds",
        monotonic() - start,
    )


if __name__ == "__main__":
    material()
