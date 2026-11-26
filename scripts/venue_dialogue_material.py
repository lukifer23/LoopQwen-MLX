"""Reserve human Cambridge venue dialogues with changed constraints and resampled inventories."""

import hashlib
import json
import random
from pathlib import Path
from urllib.request import urlopen

from loopqwen.evaluation import venue_dialogues as venues
from loopqwen.evaluation import table_questions
from loopqwen.io import read, save
from loopqwen.tool_session import compact, initial_messages

REV = "fe0c8e65cfcd8462bd33c86e35f21addc84ca82b"
SRC = Path("data/research_external/multiwoz")
DATA = Path("research/data")
SERVICES = ["restaurant", "hotel", "attraction"]


def key(text):
    return hashlib.sha256(("cambridge-882704/" + text).encode()).hexdigest()


def source(name):
    f = SRC / name
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(
            urlopen(
                "https://raw.githubusercontent.com/budzianowski/multiwoz/"
                + REV
                + "/"
                + name,
                timeout=60,
            ).read()
        )
    return read(f)


def inventories():
    full = []
    selected = []
    for service in SERVICES:
        for row in source("db/" + service + "_db.json"):
            values = [
                service if c == "service" else str(row.get(c, "")).strip().casefold()
                for c in venues.COLUMNS
            ]
            full.append(values)
            if int(key(service + row["id"]), 16) % 2:
                selected.append(values)
    return dict(full=full, subset=selected)


def options(split, total, inv):
    options = {service: [] for service in SERVICES}
    for i in range(1, total + 1):
        for convo in source(f"data/MultiWOZ_2.2/{split}/dialogues_{i:03}.json"):
            if split == "train" and "attraction" in convo["services"]:
                continue
            history = []
            earlier = {}
            pool = []
            for t in convo["turns"]:
                history.append({k: t[k] for k in ["speaker", "utterance"]})
                if t["speaker"] != "USER":
                    continue
                occupied = [
                    f for f in t["frames"] if f.get("state", {}).get("slot_values")
                ]
                if len(occupied) != 1:
                    continue
                frame = occupied[0]
                service = frame["service"]
                state = frame["state"]
                if (
                    service not in SERVICES
                    or state["active_intent"] != "find_" + service
                ):
                    continue
                normalized = {}
                legal = True
                for raw, values in state["slot_values"].items():
                    col = raw.split("-", 1)[1]
                    if col.startswith("book"):
                        continue
                    if len(values) != 1 or col not in venues.COLUMNS:
                        legal = False
                        break
                    v = values[0].casefold()
                    if v not in ("dontcare", "not mentioned", ""):
                        normalized[col] = v
                previous = earlier.get(service, {})
                earlier[service] = normalized
                if (
                    not legal
                    or not previous
                    or len(history) > 11
                    or not (normalized.items() & previous.items())
                    or not (normalized.items() - previous.items())
                ):
                    continue
                live = {"service": service, **normalized}
                gold = [venues.count(rs, live) for rs in inv.values()]
                if gold[0] == gold[1] or gold[0] < 1:
                    continue
                identity = convo["dialogue_id"] + "/" + t["turn_id"]
                pool.append(
                    dict(
                        id=identity,
                        dialogue=convo["dialogue_id"],
                        service=service,
                        turns=list(history),
                        state=live,
                    )
                )
            if pool:
                item = min(pool, key=lambda r: key(r["id"]))
                options[item["service"]].append(item)
    return {s: sorted(rows, key=lambda r: key(r["id"])) for s, rows in options.items()}


def main():
    inv = inventories()
    train = options("train", 5, inv)
    test = options("test", 2, inv)
    print("eligible train/test", [(s, len(train[s]), len(test[s])) for s in SERVICES])
    n = min(192, len(train["hotel"]), len(train["restaurant"]))
    if n < 96:
        raise ValueError("Insufficient matched dialogues")
    examples = []
    used = [r for s in ("hotel", "restaurant") for r in train[s][:n]]
    for r in used:
        args = venues.arguments(r["state"])
        action = compact(dict(name="query_table", arguments=args))
        for world, rs in inv.items():
            tools = table_questions.tools_for(venues.table(rs))
            msgs = initial_messages(venues.prompt_for(r["turns"]), tools)
            if world == "full":
                examples.append(dict(messages=msgs, visible=action, thinking=""))
            result = tools[0].callback(**args)
            gold = venues.count(rs, r["state"])
            if result["values"] != [gold]:
                raise ValueError("Different SQLite denotation")
            msgs = [
                *msgs,
                dict(role="assistant", content=action),
                dict(role="tool", content=compact(result)),
            ]
            examples.append(
                dict(
                    messages=msgs,
                    visible=str(gold),
                    thinking="",
                    pair=r["id"],
                    pair_order=int(world == "subset"),
                )
            )
    rehearsal = read(DATA / "tool-dialogue-typed-training.json")["examples"]
    examples.extend(random.Random(58201).sample(rehearsal, 512))
    save(
        DATA / "tool-venue-dialogue-training.json",
        dict(
            examples=examples, source=REV, ids=[r["id"] for r in used], dialogues=2 * n
        ),
    )
    rows = []
    for service, size in [("hotel", 8), ("restaurant", 8), ("attraction", 12)]:
        if len(test[service]) < size:
            raise ValueError("Insufficient " + service)
        for r in test[service][:size]:
            for world, rs in inv.items():
                rows.append(
                    dict(
                        r,
                        id=r["id"] + "/" + world,
                        inventory=world,
                        gold=venues.count(rs, r["state"]),
                    )
                )
    rows.extend(
        dict(r, provider="bfcl_exec")
        for r in read(DATA / "tool-exec-bfcl-01.json")["rows"]
    )
    random.Random(18723).shuffle(rows)
    save(
        DATA / "tool-venue-dialogue-test.json",
        dict(source=REV, rows=rows, inventories=inv),
    )


if __name__ == "__main__":
    main()
