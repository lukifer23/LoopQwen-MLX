"""Teach observable value reversals and reserve joint public and catalog retention."""

import hashlib
import random
from pathlib import Path

from loopqwen.io import read, save
from loopqwen.research.tool_worlds import DOMAINS, make
from loopqwen.tool_session import Tool, compact, initial_messages

DATA = Path("research/data")
WORDS = "Aspen Birch Elm Hawthorn Iris Lotus Moss Olive Peony Sage Thistle Yew Dolphin Fox Frog Gecko Heron Lion Lynx Otter Owl Raven Swan Wren".split()
SETUPS = [
    ("test_run", "elapsed time", "seconds", "read_duration"),
    ("job", "cost", "credits", "lookup_charge"),
    ("material", "weight", "grams", "weigh_material"),
    ("station", "temperature", "degrees", "read_temperature"),
    ("shop", "stock", "units", "inspect_stock"),
    ("tank", "volume", "litres", "read_volume"),
    ("run", "distance", "metres", "find_distance"),
    ("booking", "duration", "minutes", "booking_duration"),
]


def examples():
    rng = random.Random(432281)
    out = []
    for i in range(320):
        entity, quantity, unit, name = rng.choice(SETUPS)
        candidates = rng.sample(WORDS, 2)
        vals = rng.sample(range(5, 160), 2)
        if i % 4 == 0:
            vals = [round((x - 50) / 10, 1) for x in vals]
        high = rng.choice([True, False])
        direction = "larger" if high else "smaller"
        key = "measurement-pair-" + str(i)
        group_strings = []
        for j, (x, y) in enumerate((vals, vals[::-1])):
            a, b = candidates
            known = {a: x, b: y}

            def look(identifier, known=known):
                return {"value": known[identifier]}

            param = {
                "type": "object",
                "properties": {"identifier": {"type": "string"}},
                "required": ["identifier"],
                "additionalProperties": False,
            }
            api = Tool(
                name, "Read the " + quantity + " of one " + entity + ".", param, look
            )
            if i % 2:
                forms = [
                    f"Between {entity} {a} and {entity} {b}, which has the {direction} {quantity}? Check both readings. Answer with its identifier.",
                    f"Find the {direction} of the unknown readings for {a} and {b}. Return the corresponding identifier.",
                ]
            else:
                forms = [
                    f"The recorded {quantity} is {x} {unit} for {a} and {y} {unit} for {b}. Which identifier has the {direction} reading? Give only its identifier.",
                    f"{entity.title()} measurements ({unit}): {a} = {x}, {b} = {y}. I need the identifier associated with the {direction} measurement.",
                    f"I checked both myself. {a} measured {x} {unit}. {b} measured {y} {unit}. Who had the {direction} {quantity}? Name only.",
                    f"Of {a} ({x} {unit}) and {b} ({y} {unit}), name the one with the {direction} value.",
                ]
            msgs = initial_messages(forms[(i // 2) % len(forms)], [api])
            if i % 2:
                for n, word in enumerate(candidates):
                    target = compact(dict(name=name, arguments={"identifier": word}))
                    if j == 0:
                        out.append(
                            dict(
                                id=key + "-request-" + str(n),
                                messages=list(msgs),
                                visible=target,
                                thinking="",
                            )
                        )
                    observed = api.callback(identifier=word)
                    msgs += [
                        dict(role="assistant", content=target),
                        dict(role="tool", content=compact(observed)),
                    ]
            answer = (max if high else min)(known, key=known.get)
            out.append(
                dict(
                    id=key + str(j),
                    pair=key,
                    pair_order=j,
                    messages=msgs,
                    visible=answer,
                    thinking="",
                )
            )
            group_strings.append(answer)
        assert len(set(group_strings)) == 2
    return out


def main():
    old = read(DATA / "tool-dialogue-typed-training.json")["examples"]
    fuller = read(DATA / "tool-dialogue-grounded-training.json")["examples"]
    rows = examples()
    seen = {
        hashlib.sha256(
            compact([r["messages"], r["visible"], r["thinking"]]).encode()
        ).digest()
        for r in rows
    }
    for r in [*old, *fuller]:
        key = hashlib.sha256(
            compact([r["messages"], r["visible"], r["thinking"]]).encode()
        ).digest()
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    save(DATA / "tool-counterfactual-training.json", dict(examples=rows))
    items = [
        dict(r, provider="world") for r in read(DATA / "tool-dialogue-unified-01.json")
    ]
    fresh = make(
        611278,
        ["aviary", "textile", "grocery", "gallery"],
        6,
        "alternate",
        "retention-fresh",
    )
    for w in fresh:
        source, place, metric, link, measure = DOMAINS[w["domain"]]
        a, b = w["places"][:2]
        x, y = w["numbers"][:2]
        item = w["names"][0]
        forms = {
            "chain": f"Which {metric} corresponds to {source} {item}? First find its {place}. Reply with the number.",
            "lookup": f"Please read the {metric} recorded against {place} {a}. Return just that figure.",
            "compare": f"Find out whether {place} {a} or {b} has greater {metric}. Reply with the winning name.",
            "direct": f"My notes give {a} a score of {x} and {b} a score of {y}. Who is ahead? Name only.",
            "clarify": f"I need the {metric} of a certain {place}, but I have not identified which. Ask me for what you need.",
        }
        w["question"] = forms[w["kind"]]
        items.append(dict(w, provider="world", panel="fresh", generation_cap=160))
    for r in read(DATA / "tool-exec-bfcl-01.json")["rows"]:
        items.append(dict(r, provider="bfcl_exec", panel="public", generation_cap=384))
    assert len(items) == 132 and len({r["id"] for r in items}) == 132
    save(DATA / "tool-counterfactual-retention.json", dict(rows=items))
    print(len(rows), "training turns;", len(items), "qualification requests")


if __name__ == "__main__":
    main()
