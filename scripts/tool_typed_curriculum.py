"""Generate typed API supervision from local, executable, non-benchmark utilities."""

from datetime import date, timedelta
import math
import random
from loopqwen.io import read, save
from loopqwen.tool_session import Tool, compact, initial_messages


def recipe(portions, grams):
    return round(portions * grams, 2)


def packs(requested, per_box):
    return math.ceil(requested / per_box)


def ticket(attendees, unit_price):
    return round(attendees * unit_price, 2)


def advance(day, days):
    return str(date.fromisoformat(day) + timedelta(days=days))


def add_levels(readings, offset):
    return [round(x + offset, 2) for x in readings]


def shortlist(labels, limit):
    return labels[:limit]


def subtotal(prices, discount):
    return round(sum(prices) * (1 - discount / 100), 2)


def repeat_label(label, count):
    return "-".join([label] * count)


def rows():
    rng = random.Random(380140)
    arrays = {"type": "array", "items": {"type": "number"}}
    strings = {"type": "array", "items": {"type": "string"}}
    funcs = [
        (
            "ingredients_for_portions",
            "Compute total ingredient weight for the requested number of portions.",
            recipe,
            [("portions", "integer"), ("grams", "number")],
        ),
        (
            "boxes_needed",
            "Compute how many whole boxes are required to contain all items.",
            packs,
            [("requested", "integer"), ("per_box", "integer")],
        ),
        (
            "admission_total",
            "Return total admission price for the group.",
            ticket,
            [("attendees", "integer"), ("unit_price", "number")],
        ),
        (
            "offset_date",
            "Advance an ISO calendar date by a signed number of days.",
            advance,
            [("day", "string"), ("days", "integer")],
        ),
        (
            "calibrate_readings",
            "Add an offset to every sensor reading.",
            add_levels,
            [("readings", arrays), ("offset", "number")],
        ),
        (
            "take_labels",
            "Return the first entries in an ordered list of labels.",
            shortlist,
            [("labels", strings), ("limit", "integer")],
        ),
        (
            "basket_total",
            "Compute the total after applying a percentage discount.",
            subtotal,
            [("prices", arrays), ("discount", "number")],
        ),
        (
            "make_tag",
            "Build a tag by repeating a label separated by hyphens.",
            repeat_label,
            [("label", "string"), ("count", "integer")],
        ),
    ]
    tools = []
    for name, about, func, args in funcs:
        schema = dict(
            type="object",
            properties={k: {"type": t} if isinstance(t, str) else t for k, t in args},
            required=[k for k, t in args],
            additionalProperties=False,
        )
        tools.append(Tool(name, about, schema, func))
    corpus = []
    for i in range(384):
        a, b = rng.randint(2, 45), rng.randint(2, 19)
        x = round(rng.uniform(0.3, 49.9), 2)
        vals = [round(rng.uniform(1, 35), 1) for _ in range(rng.randint(2, 6))]
        words = rng.sample(
            ["indigo", "coral", "sand", "lime", "gray", "gold", "pink", "blue"],
            rng.randint(3, 6),
        )
        day = str(date(2023, 1, 1) + timedelta(days=rng.randint(1, 900)))
        offset = rng.randint(-15, 25)
        cases = [
            (
                f"I'm cooking for {a} people, and each portion takes {x} grams. How many grams should I prepare?",
                [a, x],
            ),
            (f"How many boxes should we use for {a} items if each holds {b}?", [a, b]),
            (
                f"Tickets are {x} apiece. What's the admission cost for {a} visitors?",
                [a, x],
            ),
            (
                f"Starting at {day}, move {offset} days. What date does that give?",
                [day, offset],
            ),
            (f"Apply an offset of {offset} to the readings {vals}.", [vals, offset]),
            (
                f"Take the first {b % 3 + 1} labels from this list: {', '.join(words)}.",
                [words, b % 3 + 1],
            ),
            (
                f"My basket contains prices {vals}. What is the checkout total with a {b}% discount?",
                [vals, b],
            ),
            (
                f"I need a hyphenated tag with {words[0]} repeated {b % 4 + 2} times.",
                [words[0], b % 4 + 2],
            ),
        ]
        index = i % len(tools)
        tool = tools[index]
        question, values = cases[index]
        kwargs = dict(zip(tool.parameters["required"], values, strict=True))
        distractors = rng.sample([t for t in tools if t != tool], rng.randrange(3))
        active = [tool, *distractors]
        rng.shuffle(active)
        messages = initial_messages(question, active)
        call = compact(dict(name=tool.name, arguments=kwargs))
        obs = tool.callback(**kwargs)
        visible = compact(obs) if isinstance(obs, list) else str(obs)
        corpus.append(
            dict(id=f"typed-{i}-call", messages=messages, thinking="", visible=call)
        )
        messages = messages + [
            dict(role="assistant", content=call),
            dict(role="tool", content=compact({"value": obs})),
        ]
        corpus.append(
            dict(
                id=f"typed-{i}-answer", messages=messages, thinking="", visible=visible
            )
        )
    return corpus


if __name__ == "__main__":
    parent = read("research/data/tool-dialogue-grounded-training.json")
    public = [r for r in parent["examples"] if not r["id"].startswith("grounded-")]
    controlled = [r for r in parent["examples"] if r["id"].startswith("grounded-")]
    rng = random.Random(5104)
    dataset = rows() + public + rng.sample(controlled, 128)
    save(
        "research/data/tool-dialogue-typed-training.json",
        dict(public_source=parent["public_source"], examples=dataset),
    )
    print(len(dataset), "response-only training turns")
