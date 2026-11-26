"""Private key/value worlds for reproducible tool interaction and grounded decisions."""
import random
import re
from collections import Counter

from ..tool_session import Tool, compact

# Each tuple specifies a domain, source entities, linked entities and a measurable quantity.
DOMAINS = {
    "depot": ("parcel", "hub", "capacity", "manifest_hub", "hub_capacity"),
    "museum": ("painting", "room", "area", "locate_art", "room_area"),
    "campus": ("course", "hall", "seats", "course_building", "hall_seats"),
    "library": ("book", "shelf", "width", "find_shelf", "shelf_width"),
    "garden": ("flower", "bed", "length", "plant_location", "bed_length"),
    "fleet": ("bus", "garage", "capacity", "assigned_garage", "garage_capacity"),
    "festival": ("act", "stage", "capacity", "stage_assignment", "stage_capacity"),
    "printing": ("order", "press", "speed", "assigned_press", "press_speed"),
    "observatory": ("instrument", "station", "elevation", "instrument_station", "station_elevation"),
    "railway": ("service", "platform", "length", "service_platform", "platform_length"),
    "tournament": ("team", "arena", "capacity", "home_arena", "arena_capacity"),
    "warehouse": ("shipment", "dock", "clearance", "shipment_dock", "dock_clearance"),
    "archive": ("collection", "vault", "volume", "collection_vault", "vault_volume"),
    "market": ("stall", "district", "population", "locate_stall", "district_population"),
}
NAMES = "Juniper Cobalt Maple Amber Kestrel Rowan Willow Flint Cedar Quartz Hazel Pearl".split()
PLACES = "North East West South Central Harbor Valley Grove Ridge Hill Lake Coast".split()
KINDS = ("lookup", "chain", "compare", "direct", "clarify")


def make(seed, domains, per_kind, style, prefix):
    rng = random.Random(seed)
    rows = []
    for k in range(per_kind):
        for kind in KINDS:
            domain = domains[(k + KINDS.index(kind)) % len(domains)]
            source, place, metric, link, measure = DOMAINS[domain]
            names = rng.sample(NAMES, 3)
            places = rng.sample(PLACES, 3)
            numbers = rng.sample(range(11, 99), 3)
            arg = (source, place) if style != "alternate" else ("identifier", "location")
            ops = [link, measure]
            if rng.random() < 0.5:
                ops.reverse()
            a, b, _ = places
            s = names[0]
            if style == "train":
                questions = [
                    f"Give the {metric} of {place} {a}. Reply with just the number.",
                    f"What is the {metric} of the {place} assigned to {source} {s}? Reply with just the number.",
                    f"Which {place} has the greater {metric}, {a} or {b}? Reply with its name.",
                    f"The {metric} values are {a} = {numbers[0]} and {b} = {numbers[1]}. Which is larger? Reply with its name.",
                    f"What is the {metric} of my {place}?",
                ]
            else:
                questions = [
                    f"For the {place} called {a}, look up its {metric}. Return the number only.",
                    f"I have {source} {s}. Find its assigned {place}, then tell me that site's {metric}. Just the number.",
                    f"Compare {a} with {b} using the recorded {metric}. Name the {place} with the higher value.",
                    f"Use these figures: {a} has {numbers[0]}, {b} has {numbers[1]}. Give only the name with the higher figure.",
                    f"Tell me my {place}'s {metric}. I haven't specified which {place} yet.",
                ]
            idx = KINDS.index(kind)
            link_call = dict(name=link, arguments={arg[0]: s})
            calls = [dict(name=measure, arguments={arg[1]: p}) for p in places[:2]]
            expected = [[calls[0]], [link_call, calls[0]], calls, [], []][idx]
            winner = a if numbers[0] > numbers[1] else b
            gold = [str(numbers[0]), str(numbers[0]), winner, winner, f"Which {place}?"][idx]
            rows.append(dict(id=f"{prefix}-{k}-{kind}", kind=kind, domain=domain, args=list(arg),
                             tools=ops, names=names, places=places, numbers=numbers,
                             question=questions[idx], expected=expected, gold=gold))
    rng.shuffle(rows)
    return rows


def tools_for(row):
    source, place, metric, link, measure = DOMAINS[row["domain"]]
    catalog = {}
    for name, key, description, keys, values in [
        (link, row["args"][0], f"Return the assigned {place} for a named {source}.", row["names"], row["places"]),
        (measure, row["args"][1], f"Return the recorded {metric} for a named {place}.", row["places"], row["numbers"]),
    ]:
        table = dict(zip(keys, values, strict=True))
        def callback(_table=table, _key=key, **kwargs):
            return {"value": _table[kwargs[_key]]}
        schema = dict(type="object", properties={key: {"type": "string"}}, required=[key], additionalProperties=False)
        catalog[name] = Tool(name, description, schema, callback)
    return [catalog[name] for name in row["tools"]]


def score(row, trajectory):
    turns = trajectory["turns"]
    actual = [t["call"] for t in turns if "observation" in t]
    planned = row["expected"]
    path = (Counter(compact(c) for c in actual) == Counter(compact(c) for c in planned)
            if row["kind"] == "compare" else actual == planned)
    answer = (trajectory["answer"] or "").strip()
    if row["kind"] == "clarify":
        place = DOMAINS[row["domain"]][1]
        correct = bool(re.search(r"\b" + place + r"\b", answer, re.I) and "?" in answer)
    else:
        correct = answer.strip(" .!\n").casefold() == row["gold"].casefold()
    return dict(success=path and correct and trajectory["status"] == "answered", path=path,
                correct_answer=correct, completed=trajectory["status"] == "answered")
