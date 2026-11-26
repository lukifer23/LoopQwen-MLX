"""Freeze source dialogue states and natural responses for native shared-layer supervision."""

import random
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from mlx_lm.utils import load_tokenizer

from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import SYSTEM, compact, prompt

REV = "e852981ae34990f4358979625854259302feaa78"
BASE = (
    "https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/"
    + REV
    + "/"
)
DEST = Path("research/data/dialogue-state-training.json")
ROOT = Path("data/research_external/sgd")
SEED = 173091


def get(rel, files):
    path = ROOT / rel
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urlopen(BASE + rel, timeout=60).read())
    files[rel] = digest(path)
    return read(path)


def response_rows(tok, split, n, files):
    schemas = {s["service_name"]: s for s in get(split + "/schema.json", files)}
    pool = []
    for i in range(1, n + 1):
        for d in get(f"{split}/dialogues_{i:03}.json", files):
            past = []
            # Use only the initial information-seeking task. The dialogue can later
            # switch services or move on to a reservation that is outside this fit.
            for j, (u, r) in enumerate(
                zip(d["turns"][::2], d["turns"][1::2], strict=True)
            ):
                if len(u["frames"]) != 1 or len(r["frames"]) != 1:
                    break
                uf, rf = u["frames"][0], r["frames"][0]
                s = schemas[uf["service"]]
                if s["service_name"] != rf["service"] or s["service_name"].split("_")[
                    0
                ] not in {
                    "Buses",
                    "Events",
                    "Flights",
                    "Homes",
                    "Hotels",
                    "Media",
                    "Movies",
                    "RentalCars",
                    "Restaurants",
                    "Travel",
                    "Weather",
                }:
                    break
                name = uf["state"]["active_intent"]
                intent = next(
                    (
                        v
                        for v in s["intents"]
                        if v["name"] == name
                        and not v["is_transactional"]
                        and v["required_slots"]
                    ),
                    None,
                )
                if intent is None:
                    break
                required = intent["required_slots"]
                supported = set(required) | set(intent["optional_slots"])
                props = {
                    z["name"]: {"type": "string", "description": z["description"]}
                    for z in s["slots"]
                    if z["name"] in supported
                }
                schema = {
                    "name": name,
                    "description": intent["description"],
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                        "additionalProperties": False,
                    },
                }
                past.append({"role": "user", "content": u["utterance"]})
                msgs = [
                    {"role": "system", "content": SYSTEM + compact([schema])},
                    *past,
                ]
                values = uf["state"]["slot_values"]
                known = {k: bool(values.get(k)) for k in required}
                asked = {v["slot"] for v in rf["actions"] if v["act"] == "REQUEST"}
                call = rf.get("service_call")
                ready = all(known.values())
                emit = None
                if (
                    not ready
                    and (asked & {k for k in required if not known[k]})
                    and not call
                ):
                    emit = r["utterance"]
                elif (
                    ready
                    and call
                    and call["method"] == name
                    and set(call["parameters"]) <= supported
                    and set(required) <= set(call["parameters"])
                ):
                    emit = compact({"name": name, "arguments": call["parameters"]})
                key = f"{split}-{d['dialogue_id']}-{j}"
                if (
                    emit
                    and len(tok.encode(prompt(tok, msgs), add_special_tokens=False))
                    <= 640
                    and len(tok.encode(emit)) <= 96
                ):
                    pool.append(
                        {
                            "id": key,
                            "dialogue": d["dialogue_id"],
                            "service": s["service_name"],
                            "kind": "call" if ready else "ask",
                            "messages": msgs,
                            "visible": emit,
                            "thinking": "",
                            "known": known,
                        }
                    )
                if call:
                    break
                past.append({"role": "assistant", "content": r["utterance"]})
    return pool


def select(rows, rng, count):
    rng.shuffle(rows)
    selected = []
    seen = set()
    by_kind = Counter()
    by_service = Counter()
    for w in rows:
        k = w["kind"]
        if (
            w["dialogue"] in seen
            or by_kind[k] >= count
            or by_service[w["service"], k] >= max(8, count // 4)
        ):
            continue
        seen.add(w["dialogue"])
        selected.append(w)
        by_kind[k] += 1
        by_service[w["service"], k] += 1
    assert by_kind == {"ask": count, "call": count}, by_kind
    return selected


def main():
    tok = load_tokenizer(local_snapshot())
    rng = random.Random(SEED)
    files = {}
    train = select(response_rows(tok, "train", 24, files), rng, 384)
    validation = select(response_rows(tok, "dev", 4, files), rng, 24)
    old = read("research/data/tool-dialogue-typed-training.json")["examples"]
    samples = rng.sample(old, 320)
    examples = [*train, *samples]
    save(
        DEST,
        {
            "upstream": BASE,
            "files": files,
            "license": "CC BY-SA 4.0",
            "examples": examples,
            "validation": validation,
        },
    )


if __name__ == "__main__":
    main()
