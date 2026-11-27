"""Preserve complete source calls, recorded observations, and natural follow-through across services."""

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from loopqwen import qwen_tools
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.tool_session import compact
from scripts.dialogue_state_material import BASE, ROOT
from scripts.dialogue_state_material import DEST as PRIOR
from scripts.program_language_material import norm

OUT = Path("research/sources/grounded-service-turns")
DOMAINS = {
    "Buses",
    "Calendar",
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
}


TREE = "e51ac0bf13ec1b3f76533ae2816d47f4012dd911"


def git_hash(kind, content):
    return hashlib.sha1(
        kind + b" " + str(len(content)).encode() + b"\0" + content
    ).hexdigest()


def training_files():
    index = ROOT / "train-tree.json"
    if not index.is_file():
        url = (
            "https://api.github.com/repos/google-research-datasets/dstc8-schema-guided-dialogue/git/trees/"
            + TREE
        )
        index.parent.mkdir(exist_ok=True, parents=True)
        with urlopen(url, timeout=60) as src:
            save(index, json.load(src))
    entries = sorted(read(index)["tree"], key=lambda r: r["path"])
    body = b"".join(
        r["mode"].encode() + b" " + r["path"].encode() + b"\0" + bytes.fromhex(r["sha"])
        for r in entries
    )
    assert git_hash(b"tree", body) == TREE

    def restore(entry):
        assert entry["type"] == "blob" and re.fullmatch(
            r"dialogues_[0-9]{3}\.json|schema\.json", entry["path"]
        )
        rel = "train/" + entry["path"]
        path = ROOT / rel
        if not path.exists():
            with urlopen(BASE + rel, timeout=90) as src:
                data = src.read()
            assert git_hash(b"blob", data) == entry["sha"]
            path.parent.mkdir(exist_ok=True, parents=True)
            tmp = path.with_suffix(".partial")
            tmp.write_bytes(data)
            tmp.replace(path)
        data = path.read_bytes()
        assert len(data) == entry["size"] and git_hash(b"blob", data) == entry["sha"]
        return rel, hashlib.sha256(data).hexdigest()

    with ThreadPoolExecutor(max_workers=4) as worker:
        return dict(worker.map(restore, entries))


def declarations(service):
    out = []
    for intent in service["intents"]:
        if intent["is_transactional"]:
            continue
        allowed = set(intent["required_slots"]) | set(intent["optional_slots"])
        props = {
            r["name"]: {"type": "string", "description": r["description"]}
            for r in service["slots"]
            if r["name"] in allowed
        }
        out.append(
            {
                "type": "function",
                "function": {
                    "name": intent["name"],
                    "description": intent["description"],
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": intent["required_slots"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    return out


def candidate(service, frame, utterance, history):
    call = frame["service_call"]
    defs = declarations(service)
    decl = next(
        (r["function"] for r in defs if r["function"]["name"] == call["method"]), None
    )
    if decl is None:
        return None
    params = call["parameters"]
    props = decl["parameters"]
    if not (
        set(props["required"]) <= set(params) <= set(props["properties"])
        and all(isinstance(v, str) for v in params.values())
    ):
        return None
    obs = frame.get("service_results", [])

    def contains(text, value):
        return bool(
            re.search(r"(?<!\w)" + re.escape(norm(value)) + r"(?!\w)", norm(text))
        )

    past = " ".join(m["content"] for m in history)
    supports = []
    for i, row in enumerate(obs):
        attested = {
            a["slot"]: val
            for a in frame["actions"]
            if a.get("slot") in row
            for val in a["canonical_values"]
            if isinstance(row[a["slot"]], str)
            and norm(row[a["slot"]]) == norm(val)
            and norm(val) not in ("true", "false", "yes", "no", "")
            and contains(utterance, val)
        }
        if len(set(map(norm, attested.values()))) >= 2 and any(
            not contains(past, val) for val in attested.values()
        ):
            supports.append({"result_index": i, "fields": attested})
    if not supports:
        return None
    action = qwen_tools.assistant_message(
        [{"name": call["method"], "arguments": params}]
    )
    return {
        "messages": history,
        "tools": defs,
        "target": action,
        "observation": obs,
        "reply": utterance,
        "supports": supports,
        "domain": service["service_name"].split("_")[0],
        "service": service["service_name"],
        "method": call["method"],
    }


def build():
    old = read(PRIOR)
    files = training_files()
    assert all(files[k] == v for k, v in old["files"].items() if k.startswith("train/"))
    schemas = {s["service_name"]: s for s in read(ROOT / "train/schema.json")}
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    held = {
        norm(q)
        for r in read("research/data/conditional-program-panel.json")["rows"]
        for q in r["questions"]
    }
    pool = []
    counts = Counter()
    for rel, sha in sorted(files.items()):
        if not rel.startswith("train/"):
            continue
        assert digest(ROOT / rel) == sha
        if "dialogues_" not in rel:
            continue
        for dialogue in read(ROOT / rel):
            history = []
            for j, t in enumerate(dialogue["turns"]):
                if t["speaker"] == "USER":
                    if norm(t["utterance"]) in held:
                        break
                    history.append({"role": "user", "content": t["utterance"]})
                    continue
                frames = [f for f in t["frames"] if "service_call" in f]
                if not frames:
                    history.append({"role": "assistant", "content": t["utterance"]})
                    continue
                if len(frames) != 1:
                    break
                f = frames[0]
                if f["service"].split("_")[0] not in DOMAINS:
                    break
                counts["first source calls in enrolled domains"] += 1
                row = candidate(schemas[f["service"]], f, t["utterance"], history)
                if row is None:
                    break
                key = dialogue["dialogue_id"]
                row["id"] = f"sgd/{key}/{j}"
                row["file"] = rel
                row["dialogue"] = key
                following = {
                    "messages": [
                        *history,
                        row["target"],
                        {"role": "tool", "content": compact(row["observation"])},
                    ],
                    "tools": row["tools"],
                    "target": {"role": "assistant", "content": row["reply"]},
                }
                a, y = token_pair(tok, row)
                b, z = token_pair(tok, following)
                row["lengths"] = [len(a), len(y), len(b), len(z)]
                if (
                    max(len(a) + len(y), len(b) + len(z)) <= 4096
                    and max(len(y), len(z)) <= 384
                ):
                    pool.append(row)
                else:
                    counts["supported but complete turn over limit"] += 1
                break
    OUT.mkdir(exist_ok=True, parents=True)
    path = OUT / "training.json.gz"
    if path.exists():
        assert unpack(path) == pool
    else:
        packed(path, pool)
    old_ids = {
        r["dialogue"]
        for r in old["examples"]
        if "dialogue" in r and r.get("id", "").startswith("train-")
    }
    result = {
        "source": BASE,
        "train_files": files,
        "upstream_training_tree": TREE,
        "training": digest(path),
        "prior_registration": digest(PRIOR),
        "counts": counts,
        "dialogues": len(pool),
        "turns": 2 * len(pool),
        "domains": dict(Counter(r["domain"] for r in pool)),
        "distinct_methods": len({r["method"] for r in pool}),
        "multi_method_schemas": sum(len(r["tools"]) > 1 for r in pool),
        "observed_result_rows": sum(len(r["observation"]) for r in pool),
        "prior_clarification_or_action_dialogues": sum(
            r["dialogue"] in old_ids for r in pool
        ),
        "attested_fields": sum(len(r["supports"][0]["fields"]) for r in pool),
        "max_complete_tokens": max(sum(r["lengths"][2:]) for r in pool),
    }
    assert len({r["dialogue"] for r in pool}) == len(pool)
    save(OUT / "qualification.json", result)
    print({k: v for k, v in result.items() if k != "train_files"})


if __name__ == "__main__":
    build()
