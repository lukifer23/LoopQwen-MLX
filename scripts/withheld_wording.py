"""Audit complete questions, evidence and human replies against actual fit inputs and targets."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, save
from scripts.conditional_source_overlap import training_fields
from scripts.program_language_material import norm

SOURCES = Path("research/sources")


def contrastive_fields(withheld):
    for w in withheld["tables"]:
        yield w["id"], "tables", "question", w["question"]
    for w in withheld["qualitative"]:
        for m in w["members"]:
            for key in ("question", "background"):
                yield m["id"], "qualitative", key, m[key]
    for pair in withheld["services"]:
        for w in pair:
            msgs = w["messages"]
            assert msgs[-1]["role"] == "user"
            yield w["id"], "service", "last_user", msgs[-1]["content"]
            yield (
                w["id"],
                "service",
                "full_conversation",
                " ".join(m["content"] for m in msgs),
            )
            yield w["id"], "service", "human_reply", w["reply"]


def query_fields(withheld):
    for w in withheld:
        f, k = w["family"], w["id"]
        yield k, f, "question", w["question"]
        for i, evidence in enumerate(w["evidence"] if f == "report" else w["facts"]):
            yield k, f, f"evidence/{i}", evidence
        if f == "strategy":
            for i, step in enumerate(w["decomposition"]):
                yield k, f, f"human_step/{i}", step


def repair_fields(rows):
    yield from query_fields(r for r in rows if r["family"] in ("strategy", "report"))
    for r in rows:
        if r["family"] == "field" and r["condition"] == "unassisted":
            for field, key in (
                ("question", "question"),
                ("human_reply", "human"),
                ("expert_sql", "expert_sql"),
            ):
                yield r["id"], r["family"], field, r[key]


def fit_fields(row):
    yield from training_fields(row)
    # A service transcript spans messages. Index the conversation as a contiguous
    # field as well, stopping at its first structured assistant call or observation.
    dialogue = []
    for msg in row["messages"]:
        if (
            msg["role"] not in {"user", "assistant"}
            or not isinstance(msg.get("content"), str)
            or not msg["content"].strip()
            or msg.get("tool_calls")
        ):
            break
        dialogue.append(msg["content"])
    if len(dialogue) > 1:
        yield "joined natural dialogue", " ".join(dialogue)


def scan(fields, training):
    # One anchor at each held position finds the longest contiguous normalized run.
    # The hash of a normalized training field lets repeated tool declarations and
    # replicated dialogue prefixes share the same scan. Preserve its first provenance.
    entries = []
    anchors = defaultdict(list)
    for k, family, field, text in fields:
        words = tuple(norm(text).split())
        width = min(4, len(words))
        assert width
        ix = len(entries)
        entries.append((k, family, field, words, width))
        for j in range(len(words) - width + 1):
            anchors[words[j : j + width]].append((ix, j))
    sizes = sorted({len(a) for a in anchors})
    seen = set()
    best = [{} for _ in entries]
    turns = 0
    for path in training:
        for r in unpack(path):
            turns += 1
            for kind, text in fit_fields(r):
                s = norm(text)
                sha = sha256(s.encode()).digest()
                if sha in seen:
                    continue
                seen.add(sha)
                words = tuple(s.split())
                for width in sizes:
                    for j in range(len(words) - width + 1):
                        for ix, pos in anchors.get(words[j : j + width], ()):
                            target = entries[ix][3]
                            if best[ix].get("matched_words") == len(target):
                                continue
                            if j and pos and words[j - 1] == target[pos - 1]:
                                continue
                            n = width
                            while (
                                pos + n < len(target)
                                and j + n < len(words)
                                and target[pos + n] == words[j + n]
                            ):
                                n += 1
                            if n > best[ix].get("matched_words", 0):
                                best[ix] = {
                                    "matched_words": n,
                                    "at": pos,
                                    "phrase": " ".join(target[pos : pos + n]),
                                    "training_id": r["id"],
                                    "training_field": kind,
                                    "training_group": r["group"],
                                }
    result = []
    for (key, family, f, words, width), hit in zip(entries, best, strict=True):
        result.append(
            {
                "id": key,
                "family": family,
                "field": f,
                "words": len(words),
                "anchor": width,
                "whole": hit.get("matched_words", 0) == len(words),
                **hit,
            }
        )
    counts = defaultdict(Counter)
    for r in result:
        c = counts[r["family"] + "/" + r["field"].split("/")[0]]
        c["fields"] += 1
        c["whole"] += r["whole"]
        c["whole_at_least_8_words"] += r["whole"] and r["words"] >= 8
        c["at_least_12_contiguous"] += r.get("matched_words", 0) >= 12
    return {
        "turns": turns,
        "unique_normalized_training_fields": len(seen),
        "groups": counts,
        "fields": result,
    }


def build(study):
    if study == "contrastive-worlds":
        held = SOURCES / study / "withheld.json.gz"
        training = [
            SOURCES / "program-language" / "training.json.gz",
            SOURCES / study / "training.json.gz",
        ]
        fields = contrastive_fields(unpack(held))
    elif study == "repair-query":
        held = SOURCES / study / "transfer-panel.json.gz"
        training = [SOURCES / study / "verified-training.json.gz"]
        fields = repair_fields(unpack(held)["rows"])
    else:
        held = SOURCES / study / "held.json.gz"
        training = [SOURCES / study / "training.json.gz"]
        fields = query_fields(unpack(held))
    result = scan(fields, training)
    result.update(
        training={str(p): digest(p) for p in training}, held_source=digest(held)
    )
    save(Path("docs/measurements") / (study + "-wording-overlap.json"), result)
    print(study, result["turns"], dict(result["groups"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "study", choices=("contrastive-worlds", "shared-query", "repair-query")
    )
    build(p.parse_args().study)
