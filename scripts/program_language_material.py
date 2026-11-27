"""Build whole human responses and executed SQL supervision for a shared decoder."""

import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from loopqwen import qwen_tools
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.relational_tools import tools
from loopqwen.research.checked_rationale import checked_response
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.table_worlds import (
    FILENAME,
    SOURCE,
    mount,
    prepare_tables,
    query_results,
    state,
)
from loopqwen.tool_session import compact
from scripts.grounded_table_sentences import FOLDER as FETA
from scripts.rule_dialogue_material import OUT_OF_SCOPE
from scripts.squall_table_programs import DEST as SQUALL

DEST = Path("research/sources/program-language")
SEED = 271952
DOLLY = Path("data/research_external/dolly/original.jsonl")
DOLLY_REV = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DOLLY_SHA = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
GSM = Path("data/research_external/gsm-train.jsonl")
GSM_SHA = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
OLD = Path("research/data/table-policy-training.json")
PANEL = Path("research/data/table-policy-panel.json")
LIMIT = 1792
RESPONSE_LIMIT = 384


def norm(s):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold()))


def shingles(s, n=12):
    words = norm(s).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def source_lines(path, sha):
    if path == DOLLY and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(
            f"https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/{DOLLY_REV}/databricks-dolly-15k.jsonl"
        ) as stream:
            path.write_bytes(stream.read())
    assert digest(path) == sha
    return [json.loads(line) for line in path.read_text().splitlines()]


def turn(key, group, messages, defs, target, **extra):
    return dict(
        id=key, group=group, messages=messages, tools=defs, target=target, **extra
    )


def answer(text, reasoning=""):
    return {
        "role": "assistant",
        "content": text,
        **({"reasoning_content": reasoning} if reasoning else {}),
    }


def user(text):
    return {"role": "user", "content": text}


def call(provider, **args):
    return qwen_tools.assistant_message([{"name": provider.name, "arguments": args}])


def observed(content):
    return {"role": "tool", "content": compact(content)}


def human_programs(tmp):
    worlds = {w["id"]: w for w in read(SOURCE)["cohorts"]["train"]}
    pref = {r["id"]: r for r in read(SQUALL / "preferences.json")}
    positives = [
        r
        for r in read(SQUALL / "execution.json")["executed"]
        if r["status"] == "correct both"
    ]
    prepare_tables(tmp, [worlds[r["id"]] for r in positives])
    for r in positives:
        key = r["id"]
        w = worlds[key]
        msgs, defs = state(w, tmp)
        provider = tools([tmp / key / "0" / FILENAME])[1]

        def action(statement):
            return call(provider, database_name=FILENAME, statement=statement)

        target = action(r["statement"])
        outcomes = query_results(r["statement"], w, tmp)
        assert all(o["correct"] for o in outcomes)
        negative = []
        for other in pref.get(key, {}).get("rejected", []):
            actual = query_results(other["sql"], w, tmp)
            assert all(o["used"] and not o["correct"] for o in actual)
            assert [o["observation"]["rows"][0][0] for o in actual] == other["results"]
            negative.append({"edit": other["edit"], "target": action(other["sql"])})
        yield turn(key + "-program", "program", msgs, defs, target, rejected=negative)
        yield turn(key + "-schema", "discovery", msgs[:1], defs, msgs[1])
        for v, o in zip(w["versions"], outcomes, strict=True):
            yield turn(
                key + "-result-" + str(v["world"]),
                "scalar",
                [*msgs, target, observed(o["observation"])],
                defs,
                answer(v["answer"]),
            )


def table_prompt(r, inline):
    title = r["page"] + (" — " + r["section"] if r["section"] else "")
    lines = [r["question"], "Table: " + title]
    if inline:
        lines.append(json.dumps(r["rows"], ensure_ascii=False, separators=(",", ":")))
    return "\n\n".join(lines)


def natural_tables(tmp, records):
    for r in records:
        key = "feta-" + str(r["id"])
        path = mount(tmp / key, r["rows"])
        schema, query = tools([path])
        defs = qwen_tools.declarations([schema, query])
        args = {
            "database_name": FILENAME,
            "statement": "SELECT * FROM records ORDER BY rowid",
        }
        result = query.callback(**args)
        assert (
            not result["truncated"]
            and result["columns"] == r["rows"][0]
            and result["rows"] == r["rows"][1:]
        )
        schema_args = {"database_name": FILENAME}
        mounted = [
            user(table_prompt(r, False)),
            call(schema, **schema_args),
            observed(schema.callback(**schema_args)),
            call(query, **args),
            observed(result),
        ]
        for suffix, msgs, catalog in [
            ("inline", [user(table_prompt(r, True))], []),
            ("observed", mounted, defs),
        ]:
            yield turn(
                key + "-" + suffix, "explanation", msgs, catalog, answer(r["reference"])
            )


def dolly_candidates(tok, forbidden):
    by = defaultdict(list)
    seen = set()
    for i, r in enumerate(source_lines(DOLLY, DOLLY_SHA)):
        q, ctx, a = (r[k].strip() for k in ("instruction", "context", "response"))
        if not unicodedata.is_normalized("NFC", q + ctx + a):
            continue
        if not q or not a or OUT_OF_SCOPE.search(" ".join([q, ctx, a])):
            continue
        if re.search(r"<[|/]?(?:think|tool_call|im_start|im_end)", q + ctx + a, re.I):
            continue
        text = q + "\n\n" + ctx if ctx else q
        identity = norm(text)
        if identity in seen or identity in forbidden or norm(q) in forbidden:
            continue
        row = turn(
            "dolly-" + str(i),
            "instruction",
            [user(text)],
            [],
            answer(a),
            category=r["category"],
        )
        p, y = token_pair(tok, row)
        if len(p) + len(y) > LIMIT or not 9 <= len(y) <= RESPONSE_LIMIT:
            continue
        seen.add(identity)
        by[r["category"]].append((row, q, ctx))
    reserved, training = [], []
    blocked = set()
    split_ids = set()
    question_set = set()
    # Cover every source category before taking any training references.
    for cat, entries in sorted(by.items()):
        entries.sort(key=lambda e: fingerprint([SEED, e[0]["id"]]))
        for row, q, ctx in entries:
            tokens = shingles(q) | shingles(ctx) | shingles(row["target"]["content"])
            if norm(q) in question_set or tokens & blocked:
                continue
            reserved.append(row)
            split_ids.add(row["id"])
            question_set.add(norm(q))
            blocked.update(tokens)
            if sum(r["category"] == cat for r in reserved) == 3:
                break
    assert len(reserved) == 24
    for cat, entries in sorted(by.items()):
        selected = 0
        for row, q, ctx in entries:
            if (
                row["id"] in split_ids
                or norm(q) in question_set
                or (shingles(q) | shingles(ctx) | shingles(row["target"]["content"]))
                & blocked
            ):
                continue
            training.append(row)
            selected += 1
            if selected == 384:
                break
        assert selected == 384
    return training, reserved


def math_responses(blocked):
    historical = set().union(
        *json.loads(
            gzip.decompress(
                Path("research/registry/historical-text-ids.json.gz").read_bytes()
            )
        ).values()
    )
    raw = source_lines(GSM, GSM_SHA)
    used = set()
    for i, r in sorted(enumerate(raw), key=lambda t: fingerprint([SEED, "gsm", t[0]])):
        q = r["question"].strip()
        original = " ".join(q.split())
        if norm(q) in blocked | used or OUT_OF_SCOPE.search(q + " " + r["answer"]):
            continue
        if any(
            hashlib.sha256(("text:" + original + s).encode()).hexdigest() in historical
            for s in (
                "",
                " Answer:",
                " Show your steps. End with Final answer: <number>. Solution:",
            )
        ):
            continue
        if checked_response(r["answer"]) is None:
            continue
        reasoning, final = r["answer"].rsplit("####", 1)
        reasoning = re.sub(r"<<[^<>]+>>", "", reasoning).strip()
        yield turn(
            "gsm-" + str(i),
            "arithmetic",
            [user(q)],
            [],
            answer(final.strip(), reasoning),
        )
        used.add(norm(q))
        if len(used) == 1024:
            return
    raise ValueError("Insufficient original math questions")


def build():
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    feta = read(FETA / "originals.json")
    forbidden = {norm(q) for r in read(PANEL)["rows"] for q in r["questions"]}
    forbidden.update(norm(r["question"]) for r in feta["test"])
    for path in Path("research/data").glob("*panel.json"):
        obj = read(path)
        rows = obj.get("rows", []) if isinstance(obj, dict) else obj
        for row in rows:
            if not isinstance(row, dict):
                continue
            for field in ("questions", "question", "prompt"):
                val = row.get(field, [])
                for q in (
                    [val]
                    if isinstance(val, str)
                    else val
                    if isinstance(val, list)
                    else []
                ):
                    if isinstance(q, str):
                        forbidden.add(norm(q))
    broad, held = dolly_candidates(tok, forbidden)
    forbidden.update(norm(r["messages"][0]["content"]) for r in held)
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        candidates = [
            *human_programs(folder),
            *natural_tables(folder, feta["train"]),
            *broad,
            *math_responses(forbidden),
        ]
    for old in read(OLD)["examples"]:
        if not old["group"].startswith("table_"):
            candidates.append({**old, "group": "legacy", "source_group": old["group"]})
    joined = []
    excluded = Counter()
    groups = defaultdict(Counter)
    for row in candidates:
        if any(
            norm(m["content"]) in forbidden
            for m in row["messages"]
            if m["role"] == "user"
        ):
            excluded[row["group"] + "/held user prompt"] += 1
            continue
        if not unicodedata.is_normalized("NFC", json.dumps(row, ensure_ascii=False)):
            excluded[
                row["group"] + "/source differs from tokenizer Unicode normalization"
            ] += 1
            continue
        p, y = token_pair(tok, row)
        if len(p) + len(y) > LIMIT or len(y) > RESPONSE_LIMIT:
            excluded[row["group"] + "/complete turn exceeds limit"] += 1
            continue
        for n in row.get("rejected", []):
            np, ny = token_pair(tok, {**row, "target": n["target"]})
            assert (
                p == np
                and y != ny
                and len(p) + len(ny) <= LIMIT
                and len(ny) <= RESPONSE_LIMIT
            )
        counts = groups[row["group"]]
        counts["examples"] += 1
        counts["input_positions"] += len(p) + len(y) - 1
        counts["all_response_positions"] += len(y)
        counts["execution_negatives"] += len(row.get("rejected", []))
        joined.append(row)
    assert len({r["id"] for r in joined}) == len(joined)
    assert {r["id"] for r in joined}.isdisjoint({r["id"] for r in held})
    DEST.mkdir(exist_ok=True, parents=True)
    name = DEST / "training.json.gz"
    packed(name, joined)
    assert unpack(name) == joined
    save(DEST / "fresh-requests.json", held)
    info = {
        "selection_seed": SEED,
        "complete_sequence_limit": LIMIT,
        "response_limit": RESPONSE_LIMIT,
        "sources": {
            str(p): digest(p)
            for p in (
                DOLLY,
                GSM,
                SOURCE,
                SQUALL / "preferences.json",
                SQUALL / "execution.json",
                FETA / "originals.json",
                OLD,
                PANEL,
            )
        },
        "groups": groups,
        "omitted": excluded,
        "training_turns": len(joined),
        "training": digest(name),
        "reserved_instructions": digest(DEST / "fresh-requests.json"),
        "reserved_by_category": dict(Counter(r["category"] for r in held)),
        "held_normalized_prompt_count": len(forbidden),
    }
    save(DEST / "qualification.json", info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    build()
