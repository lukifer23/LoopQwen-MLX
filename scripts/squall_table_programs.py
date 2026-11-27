"""Transfer published human SQL annotations onto the exact two raw training-table versions."""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.table_worlds import (
    SOURCE,
    canonical,
    prepare_tables,
    query_results,
)

REV = "ee6dda91eacafd5ee800eb3fdc5d3a6575dc47e7"
BASE = f"https://raw.githubusercontent.com/tzshi/squall/{REV}/"
SHA = "ceb244393916dc035a2f71a784c634d6a5fd4595b63e05d3b2fa58883af4f5cd"
LOCAL = Path("data/research_external/squall/squall.json")
DEST = Path("research/sources/aligned-table-programs")
NUMBER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?")
KEYWORDS = frozenset(
    [
        "select",
        "from",
        "where",
        "=",
        "(",
        ")",
        "by",
        "1",
        "order",
        "limit",
        "count",
        "desc",
        "*",
        "and",
        "asc",
        "-",
        ">",
        "group",
        "+",
        "!=",
        "<",
        "in",
        ",",
        ">=",
        "<=",
        "abs",
        "sum",
        "0",
        "max",
        "having",
        "distinct",
        "min",
        "is",
        "null",
        "or",
        "between",
        "offset",
        "avg",
    ]
)


def normalized_header(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def column(token, record, world):
    match = re.fullmatch(r"c([1-9][0-9]*)(?:_(.+))?", token)
    if not match:
        raise ValueError("unrecognized column")
    source_index, kind = int(match[1]) - 1, match[2]
    header = normalized_header(record["columns"][source_index][0])
    raw_headers = world["versions"][0]["rows"][0]
    candidates = [
        i for i, h in enumerate(raw_headers) if normalized_header(h) == header
    ]
    if len(candidates) != 1:
        raise ValueError("missing or ambiguous original header")
    i = candidates[0]
    sql = '"' + raw_headers[i].replace('"', '""') + '"'
    if not kind:
        return f"({sql} COLLATE NOCASE)"
    if kind != "number":
        raise ValueError("unsupported source normalization: " + kind)
    values = {str(r[i]).strip() for v in world["versions"] for r in v["rows"][1:]}
    # A decimal cast is meaningful only if every present raw cell has a well-defined numeric interpretation.
    missing = sorted(values & {"", "-", "–", "—"})
    if not all(NUMBER.fullmatch(v) for v in values - set(missing)):
        raise ValueError("non-numeric raw value in projected numeric column")
    for value in missing:
        sql = f"NULLIF(TRIM({sql}), '{value}')"
    if any("," in value for value in values):
        sql = f"REPLACE({sql}, ',', '')"
    return f"CAST({sql} AS NUMERIC)"


def translate(record, world):
    group, file = world["context"].split("/")[1:]
    assert record["tbl"] == group.split("-")[0] + "_" + file.removesuffix(".csv")
    tokens = []
    for category, v, _ in record["sql"]:
        if category == "Column":
            tokens.append(column(v, record, world))
        elif category == "Literal.String":
            if not (v.startswith("'") and v.endswith("'")):
                raise ValueError("unrecognized string literal")
            tokens.append(v)
        elif category == "Literal.Number" and re.fullmatch(
            r"[+-]?[0-9]+(?:\.[0-9]+)?", v
        ):
            tokens.append(v)
        elif category == "Keyword":
            if v == "w":
                tokens.append("records")
            elif v == "id":
                headers = {h.casefold() for h in world["versions"][0]["rows"][0]}
                alias = next(
                    (a for a in ("rowid", "_rowid_", "oid") if a not in headers), None
                )
                if not alias:
                    raise ValueError("shadowed row number")
                tokens.append(alias)
            elif v in KEYWORDS:
                tokens.append(v.upper())
            else:
                raise ValueError("unsupported SQL keyword")
        else:
            raise ValueError("unsupported SQL token")
    if not tokens or tokens[0] != "SELECT":
        raise ValueError("expected a SELECT")
    return " ".join(tokens)


def build():
    if not LOCAL.is_file():
        LOCAL.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(BASE + "data/squall.json", timeout=90) as response:
            LOCAL.write_bytes(response.read())
    assert digest(LOCAL) == SHA
    upstream = {r["nt"]: r for r in read(LOCAL)}
    worlds = read(SOURCE)["cohorts"]
    train = worlds["train"]
    assert not ({r["context"] for r in train} & {r["context"] for r in worlds["dev"]})
    # Only training IDs enter the annotation join or any execution in this diagnostic.
    result = []
    tally = Counter()
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        prepare_tables(folder, train)
        for w in train:
            r = upstream.get(w["id"])
            if r is None or not r["sql"]:
                tally["no annotation"] += 1
                continue
            if canonical(r["tgt"]) != canonical(w["versions"][0]["answer"]):
                tally["source gold disagreement"] += 1
                continue
            try:
                sql = translate(r, w)
            except ValueError as e:
                tally[str(e)] += 1
                continue
            evaluated = query_results(sql, w, folder)
            status = (
                "correct both"
                if all(x["correct"] for x in evaluated)
                else (
                    "correct original only"
                    if evaluated[0]["correct"]
                    else (
                        "correct altered only"
                        if evaluated[1]["correct"]
                        else "neither correct"
                    )
                )
            )
            tally[status] += 1
            result.append(
                {
                    "id": w["id"],
                    "table": w["context"],
                    "question": w["question"],
                    "source_record_sha256": fingerprint(r),
                    "human_sql_tokens": r["sql"],
                    "statement": sql,
                    "status": status,
                    "versions": [
                        {
                            "world": v["world"],
                            "table_sha256": v["sha256"],
                            "gold": v["answer"],
                            **o,
                        }
                        for v, o in zip(w["versions"], evaluated, strict=True)
                    ],
                }
            )
    assert sum(tally.values()) == len(train)
    DEST.mkdir(parents=True, exist_ok=True)
    where = DEST / "execution.json"
    save(
        where,
        {
            "source_dataset_sha256": SHA,
            "training_tables": digest(SOURCE),
            "executed": result,
        },
    )
    info = {
        "annotation_source": "https://github.com/tzshi/squall",
        "revision": REV,
        "sha256": SHA,
        "paper": "https://aclanthology.org/2020.findings-emnlp.167/",
        "license": "CC BY-SA 4.0",
        "published_annotations": len(upstream),
        "enrolled_training_questions": len(train),
        "matched": sum(
            w["id"] in upstream and bool(upstream[w["id"]]["sql"]) for w in train
        ),
        "executed": len(result),
        "outcomes": dict(sorted(tally.items())),
        "accepted_ids": [v["id"] for v in result if v["status"] == "correct both"],
        "unique_accepted_statements": len(
            {v["statement"] for v in result if v["status"] == "correct both"}
        ),
        "execution_sha256": digest(where),
        "method": "Translate typed source columns by matching unique actual headers, preserve human SQL structure and constants, and run the same read-only statement against both complete source tables. Reject unsupported projections. Accept only exact single scalar denotations matching three agreeing human labels in each world.",
    }
    save(DEST / "qualification.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "accepted_ids"}, indent=2))


if __name__ == "__main__":
    build()
