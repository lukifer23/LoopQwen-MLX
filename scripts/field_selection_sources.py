"""Join original InfoTabS claims, human evidence votes and a disjoint cross-domain split."""

import ast
import csv
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from loopqwen.io import digest
from loopqwen.research.reasoned_evidence_data import choose

SOURCE = Path("data/research_external/infotabs/source.zip")
ANNOTATION = SOURCE.with_name("annotation.zip")
SHA = "86c62510db0a8200967205313fdf8b3183a8d7c7ea73e8ef4185207606f04cb3"
ANNOTATION_SHA = "8df9ce4852ebf143172411005890732b2a9138cdb691559f927fa866d352e476"
SEED = 209345
LABELS = {"E": "true", "C": "false", "N": "uncertain"}
NUMERIC = {"E": 0, "C": 2, "N": 1}


def records(z, suffix):
    (name,) = [n for n in z.namelist() if n.endswith(suffix)]
    with z.open(name) as f:
        return list(
            csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"), delimiter="\t")
        )


def take_tables(*, excluded_test=(), sample_seed=SEED, held_split="test_alpha3"):
    assert digest(SOURCE) == SHA and digest(ANNOTATION) == ANNOTATION_SHA
    splits = {}
    with ZipFile(SOURCE) as z, ZipFile(ANNOTATION) as ann:
        tables = {
            Path(n).stem: json.loads(z.read(n))
            for n in z.namelist()
            if "/tables/json/" in n and n.endswith(".json")
        }
        validations = {
            (r["table_id"], r["hypothesis"]): r
            for r in records(z, "/infotabs_valid_" + held_split + ".tsv")
        }
        for split, member, limit in (
            (
                held_split,
                "results_test/modified_batches_integrated_batch_annotations.tsv",
                12,
            ),
            (
                "train",
                "results_train/modified_row_annotated_batches_integrated_batch_annotations.tsv",
                128,
            ),
        ):
            gold = {
                (r["table_id"], r["hypothesis"], r["label"])
                for r in records(z, "/maindata/infotabs_" + split + ".tsv")
            }
            groups = defaultdict(lambda: defaultdict(list))
            forbidden = {
                w["fields"]["title"].casefold() for w in splits.get(held_split, [])
            }
            for r in records(ann, member):
                ident, h, y = r["Table ID"], r["Hypothesis"], r["Truth Label"]
                if (
                    (ident, h, y) not in gold
                    or r["#Rows"] != "1"
                    or r["OOT Used"] != "0"
                ):
                    continue
                fields = {k: " | ".join(v) for k, v in tables[ident].items()}
                if (
                    not fields.get("title")
                    or fields["title"].casefold() in forbidden
                    or len(str(fields)) > 1700
                ):
                    continue
                keys = [
                    k
                    for k in ast.literal_eval(r["Rows Used"])
                    if k.casefold() != "title"
                ]
                if len(keys) != 1:
                    continue
                actual = [k for k in fields if k.casefold() == keys[0].casefold()]
                if len(actual) != 1 or not 1 <= len(fields[actual[0]].split()) <= 30:
                    continue
                # Each ballot is a binary field mask. Count exact agreement on the unique field.
                majority = r["Majority Label"]
                if majority.startswith("["):
                    (majority,) = ast.literal_eval(majority)
                votes = ast.literal_eval(r["Labels"])
                evidence_votes = votes.count(majority)
                if majority.count("1") != 1 or evidence_votes < 3:
                    continue
                row = {
                    "claim": h,
                    "label": LABELS[y],
                    "field": actual[0],
                    "field_votes": evidence_votes,
                    "annotators": len(votes),
                }
                if split != "train":
                    v = validations.get((ident, h))
                    if not v:
                        continue
                    decisions = ast.literal_eval(v["other_human_labels"])
                    support = decisions.count(NUMERIC[y])
                    if (
                        support < 4
                        or int(v["majoritylabel"]) != NUMERIC[y]
                        or int(v["first_human_label"]) != NUMERIC[y]
                    ):
                        continue
                    row["verdict_votes"] = support
                    row["validator_count"] = len(decisions)
                groups[ident][y].append(row)
            selected = []
            used = set()
            for ident in choose(groups, sample_seed, str):
                if split == held_split and ident in excluded_test:
                    continue
                by_label = groups[ident]
                if set(by_label) != set(LABELS):
                    continue
                fields = {k: " | ".join(v) for k, v in tables[ident].items()}
                if fields["title"].casefold() in used:
                    continue
                used.add(fields["title"].casefold())
                selected.append(
                    {
                        "id": ident,
                        "fields": fields,
                        "claims": [
                            choose(by_label[y], sample_seed, lambda r: r["claim"])[0]
                            for y in LABELS
                        ],
                    }
                )
                if len(selected) == limit:
                    break
            assert len(selected) == limit, (split, len(selected))
            splits[split] = selected
    for key in ("id", "claim", "title"):

        def values(rows, key=key):
            if key == "claim":
                return {
                    a["claim"].strip().casefold() for w in rows for a in w["claims"]
                }
            return {
                w["id"] if key == "id" else w["fields"]["title"].casefold()
                for w in rows
            }

        assert not values(splits["train"]) & values(splits[held_split])
    return splits


def roles():
    from scripts.table_question_panel import SHA as wiki_sha

    source = Path("data/research_external/wikisql/source.tar.bz2")
    assert digest(source) == wiki_sha
    with tarfile.open(source, "r:bz2") as t:
        rows = list(map(json.loads, t.extractfile("data/train.tables.jsonl")))
    result = []
    for w in choose(rows, SEED, lambda row: row["id"]):
        h = w["header"]
        if (
            not 3 <= len(h) <= 8
            or len(set(h)) != len(h)
            or not 3 <= len(w["rows"]) <= 22
        ):
            continue
        data = [[str(v) for v in row] for row in w["rows"]]
        if any(len(r) != len(h) for r in data):
            continue
        idx = [
            i
            for i in range(len(h))
            if len({r[i] for r in data}) == len(data)
            and all(0 < len(r[i]) <= 70 for r in data)
        ]
        if len(idx) < 2:
            continue
        a, b = choose(idx, SEED + w["id"].count("-"), str)[:2]
        rec = next(
            (
                r
                for r in data
                if r[a] not in {v[b] for v in data} and r[b] not in {v[a] for v in data}
            ),
            None,
        )
        if rec is None:
            continue
        result.append(
            {
                "id": w["id"],
                "header": h,
                "rows": data,
                "a": h[a],
                "b": h[b],
                "x": rec[a],
                "y": rec[b],
            }
        )
        if len(result) == 96:
            break
    assert len(result) == 96
    return result
