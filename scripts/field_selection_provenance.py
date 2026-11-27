"""Preserve original table records and anonymous ballots, and audit train/test text overlap."""

import argparse
import ast
import gzip
import json
import re
from pathlib import Path
from zipfile import ZipFile

from loopqwen.io import digest, read, save
from scripts.paper_overlap import normalize, shingle

LABELS = {"E": "true", "C": "false", "N": "uncertain"}


def study_paths(study):
    return (
        Path("research/sources") / study,
        Path("docs/measurements") / (study + "-provenance.json"),
        read("research/data/" + study + "-training.json")["tables"],
        read("research/data/" + study + "-panel.json")["tables"],
    )


def run(study="field-selection"):
    from scripts.field_selection_sources import (
        ANNOTATION,
        ANNOTATION_SHA,
        SHA,
        SOURCE,
        records,
    )

    folder, dest, tr, held = study_paths(study)
    split_name = "test_alpha1" if study == "valid-return" else "test_alpha3"
    assert digest(SOURCE) == SHA and digest(ANNOTATION) == ANNOTATION_SHA
    snapshots = []
    with ZipFile(SOURCE) as z, ZipFile(ANNOTATION) as ann:
        for split, rows, suffix in (
            (
                "train",
                tr,
                "results_train/modified_row_annotated_batches_integrated_batch_annotations.tsv",
            ),
            (
                split_name,
                held,
                "results_test/modified_batches_integrated_batch_annotations.tsv",
            ),
        ):
            ann_by = {
                (r["Table ID"], r["Hypothesis"], LABELS[r["Truth Label"]]): r
                for r in records(ann, suffix)
            }
            original = {
                (r["table_id"], r["hypothesis"], LABELS[r["label"]]): r
                for r in records(z, "/maindata/infotabs_" + split + ".tsv")
            }
            validators = (
                {
                    (r["table_id"], r["hypothesis"]): r
                    for r in records(z, "/infotabs_valid_" + split + ".tsv")
                }
                if split != "train"
                else {}
            )
            for w in rows:
                (name,) = [
                    n
                    for n in z.namelist()
                    if n.endswith("/tables/json/" + w["id"] + ".json")
                ]
                source = json.loads(z.read(name))
                assert {k: " | ".join(v) for k, v in source.items()} == w["fields"]
                claims = []
                for q in w["claims"]:
                    k = (w["id"], q["claim"], q["label"])
                    a = ann_by[k]
                    original_row = original[k]
                    ballot = {
                        key: a[key]
                        for key in (
                            "Labels",
                            "Majority Label",
                            "Rows Used",
                            "#Rows",
                            "OOT Used",
                            "#Annotators",
                        )
                    }
                    vote = validators.get(k[:2])
                    claims.append(
                        {
                            "hypothesis": original_row["hypothesis"],
                            "label": original_row["label"],
                            "evidence": ballot,
                            "validation": {
                                key: vote[key]
                                for key in (
                                    "first_human_label",
                                    "majoritylabel",
                                    "other_human_labels",
                                )
                            }
                            if vote
                            else None,
                        }
                    )
                snapshots.append(
                    {
                        "split": split,
                        "source_table_id": w["id"],
                        "original_table": source,
                        "claims": claims,
                    }
                )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "human-source-records.jsonl.gz"
    raw = "".join(
        json.dumps(w, ensure_ascii=False, separators=(",", ":")) + "\n"
        for w in snapshots
    ).encode()
    if path.exists():
        assert gzip.decompress(path.read_bytes()) == raw
    else:
        with (
            path.open("wb") as f,
            gzip.GzipFile(fileobj=f, mode="wb", filename="", mtime=0) as out,
        ):
            out.write(raw)

    def mask(claim, title):
        named = re.sub(re.escape(title), "<entity>", claim, flags=re.IGNORECASE)
        return normalize(named, True)

    training = {
        (w["id"], q["claim"]): mask(q["claim"], w["fields"]["title"])
        for w in tr
        for q in w["claims"]
    }
    lexical = []
    for w in held:
        for q in w["claims"]:
            words = mask(q["claim"], w["fields"]["title"])
            grams = shingle(words)

            def similarity(key, grams=grams):
                other = shingle(training[key])
                return len(grams & other) / len(grams | other) if grams | other else 0

            key = max(training, key=similarity)
            lexical.append(
                {
                    "table": w["id"],
                    "claim": q["claim"],
                    "closest_table": key[0],
                    "closest_claim": key[1],
                    "shingle_jaccard": similarity(key),
                    "masked_exact": words in training.values(),
                }
            )
    r = {
        "original_table_records": len(snapshots),
        "selected_claims": sum(len(w["claims"]) for w in snapshots),
        "preserved_anonymous_source_sha256": digest(path),
        "full_archives": {"infotabs": SHA, "evidence": ANNOTATION_SHA},
        "held_table_ids": len(held),
        "full_exact_title_overlap": len(
            {w["fields"]["title"].casefold() for w in held}
            & {w["fields"]["title"].casefold() for w in tr}
        ),
        "masked_exact_claims": sum(v["masked_exact"] for v in lexical),
        "maximum_masked_shingle_jaccard": max(v["shingle_jaccard"] for v in lexical),
        "eligibility": f"Mask a verbatim occurrence of the full title and normalize numbers, case and punctuation. Compare each original held statement to all {len(training)} original training statements. The paper_overlap.shingle routine uses contiguous five-word shingles and returns an empty set for shorter texts. No evaluation item is removed.",
        "nearest_by_claim": lexical,
    }
    if dest.exists():
        assert read(dest) == r
    else:
        save(dest, r)
    print(
        len(snapshots),
        "original tables and",
        r["selected_claims"],
        "original claims with human source ballots;",
        r["masked_exact_claims"],
        "masked exact matches, maximum Jaccard",
        r["maximum_masked_shingle_jaccard"],
    )


def compact(study="field-selection"):
    folder, _, training, held = study_paths(study)
    rows = [
        json.loads(line)
        for line in gzip.decompress(
            (folder / "human-source-records.jsonl.gz").read_bytes()
        ).splitlines()
    ]
    data = training + held
    assert len(rows) == len(data) == (76 if study == "valid-return" else 140)
    for w, src in zip(data, rows, strict=True):
        assert w["id"] == src["source_table_id"] and w["fields"] == {
            k: " | ".join(v) for k, v in src["original_table"].items()
        }
        for q, orig in zip(w["claims"], src["claims"], strict=True):
            assert (
                q["claim"] == orig["hypothesis"] and q["label"] == LABELS[orig["label"]]
            )
            b = orig["evidence"]
            fields = [
                s.casefold()
                for s in ast.literal_eval(b["Rows Used"])
                if s.casefold() != "title"
            ]
            mask = b["Majority Label"]
            if mask.startswith("["):
                (mask,) = ast.literal_eval(mask)
            ballots = ast.literal_eval(b["Labels"])
            assert (
                fields == [q["field"].casefold()]
                and len(ballots) == q["annotators"]
                and ballots.count(mask) == q["field_votes"] >= 3
                and mask.count("1") == 1
            )
            if src["split"] != "train":
                v = orig["validation"]
                majority = int(v["majoritylabel"])
                assert (
                    majority
                    == int(v["first_human_label"])
                    == {"E": 0, "C": 2, "N": 1}[orig["label"]]
                )
                ballots = ast.literal_eval(v["other_human_labels"])
                assert (
                    len(ballots) == q["validator_count"]
                    and ballots.count(majority) == q["verdict_votes"] >= 4
                )
    print(
        "Replayed the original field and verdict ballots for",
        sum(len(w["claims"]) for w in rows),
        "sampled claims.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--valid-return", action="store_true")
    args = parser.parse_args()
    study = "valid-return" if args.valid_return else "field-selection"
    if not args.compact:
        run(study)
    compact(study)
