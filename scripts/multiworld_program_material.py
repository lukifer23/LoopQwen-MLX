"""Recover training-only human SQL that survives multiple separately annotated table edits."""

import csv
import hashlib
import io
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.table_worlds import SOURCE, prepare_tables, query_results
from scripts.changed_table_sources import CACHE, FILES, REV, labels, norm, table
from scripts.squall_table_programs import LOCAL, SHA, translate

OUT = Path("research/sources/multiworld-programs")


def source_pool(upstream):
    ratings = labels()
    held = read(SOURCE)["cohorts"]["dev"]
    contexts = {w["context"] for w in held}
    questions = {norm(w["question"]) for w in held}
    held_tables = {fingerprint(v["rows"]) for w in held for v in w["versions"]}
    natural = read("research/sources/grounded-table-sentences/originals.json")["test"]
    questions.update(norm(w["question"]) for w in natural)
    held_tables.update(fingerprint(w["rows"]) for w in natural)
    pool = {}
    with ZipFile(CACHE / "original.zip") as archive:
        root = f"WikiTableQuestions-{REV}/"
        data = archive.read(root + "data/random-split-1-train.tsv").decode()
        for q in csv.DictReader(
            io.StringIO(data), delimiter="\t", quoting=csv.QUOTE_NONE
        ):
            key = q["id"]
            if (
                key not in upstream
                or q["context"] in contexts
                or norm(q["utterance"]) in questions
            ):
                continue
            votes = ratings.get(key, {})
            if 0 not in votes or norm(votes[0]["answer"]) != norm(q["targetValue"]):
                continue
            if len({norm(v["answer"]) for v in votes.values()}) < 3:
                continue
            raw = archive.read(root + q["context"])
            rows = table(raw, ",")
            if rows is None or fingerprint(rows) in held_tables:
                continue
            pool[key] = {
                "id": key,
                "context": q["context"],
                "question": q["utterance"],
                "split": "train",
                "versions": [
                    {
                        "world": 0,
                        "rows": rows,
                        **votes[0],
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                ],
            }
    with tarfile.open(CACHE / "altered.tar.gz", mode="r|gz") as archive:
        for member in archive:
            if len(archive.members) > 1000:
                archive.members.clear()
            if not member.isfile():
                continue
            _, key, name = member.name.split("/")
            if key not in pool or not name.endswith(".tsv"):
                continue
            world = int(name[:-4])
            if not world or world not in ratings[key]:
                continue
            raw = archive.extractfile(member).read()
            rows = table(raw, "\t")
            original = pool[key]["versions"][0]["rows"]
            if (
                rows is None
                or rows[0] != original[0]
                or rows == original
                or fingerprint(rows) in held_tables
            ):
                continue
            pool[key]["versions"].append(
                {
                    "world": world,
                    "rows": rows,
                    **ratings[key][world],
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    return pool


def build():
    for name, (_, checksum) in FILES.items():
        assert digest(CACHE / name) == checksum
    assert digest(LOCAL) == SHA
    upstream = {r["nt"]: r for r in read(LOCAL) if r["sql"]}
    pool = source_pool(upstream)
    counts = Counter()
    accepted = []
    for key, w in sorted(pool.items()):
        versions = sorted(w["versions"], key=lambda v: v["world"])
        assert len({v["world"] for v in versions}) == len(versions)
        w["versions"] = versions
        if len({norm(v["answer"]) for v in versions}) < 3:
            counts["fewer than three distinct human answers after source checks"] += 1
            continue
        if norm(upstream[key]["tgt"]) != norm(versions[0]["answer"]):
            counts["SQUALL answer differs from original human agreement"] += 1
            continue
        try:
            statement = translate(upstream[key], w)
        except ValueError as e:
            counts[str(e)] += 1
            continue
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            prepare_tables(path, [w])
            results = query_results(statement, w, path)
        if not all(r["correct"] for r in results):
            counts["executed program misses at least one eligible human answer"] += 1
            continue
        for v, r in zip(versions, results, strict=True):
            v["observed_scalar"] = r["observation"]["rows"][0][0]
        accepted.append({**w, "statement": statement})
        counts["accepted"] += 1
    old = read("research/sources/aligned-table-programs/execution.json")["executed"]
    old_ids = {r["id"] for r in old if r["status"] == "correct both"}
    by_world = Counter(len(w["versions"]) for w in accepted)
    kinds = defaultdict(set)
    for w in accepted:
        for token in (
            "COUNT",
            "SUM",
            "AVG",
            "MAX",
            "MIN",
            "ORDER BY",
            "WHERE",
            "GROUP BY",
            "LIMIT",
            "DISTINCT",
            "SELECT",
        ):
            if token in w["statement"]:
                kinds[token].add(w["id"])
    OUT.mkdir(exist_ok=True, parents=True)
    data = OUT / "training.json.gz"
    if data.exists():
        assert unpack(data) == accepted
    else:
        packed(data, accepted)
    info = {
        "original_source": REV,
        "squall_sha256": SHA,
        "source_files": {k: h for k, (_, h) in FILES.items()},
        "eligible_originals": len(pool),
        "disposition": counts,
        "questions": len(accepted),
        "distinct_questions": len({norm(r["question"]) for r in accepted}),
        "source_tables": len({r["context"] for r in accepted}),
        "distinct_programs": len({w["statement"] for w in accepted}),
        "overlap_with_previous_149": len({r["id"] for r in accepted} & old_ids),
        "version_count_histogram": {str(k): v for k, v in sorted(by_world.items())},
        "annotated_worlds": sum(k * v for k, v in by_world.items()),
        "raw_agreeing_ratings": sum(
            len(v["votes"]) for w in accepted for v in w["versions"]
        ),
        "operator_presence": {k: len(v) for k, v in kinds.items()},
        "training_sha256": digest(data),
    }
    assert sum(counts.values()) == len(pool)
    save(OUT / "qualification.json", info)
    print(info, flush=True)


if __name__ == "__main__":
    build()
