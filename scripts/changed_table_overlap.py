"""Compare each held source with every enrolled training table and original question."""

from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.io import digest, read, save
from loopqwen.research.table_worlds import SOURCE
from scripts.assistance_deliberation import write_csv
from scripts.assistance_source_overlap import nearest, words


def cells(rows):
    # Repeated values retain their multiplicity. Header overlap is reported separately.
    return Counter(
        " ".join(words(cell)) for row in rows[1:] for cell in row if words(cell)
    )


def similarity(a, b):
    common = sum((a & b).values())
    return common / (sum(a.values()) + sum(b.values()) - common)


def build():
    data = read(SOURCE)["cohorts"]
    train, held = data["train"], data["dev"]
    questions = nearest(
        "original_utterance",
        [(w["id"], w["question"]) for w in train],
        [(w["id"], "changed_table", w["question"]) for w in held],
    )
    lookup = {r["id"]: r for r in questions}
    counts = defaultdict(int)
    index = []
    for w in train:
        for v in w["versions"]:
            index.append(
                (
                    w["id"],
                    v["world"],
                    cells(v["rows"]),
                    v["sha256"],
                    tuple(v["rows"][0]),
                )
            )
    records = []
    for w in held:
        for v in w["versions"]:
            raw = cells(v["rows"])
            high, key, world = max(
                (similarity(raw, t), k, n) for k, n, t, _, _ in index
            )
            row = dict(
                lookup[w["id"]],
                version=v["world"],
                closest_table_cell_overlap=high,
                closest_training_table=key,
                closest_training_world=world,
                matching_source_hashes=sum(
                    sha == v["sha256"] for _, _, _, sha, _ in index
                ),
                identical_ordered_column_names=sum(
                    h == tuple(v["rows"][0]) for _, _, _, _, h in index
                ),
            )
            records.append(row)
            counts["held_versions"] += 1
            counts["duplicate_table_versions"] += row["matching_source_hashes"] > 0
            counts["held_versions_with_matching_ordered_schema"] += (
                row["identical_ordered_column_names"] > 0
            )
    output = Path("paper/changed-table-source-overlap.csv")
    write_csv(output, records)
    result = {
        "upstream_enrollment": digest(SOURCE),
        "training_tables": len(index),
        "training_original_questions": len(train),
        "held_questions": len(questions),
        "duplicate_question_wording": sum(
            r["normalized_exact_matches"] > 0 for r in questions
        ),
        "highest_question_fourgram_jaccard": max(
            r["nearest_fourgram_jaccard"] for r in questions
        ),
        "highest_cell_multiset_jaccard": max(
            r["closest_table_cell_overlap"] for r in records
        ),
        **counts,
        "rows": digest(output),
    }
    target = Path("research/sources/changed-tables/text-overlap.json")
    if target.exists():
        assert read(target) == result
    else:
        save(target, result)
    return result


if __name__ == "__main__":
    print(build())
