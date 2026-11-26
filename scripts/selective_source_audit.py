"""Resolve every frozen manual excerpt directly against the pinned CPython paragraphs."""

from mlx_lm.utils import load_tokenizer
from selective_raw_material import DATA, TEST_DOCS, TRAIN_DOCS, docs, identity

from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot

DEST = DATA / "selective-manual-attribution.json"


def resolve():
    tok = load_tokenizer(local_snapshot())
    sources = {}
    select = {}
    rows = {}
    for filename, category, names in (
        ("selective-raw-panel", "new_documentation", TEST_DOCS),
        ("selective-raw-training", "documentation", TRAIN_DOCS),
    ):
        frozen = read(DATA / (filename + ".json"))
        lookup = {}
        for name, paragraphs in docs(names, sources):
            file = name.replace("/", "-") + ".rst"
            assert sources[file] == frozen["sources"][file]
            for n, text in enumerate(paragraphs):
                ids = tok.encode(text, add_special_tokens=False)[:129]
                if len(ids) >= 64:
                    lookup.setdefault(identity(ids), []).append((name, n))
        for row in frozen["rows"]:
            if row["category"] != category:
                continue
            hits = lookup[identity(row["ids"])]
            authors = {label for label, _ in hits}
            if len(authors) != 1:
                raise ValueError(f"Ambiguous provenance for {row['id']}: {hits}")
            label = authors.pop()
            rows[row["id"]] = {
                "archive_label": row["source"],
                "source": label,
                "paragraphs": [i for _, i in hits],
                "split": filename,
            }
        select[filename] = digest(DATA / (filename + ".json"))
    assert len(rows) == 544
    return {"frozen_datasets": select, "publisher_files": sources, "rows": rows}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    verified = resolve()
    if args.verify:
        assert read(DEST) == verified
    else:
        save(DEST, verified)
    print(len(verified["rows"]), "publisher-matched excerpts")
