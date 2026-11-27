"""Enroll unmodified human reasoning pairs that need opposite answers under the same fact."""

import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.program_language_material import norm, shingles
from scripts.rule_dialogue_material import OUT_OF_SCOPE

REV = "28c1dbb56caf81799296cb17892fa73402e23464"
URL = f"https://huggingface.co/datasets/allenai/quartz/resolve/{REV}/data/train-00000-of-00001.parquet"
SHA = "edeb1fa716d945ee198c9f5ce5ab6cad4d847e714ff7463834dee422c8450035"
CACHE = Path("data/research_external/quartz/original-train.parquet")
OUT = Path("research/sources/qualitative-pairs")
PANEL = Path("research/data/conditional-program-panel.json")


def build():
    if not CACHE.exists():
        CACHE.parent.mkdir(exist_ok=True, parents=True)
        with urllib.request.urlopen(URL, timeout=60) as stream:
            data = stream.read()
        tmp = CACHE.with_suffix(".partial")
        tmp.write_bytes(data)
        assert digest(tmp) == SHA
        tmp.replace(CACHE)
    assert digest(CACHE) == SHA
    held = read(PANEL)["rows"]
    held_words = set()
    held_text = set()
    for r in held:
        for q in r["questions"]:
            held_words.update(shingles(q, 8))
            held_text.add(norm(q))
    obj = pq.read_table(CACHE).to_pylist()
    counts = Counter()
    groups = defaultdict(list)
    for r in obj:
        opts = dict(zip(r["choices"]["label"], r["choices"]["text"], strict=True))
        assert set(opts) == {"A", "B"} and r["answerKey"] in opts
        fields = [r["para"], r["question"], *opts.values()]
        if OUT_OF_SCOPE.search(" ".join(fields)):
            counts["outside domain"] += 1
            continue
        if any(shingles(x, 8) & held_words or norm(x) in held_text for x in fields):
            counts["shared eight-word passage or entire field with held question"] += 1
            continue
        texts = tuple(sorted(norm(x) for x in opts.values()))
        if len(set(texts)) != 2:
            counts["identical normalized options"] += 1
            continue
        groups[r["para_id"], norm(r["para"]), texts].append(
            {
                "id": r["id"],
                "background": r["para"],
                "question": r["question"],
                "choices": list(opts.values()),
                "correct": opts[r["answerKey"]],
                "source_key": r["answerKey"],
            }
        )
    pairs = []
    n_eligible = sum(map(len, groups.values()))
    for (pid, _, choices), group in sorted(groups.items()):
        # The authors identify their perturbed twin in the original training IDs.
        by_id = {m["id"]: m for m in group}
        twinned = []
        for ka, a in by_id.items():
            kb = ka + "-flip"
            if kb not in by_id:
                continue
            b = by_id[kb]
            if norm(a["correct"]) == norm(b["correct"]) or norm(a["question"]) == norm(
                b["question"]
            ):
                continue
            x, y = (set(norm(r["question"]).split()) for r in (a, b))
            twinned.append((len(x & y) / len(x | y), ka, kb))
        for similarity, ka, kb in sorted(twinned, key=lambda t: (-t[0], t[1], t[2])):
            pairs.append(
                {
                    "id": fingerprint([ka, kb]),
                    "source_background": pid,
                    "choices": choices,
                    "question_word_jaccard": similarity,
                    "members": [by_id[ka], by_id[kb]],
                }
            )
    ids = [m["id"] for pair in pairs for m in pair["members"]]
    assert len(ids) == len(set(ids))
    OUT.mkdir(exist_ok=True, parents=True)
    path = OUT / "training.json.gz"
    if path.exists():
        assert unpack(path) == pairs
    else:
        packed(path, pairs)
    summary = {
        "upstream": URL,
        "train_sha256": SHA,
        "held_panel": digest(PANEL),
        "source_rows": len(obj),
        "source_backgrounds": len({r["para_id"] for r in obj}),
        "exclusions": dict(counts),
        "eligible_rows": n_eligible,
        "unpaired_eligible_rows": n_eligible - len(ids),
        "pairs": len(pairs),
        "questions": len(ids),
        "used_backgrounds": len({r["source_background"] for r in pairs}),
        "distinct_choice_sets": len({tuple(r["choices"]) for r in pairs}),
        "pairs_question_word_jaccard_at_least_half": sum(
            r["question_word_jaccard"] >= 0.5 for r in pairs
        ),
        "source_answer_positions": dict(
            Counter(m["source_key"] for w in pairs for m in w["members"])
        ),
        "training_sha256": digest(path),
    }
    assert len(obj) == n_eligible + sum(counts.values())
    save(OUT / "qualification.json", summary)
    print(summary)


if __name__ == "__main__":
    build()
