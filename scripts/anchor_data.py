"""Freeze publisher item text and IDs without adding dataset dependencies to training."""

from pathlib import Path
import hashlib, json
from loopqwen.io import read, save, digest
from loopqwen.identity import fingerprint


def main():
    s = read("research/experiments/external-anchor-01.json")
    out = Path("research/data/external-anchor.json")
    root = Path("data/research_external")
    g = root / "gsm-train.jsonl"
    p = root / "arc-easy-validation.parquet"
    if (
        digest(g) != s["sources"]["gsm"]["sha256"]
        or digest(p) != s["sources"]["arc"]["sha256"]
    ):
        raise ValueError("Publisher identity mismatch")
    # Parquet conversion is deliberately an offline build dependency.
    import pyarrow.parquet as pq

    arc = pq.read_table(p).to_pylist()
    train = []
    for source in ("pilot", "v2", "v3", "v4"):
        file = Path("data") / source / "train.jsonl"
        if source == "v3" and not file.exists():
            continue  # V3 uses the immutable V2 training split
        for line in file.open():
            r = json.loads(line)
            train.append(r.get("question", r.get("prompt", "")))
    train = "\n".join(train)
    gsm = []
    for i, line in enumerate(g.open()):
        r = json.loads(line)
        q = r["question"].strip()
        if "<<" not in r["answer"] and q not in train:
            gsm.append(
                dict(
                    id=fingerprint(q),
                    source_row=i,
                    question=q,
                    gold=r["answer"].rsplit("####", 1)[-1].strip().replace(",", ""),
                )
            )
    old = {
        r["id"] for r in sorted(gsm, key=lambda x: fingerprint([71109, x["id"]]))[:12]
    }

    def choose(rows, n):
        return sorted(rows, key=lambda x: fingerprint([s["seed"], x["id"]]))[:n]

    gsm = choose([r for r in gsm if r["id"] not in old], s["gsm_items"])
    choices = []
    for r in arc:
        q = r["question"]
        c = r["choices"]
        if len(c["text"]) == 4 and all(c["text"]) and q.strip() not in train:
            choices.append(
                dict(
                    id=r["id"],
                    question=q,
                    choices=c["text"],
                    gold=c["label"].index(r["answerKey"]),
                )
            )
    data = dict(gsm=gsm, arc=choose(choices, s["arc_items"]), sources=s["sources"])
    if len(gsm) != 32 or len(data["arc"]) != 32:
        raise ValueError("Insufficient independent items")
    if out.exists() and read(out) != data:
        raise ValueError("Reserved items differ")
    save(out, data)


if __name__ == "__main__":
    main()
