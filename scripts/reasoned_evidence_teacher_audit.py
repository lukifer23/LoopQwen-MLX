"""Reconstruct every training-only teacher prompt, emitted verdict, citation and stop."""

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.reasoned_evidence import NAME
from loopqwen.research.reasoned_evidence_fit import parse_teacher


def run(root):
    s = read(root / "registration.json")
    tok = tokenizer(s)
    pool = read(s["training"])["teacher_pool"]
    generated = unpack(root / "teacher-generations.json.gz")
    report = read(root / "teacher-fit.json")
    reasons = Counter()
    per = defaultdict(Counter)
    total = 0
    parsed_count = 0
    ranked = 0
    records = []
    for w, r in zip(pool, generated, strict=True):
        wire = tok.decode(r["prompt"])
        assert w["id"] == r["id"] and w["evidence"] in wire and w["claim"] in wire
        assert (
            wire.startswith("<|im_start|>user\n")
            and tok.encode(wire, add_special_tokens=False) == r["prompt"]
        )
        assert r["text"] == tok.decode(r["generated"])
        parsed = parse_teacher(r["text"], len(w["evidence"].splitlines()))
        assert parsed == r["parsed"]
        clean = r["text"].strip().replace("**", "")
        found = re.search(
            r"(?:^|\n)\s*(?:final\s+)?verdict\s*:\s*(true|false|uncertain)\s*[.!]?\s*$",
            clean,
            re.IGNORECASE,
        )
        label = found.group(1).lower() if found else None
        proof = clean[: found.start()].strip() if found else clean
        n = len(
            re.sub(
                r"^\s*(?:explanation|reasoning|reason)\s*:\s*",
                "",
                proof,
                flags=re.IGNORECASE,
            ).split()
        )
        accepted = bool(
            len(r["generated"]) < s["teacher_cap"]
            and parsed
            and parsed["label"] == w["label"]
        )
        assert accepted == r["accepted"]
        if len(r["generated"]) == s["teacher_cap"]:
            status = "Reached output cap"
        elif label is None:
            status = "No final verdict"
        elif not 5 <= n <= 100:
            status = "Explanation length"
        elif not parsed:
            status = "Missing or invalid premise citation"
        elif label != w["label"]:
            status = "Different verdict"
        else:
            status = "Accepted"
        assert (status == "Accepted") == accepted
        total += len(r["generated"])
        parsed_count += bool(label)
        ranked += label == w["label"]
        reasons[status] += 1
        per[w["label"]][status] += 1
        records.append(
            {
                "id": w["id"],
                "human": w["label"],
                "teacher": label,
                "status": status,
                "tokens": len(r["generated"]),
                "words_before_verdict": n,
                "evidence": w["evidence"],
                "claim": w["claim"],
                "output": r["text"],
            }
        )
    with Path("paper/reasoned-evidence-teacher.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    assert len(records) == report["checked"] and reasons["Accepted"] == sum(
        report["accepted"].values()
    )
    out = {
        "training_sha256": digest(Path(s["training"])),
        "generations_sha256": digest(root / "teacher-generations.json.gz"),
        "n": len(records),
        "tokens": total,
        "terminal_verdict": parsed_count,
        "terminal_correct": ranked,
        "stops": dict(reasons),
        "by_label": dict(per),
        "seconds": report["seconds"],
        "peak_bytes": report["peak"],
    }
    dest = Path("docs/measurements/reasoned-evidence-teacher.json")
    if dest.exists():
        assert read(dest) == out
    else:
        save(dest, out)
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", type=Path, default=Path("research/results") / NAME)
    run(p.parse_args().study_dir)
