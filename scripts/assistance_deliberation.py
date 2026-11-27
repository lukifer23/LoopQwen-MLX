"""Count reasoning-channel exposure and observed text on the fixed common requests."""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, save
from loopqwen.research.cross_view_study import tokenizer

R = Path("research/results")
P = Path("paper")
COMMON = Path("paper/followup-retention.csv")
SPEC = Path("research/experiments/integrated-assistance-01.json")
STUDIES = (
    "valid-return-01",
    "linked-rollin-02",
    "followup-protected-01",
    "visit-residual-01",
)
WORDS = re.compile(r"\b\w+\b")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build():
    with COMMON.open(newline="") as f:
        ids = [r["id"] for r in csv.DictReader(f)]
    assert len(set(ids)) == len(ids) == 36
    spec = read(SPEC)
    tok = tokenizer(spec)
    closing = tok.convert_tokens_to_ids("</think>")
    training = defaultdict(Counter)
    for w in read(spec["training"])["examples"]:
        c = training[w["kind"] if w["group"] == "database" else w["group"]]
        words = WORDS.findall(w["target"].get("reasoning_content", ""))
        c["source_turns"] += 1
        c["with_reasoning_words"] += bool(words)
        c["reasoning_words"] += len(words)
    records, summary = [], []
    base_tokens = {}
    first_openings = {}
    first_base = {}
    for name in STUDIES:
        cap = read(R / name / "registration.json")["max_tokens"]
        for arm in ("candidate", "control"):
            by_id = {r["id"]: r for r in unpack(R / name / (arm + ".json.gz"))}
            result = Counter()
            words = []
            for key in ids:
                q = by_id[key]["requests"][0]
                turns = q["trajectory"]["turns"]
                signatures = [(t["prompt_ids"], t["token_ids"]) for t in turns]
                opening = signatures[0][0]
                assert opening == first_openings.setdefault(key, opening)
                if arm == "control":
                    assert signatures == base_tokens.setdefault((key, cap), signatures)
                    result["identical_to_first_base"] += (
                        signatures == first_base.setdefault(key, signatures)
                    )
                completed = q["score"]["completed"]
                solved = q["score"]["success"]
                case_words = 0
                for n, t in enumerate(turns):
                    token_ids = t["token_ids"]
                    closed = closing in token_ids
                    fragment = (
                        token_ids[: token_ids.index(closing)] if closed else token_ids
                    )
                    count = len(WORDS.findall(tok.decode(fragment)))
                    case_words += count
                    records.append(
                        {
                            "study": name,
                            "arm": arm,
                            "id": key,
                            "turn": n,
                            "closed": closed,
                            "reasoning_tokens": len(fragment),
                            "reasoning_words": count,
                            "emitted_tokens": len(token_ids),
                            "answer_completed": completed,
                            "grounded": solved,
                        }
                    )
                    result["turns"] += 1
                    result["closed"] += closed
                    result["closed_empty"] += closed and count == 0
                    result["unclosed"] += not closed
                result["requests_with_words"] += case_words > 0
                result["solved"] += solved
                result["completed"] += completed
                words.append(case_words)
            summary.append(
                dict(
                    study=name,
                    arm=arm,
                    requests=len(ids),
                    budget=cap,
                    median_words=median(words),
                    total_words=sum(words),
                    identical_to_first_base=result.pop("identical_to_first_base", ""),
                    **result,
                )
            )
    write_csv(P / "common-reasoning-turns.csv", records)
    write_csv(P / "common-reasoning-summary.csv", summary)
    save(
        P / "assistance-deliberation.json",
        {
            "training": {k: dict(c) for k, c in sorted(training.items())},
            "common": summary,
            "interpretation": "Words before the first closing marker are observable reasoning-channel output. Empty text, verbosity and missing closure do not establish the amount or quality of latent reasoning. All opening prompts match across studies. Full Base trajectories are identical within each token-budget pair. The 384- and 512-token comparisons can have different outputs.",
        },
    )


if __name__ == "__main__":
    build()
