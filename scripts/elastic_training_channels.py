"""Count real thought and visible response tokens in the finished unassisted training probes."""

from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from loopqwen.research.cross_view_study import tokenizer
from scripts.assistance_deliberation import WORDS, write_csv
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

ROOT = Path("research/results/elastic-world-fit-01")


def run():
    committed(ROOT / "checksums.json")
    spec = read(ROOT / "registration.json")
    tok = tokenizer(spec)
    close = tok.convert_tokens_to_ids("</think>")
    stop = set(tok.eos_token_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}
    fields = defaultdict(Counter)
    for row in unpack(Path(spec["training"])):
        g = fields[row["group"]]
        g["turns"] += 1
        g["with_reasoning_words"] += bool(
            WORDS.search(row["target"].get("reasoning_content", ""))
        )
    raw = ROOT / "training-free-generation.json.gz"
    counts = defaultdict(Counter)
    records = []
    for r in unpack(raw):
        ids = r["tokens"]
        boundary = ids.index(close) if close in ids else len(ids)
        thought = ids[:boundary]
        answer = ids[boundary + 1 :]
        stop_id = bool(answer and answer[-1] in stop)
        visible = answer[:-1] if stop_id else answer
        words = len(WORDS.findall(tok.decode(thought)))
        g = counts[r["group"]]
        g["turns"] += 1
        g["with_thinking_words"] += bool(words)
        g["thought_tokens"] += len(thought)
        g["delivered_visible_words"] += len(WORDS.findall(r["visible"]))
        g["emitted"] += len(ids)
        g["completed"] += r["closed"]
        g["correct"] += r["score"]["correct"]
        assert len(ids) == len(thought) + (close in ids) + len(visible) + stop_id
        records.append(
            {
                "id": r["id"],
                "group": r["group"],
                "emitted": len(ids),
                "thought_tokens": len(thought),
                "thought_words": words,
                "close_token": close in ids,
                "after_close_excluding_stop": len(visible),
                "closed": r["closed"],
                "registered_correct": r["score"]["correct"],
            }
        )
    write_csv(Path("paper/elastic-world-training-channels.csv"), records)
    output = {
        "fit": digest(ROOT / "checksums.json"),
        "training": digest(Path(spec["training"])),
        "emissions": digest(raw),
        "training_fields": fields,
        "generation": counts,
        "scope": "The new contrastive corpus is counted once before optimizer resampling. For the free training probes, all emitted tokens before the first thinking close are counted, including whitespace. A word requires the same Unicode word matcher as the previous product study. Natural-language reasoning may also appear after that boundary. These counts measure channel use and do not determine correctness.",
    }
    persist(Path("docs/measurements/elastic-training-channels.json"), output)
    print(
        "qualitative supervision",
        dict(fields["qualitative"]),
        {k: dict(v) for k, v in counts.items()},
    )


if __name__ == "__main__":
    run()
