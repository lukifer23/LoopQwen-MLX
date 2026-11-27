"""Measure where a fixed greedy generation spent its tokens across the held comparison."""

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read
from loopqwen.research.cross_view_study import tokenizer
from scripts.assistance_deliberation import WORDS, write_csv


def account(root, streams):
    spec = read(root / "registration.json")
    tok = tokenizer(spec)
    opening = tok.convert_tokens_to_ids("<think>")
    closing = tok.convert_tokens_to_ids("</think>")
    training = defaultdict(Counter)
    for row in unpack(Path(spec["training"])):
        group = training[row["group"]]
        group["turns"] += 1
        group["with_reasoning"] += bool(
            WORDS.search(row["target"].get("reasoning_content", ""))
        )
    records = []
    result = {}
    for arm, stream in zip(("candidate", "control"), streams, strict=True):
        groups = defaultdict(list)
        for r in stream:
            (q,) = r["requests"]
            counts = Counter()
            for t in q["trajectory"]["turns"]:
                ids = t["token_ids"]
                # The system may quote earlier turns. Check the actual opening suffix.
                assert opening in t["prompt_ids"][-4:]
                cut = ids.index(closing) if closing in ids else len(ids)
                counts["turns"] += 1
                counts["closed"] += cut < len(ids)
                counts["unclosed"] += cut == len(ids)
                counts["reopened"] += opening in ids[cut + 1 :]
                counts["empty_closed"] += cut < len(ids) and not WORDS.search(
                    tok.decode(ids[:cut])
                )
                counts["reasoning_words"] += len(WORDS.findall(tok.decode(ids[:cut])))
                counts["before_close_tokens"] += cut
                counts["closing_tokens"] += int(cut < len(ids))
                counts["after_close_tokens"] += max(len(ids) - cut - 1, 0)
                counts["emitted"] += len(ids)
            assert (
                counts["before_close_tokens"]
                + counts["closing_tokens"]
                + counts["after_close_tokens"]
                == counts["emitted"]
            )
            record = dict(id=r["id"], family=r["family"], arm=arm, **counts)
            records.append(record)
            groups["all"].append(record)
            groups[r["family"]].append(record)
            if r["family"] in ("effects", "human_first", "human_followup"):
                groups["reasoning"].append(record)
        result[arm] = {}
        for family, rows in groups.items():
            result[arm][family] = dict(
                requests=len(rows),
                median_words=median(r["reasoning_words"] for r in rows),
                requests_with_words=sum(r["reasoning_words"] > 0 for r in rows),
                **{k: sum(r[k] for r in rows) for k in counts},
            )
    write_csv(Path("paper/conditional-primary-generation-channels.csv"), records)
    return dict(training={k: dict(v) for k, v in training.items()}, streams=result)
