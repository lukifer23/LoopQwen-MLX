"""Reserve unused human questions and freeze verified GSM8K program targets before fitting."""

from collections import Counter
from pathlib import Path

from mlx_lm.utils import load_tokenizer
from operand_material import external, grams
from program_reasoning_material import former, norm, sha

from loopqwen.calculator_tool import tool
from loopqwen.evaluation.program_questions import INSTRUCTION
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import initial_messages, prompt

DATA = Path("research/data")
PRIOR = "977d22b"


def main():
    rows = external()
    old = former({q for _, q, _ in rows}, prior=PRIOR)
    blocked = {tuple(norm(q)[:8]) for q in old}
    tok = load_tokenizer(local_snapshot())
    fresh = []
    counts = Counter()
    for source, question, gold in sorted(
        rows, key=lambda r: sha("callback-field-57:" + r[1])
    ):
        opening = tuple(norm(question)[:8])
        if gold is None or counts[source] >= 24 or opening in blocked:
            continue
        user = question + INSTRUCTION
        ids = tok.encode(
            prompt(tok, initial_messages(user, [tool()])), add_special_tokens=False
        )
        fresh.append(
            {
                "id": sha(question),
                "family": "arithmetic",
                "cohort": source,
                "question": question,
                "user_request": user,
                "gold": gold,
                "initial_ids": ids,
            }
        )
        blocked.add(opening)
        counts[source] += 1
    assert sorted(counts.values()) == [24, 24]
    past = read(DATA / "unified-session-panel.json")["rows"]
    retention = [
        dict(w, cohort="retention")
        for w in past
        if w["family"] not in ("math", "judgment", "user_extraction")
    ]
    combined = sorted(
        [*retention, *fresh], key=lambda r: sha("callback-order-571:" + r["id"])
    )
    assert len({w["id"] for w in combined}) == len(combined) == 120
    original = DATA / "tool-program-reasoning-training.json"
    paired = read(original)["examples"][:1280]
    held = [grams(w["question"]) for w in fresh]
    pool = []
    for call, reply in zip(paired[::2], paired[1::2], strict=True):
        assert call["id"].removesuffix("-call") == reply["id"].removesuffix("-result")
        user = call["messages"][-1]["content"]
        gram = grams(user.removesuffix(INSTRUCTION))
        similarity = max(
            (len(gram & b) / len(gram | b) for b in held if gram | b), default=0
        )
        if similarity < 0.4:
            pool.append(
                {
                    "id": call["id"],
                    "user": user,
                    "target": call["visible"],
                    "gold": reply["visible"],
                    "overlap": similarity,
                }
            )
    pool.sort(key=lambda r: sha("callback-train-573:" + r["id"]))
    assert len(pool) >= 256
    save(
        DATA / "callback-recovery-training.json",
        {
            "rows": pool[:256],
            "source": str(original),
            "source_sha256": digest(original),
        },
    )
    save(
        DATA / "callback-recovery-panel.json",
        {"excluded_through": PRIOR, "rows": combined},
    )
    print(len(combined), len(fresh), len(pool[:256]), flush=True)


if __name__ == "__main__":
    main()
