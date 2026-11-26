"""Enroll real executed fitting programs and fresh human arithmetic before generating any drafts."""

from collections import Counter
from fractions import Fraction
from pathlib import Path

from mlx_lm.utils import load_tokenizer
from operand_material import external, grams
from program_reasoning_material import former, norm, sha

from loopqwen.arithmetic_actions import as_steps, mentions
from loopqwen.calculator_tool import execute, tool
from loopqwen.evaluation.program_questions import INSTRUCTION
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import prompt, system_message

DATA = Path("research/data")
PRIOR = "f2bc54781f1e46b5799ce21be629f8587fa0b490"


def main():
    ext = external()
    excluded = former({q for _, q, _ in ext}, prior=PRIOR)
    blocked = {tuple(norm(q)[:8]) for q in excluded}
    tok = load_tokenizer(local_snapshot())
    panel = []
    counts = Counter()
    for src, q, gold in sorted(
        ext, key=lambda r: sha("draft-program-field-3281:" + r[1])
    ):
        opening = tuple(norm(q)[:8])
        if gold is None or opening in blocked or counts[src] >= 24:
            continue
        user = q + INSTRUCTION
        ids = tok.encode(
            prompt(tok, [system_message([tool()]), {"role": "user", "content": user}]),
            add_special_tokens=False,
        )
        panel.append(
            {
                "id": sha(q),
                "source": src,
                "question": q,
                "user_request": user,
                "gold": gold,
                "prompt_ids": ids,
                "sha256": fingerprint(ids),
            }
        )
        blocked.add(opening)
        counts[src] += 1
    assert sorted(counts.values()) == [24, 24]
    banned = [grams(r["question"]) for r in panel]
    origin = DATA / "operand-planner-training.json"
    pool = read(origin)["rows"]
    fit = []
    counts = Counter()
    worst = 0.0
    for r in sorted(pool, key=lambda v: sha("native-draft-training-3298:" + v["id"])):
        q = r["question"]
        if counts[r["source"]] >= 256:
            continue
        ngram = grams(q)
        nearest = max(
            (len(ngram & g) / len(ngram | g) for g in banned if ngram | g), default=0.0
        )
        if nearest >= 0.4:
            continue
        ops = as_steps([v for _, _, v in mentions(q)], r["actions"])
        assert execute(ops)[-1] == Fraction(r["gold"])
        fit.append(
            {
                "id": r["id"],
                "source": r["source"],
                "question": q,
                "steps": ops,
                "gold": r["gold"],
            }
        )
        counts[r["source"]] += 1
        worst = max(worst, nearest)
    assert sorted(counts.values()) == [256, 256]
    save(
        DATA / "draft-program-training.json",
        {
            "source": str(origin),
            "source_sha256": digest(origin),
            "max_masked_fourgram_jaccard": worst,
            "rows": fit,
        },
    )
    save(
        DATA / "draft-program-panel.json",
        {"excluded_through": PRIOR, "instruction": INSTRUCTION, "rows": panel},
    )
    print(len(fit), len(panel), worst)


if __name__ == "__main__":
    main()
