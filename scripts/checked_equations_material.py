"""Reserve fresh publisher arithmetic for the checked-equality tool on both native models."""

from collections import Counter
from pathlib import Path
from mlx_lm.utils import load_tokenizer
from operand_material import external
from program_reasoning_material import former, norm, sha
from loopqwen.calculator_tool import tool
from loopqwen.io import read, save
from loopqwen.evaluation.program_questions import INSTRUCTION
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import initial_messages, prompt

PRIOR = "36be03d"


def main():
    tok = load_tokenizer(local_snapshot())
    rows = external()
    used = {tuple(norm(q)[:8]) for q in former({q for _, q, _ in rows}, PRIOR)}
    catalog = [tool(check_equalities=True)]
    panel = []
    counts = Counter()
    for family, q, gold in sorted(
        rows, key=lambda r: sha("checked-arithmetic-field-671:" + r[1])
    ):
        key = tuple(norm(q)[:8])
        if gold is None or key in used or counts[family] >= 12:
            continue
        used.add(key)
        counts[family] += 1
        user = q + INSTRUCTION
        ids = tok.encode(
            prompt(tok, initial_messages(user, catalog)), add_special_tokens=False
        )
        panel.append(
            dict(id=sha(q), family=family, questions=[user], initial_ids=ids, gold=gold)
        )
    assert sorted(counts.values()) == [12, 12]
    dest = Path("research/data/checked-equations-field.json")
    result = dict(excluded_through=PRIOR, rows=panel)
    if dest.exists():
        assert read(dest) == result
    else:
        save(dest, result)


if __name__ == "__main__":
    main()
