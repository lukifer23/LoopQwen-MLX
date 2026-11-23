"""Prespecified OOD output-format transfer on the seed-43 paired models."""

import re
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from ..checkpoint import load_trained
from ..data.build_data import digest
from ..data.v2 import ROOT, rows
from ..io import save
from .phase2 import nll, predictions

FORMS = [
    (
        "decimal",
        "Write the answer with exactly three digits after the decimal point.",
        r"[+-]?\d+\.\d{3}",
    ),
    (
        "fraction",
        "Write the answer as a reduced fraction, including /1 for an integer.",
        r"[+-]?\d+/1",
    ),
    (
        "scientific",
        "Write the answer in scientific notation using e and a signed exponent.",
        r"[+-]?\d(?:\.\d+)?[eE][+-]\d+",
    ),
]


def run():
    result = {"ood_sha256": digest(ROOT / "ood.jsonl"), "runs": {}}
    for arm in ("p2_confirm_43_dense", "p2_confirm_43_loop"):
        model, tok, _ = load_trained(Path("checkpoints") / arm)
        docs = []
        for i, r in enumerate(rows("ood", "answer")[:45]):
            num = Fraction(r["answer"])
            if num.denominator != 1:
                raise ValueError("Registered format suite expects integer golds")
            form, message, regex = FORMS[i % 3]
            val = num.numerator
            gold = {
                "decimal": f"{val}.000",
                "fraction": f"{val}/1",
                "scientific": f"{Decimal(val):E}",
            }[form]
            prompt = r["question"] + "\n" + message + "\nAnswer:"
            prefix = tok.encode(prompt, add_special_tokens=False)
            ids = (
                prefix
                + tok.encode(" " + gold, add_special_tokens=False)
                + [tok.eos_token_id]
            )
            docs.append(
                r
                | dict(
                    prompt=prompt,
                    answer=gold,
                    prefix=len(prefix),
                    ids=ids,
                    input_length=len(ids) - 1,
                    format=form,
                )
            )
        generated = predictions(model, tok, docs, 20, padded=True)
        for i, record in enumerate(generated["records"]):
            form, _, regex = FORMS[i % 3]
            record.update(
                format=form,
                format_compliant=bool(re.fullmatch(regex, record["text"].strip())),
            )
        result["runs"][arm] = dict(loss=nll(model, docs), generations=generated)
    save("outputs/phase2/format_shift.json", result)


if __name__ == "__main__":
    run()
