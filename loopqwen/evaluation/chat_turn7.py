"""Score the first assistant turn, including a chat end emitted before the Base EOS."""

from ..evaluation.answers import numeric


def first_turn(record, end_id):
    ids = record["token_ids"]
    end = ids.index(end_id) if end_id in ids else len(ids)
    first = record["text"].split("<|im_end|>", 1)[0]
    if (end < len(ids)) != ("<|im_end|>" in record["text"]):
        raise ValueError("Token and decoded chat boundaries disagree")
    turns = ids[:end]
    closes = [i for i, v in enumerate(turns) if v == record["closing_id"]]
    if len(closes) != first.count("</think>"):
        raise ValueError("Thinking boundaries disagree")
    visible = (
        first.split("</think>", 1)[1].split("<|endoftext|>", 1)[0]
        if len(closes) == 1 and record["opening_id"] not in turns
        else ""
    )
    finished = end < len(ids) or record["stop"] == "eos"
    number = numeric(visible) if finished else None
    return dict(
        id=record["item"]["id"],
        end_position=end if end < len(ids) else None,
        visible=visible,
        stop="chat_end" if end < len(ids) else record["stop"],
        answer=number,
        correct=number is not None and number == numeric(record["item"]["gold"]),
    )


def audit():
    from ..training.exposure import OUT, S
    from ..model_loader import local_snapshot
    from ..io import digest, save, read
    from .mechanism import unpack
    from transformers import AutoTokenizer

    file = local_snapshot() / "tokenizer.json"
    tok = AutoTokenizer.from_pretrained(file.parent)
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    if end_id is None:
        raise ValueError("Missing chat end")
    root = OUT / "language"
    result = dict(tokenizer=digest(file), chat_end=end_id, protocols={}, selected=None)
    for name in read(root / "decision.json")["calibration"]:
        rows = []
        for p in sorted((root / name).glob("*.json.gz")):
            r = unpack(p)
            if tok.decode(r["token_ids"], skip_special_tokens=False) != r["text"]:
                raise ValueError("Unable to recover generated text")
            rows.append(first_turn(r, end_id))
        n = len(rows)
        valid = sum(r["answer"] is not None for r in rows)
        correct = sum(r["correct"] for r in rows)
        result["protocols"][name] = dict(
            n=n, valid=valid, correct=correct, records=rows
        )
        if (
            result["selected"] is None
            and n == S["calibration"]["items"]
            and valid >= 6
            and 0 < correct < n
        ):
            result["selected"] = name
    save(root / "first-turn.json", result)
    print(
        {k: (v["valid"], v["correct"]) for k, v in result["protocols"].items()},
        flush=True,
    )
    return result
