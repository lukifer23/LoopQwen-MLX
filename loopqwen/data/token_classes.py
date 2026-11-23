"""Response positions aligned to the target mask, including EOS."""

import re


def token_classes(row, tok):
    if row.get("protocol") == "answer":
        return ["final"] * (len(row["ids"]) - row["prefix"] - 1) + ["eos"]
    if row.get("protocol") == "language":
        return ["language"] * (len(row["ids"]) - row["prefix"] - 1) + ["eos"]
    response = " " + row["trace"]
    encoded = tok._tokenizer(
        response, add_special_tokens=False, return_offsets_mapping=True
    )
    if encoded["input_ids"] + [tok.eos_token_id] != row["ids"][row["prefix"] :]:
        raise ValueError("Teacher-forced response tokenization changed")
    boundary = response.rindex("Final answer:") + len("Final answer:")
    result = []
    for begin, end in encoded["offset_mapping"]:
        chars = response[begin:end]
        if end > boundary and response[max(begin, boundary) : end].strip():
            group = "final"
        elif begin < boundary and re.search(r"[0-9+*/%\-]", chars):
            group = "intermediate"
        else:
            group = "format"
        result.append(group)
    return result + ["eos"]
