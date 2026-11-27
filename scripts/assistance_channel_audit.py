"""Reconstruct visible thinking delimiters from original training targets and served answers."""

from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair

ROOT = Path("research/results/integrated-assistance-01")


def audit():
    s = read(ROOT / "registration.json")
    tok = tokenizer(s)
    close = tok.convert_tokens_to_ids("</think>")
    source = Counter()
    for w in read(s["training"])["examples"]:
        _, y = token_pair(tok, w)
        source[y.count(close)] += 1
    arms = {}
    for arm in ("candidate", "control"):
        groups = defaultdict(Counter)
        offenders = []
        for row in unpack(ROOT / (arm + ".json.gz")):
            (q,) = row["requests"]
            tr = q["trajectory"]
            answer = tr["answer"] or ""
            n = answer.count("</think>")
            if n:
                offenders.append(row["id"])
                # The extra delimiter came from the decoder's generated token stream.
                raw = tr["turns"][-1]["token_ids"]
                assert raw.count(close) >= n + 1
            for g in (row["family"], "all"):
                c = groups[g]
                c["requests"] += 1
                c["nonempty"] += bool(answer.strip())
                c["visible_extra_closures"] += n
                c["requests_with_extra_closure"] += bool(n)
        arms[arm] = {"families": groups, "affected_requests": offenders}
    receipt = {
        "weights": digest(ROOT / "cross-view.safetensors"),
        "archive": digest(ROOT / "checksums.json"),
        "training_targets_by_number_of_close_tokens": {
            str(k): v for k, v in source.items()
        },
        "arms": arms,
    }
    dest = Path("docs/measurements/integrated-assistance-channel-audit.json")
    if dest.exists():
        assert read(dest) == receipt
    else:
        save(dest, receipt)
    print(
        receipt["training_targets_by_number_of_close_tokens"],
        {k: v["families"]["open"] for k, v in arms.items()},
    )


if __name__ == "__main__":
    audit()
