"""Measure padding and eligibility over the same complete V2 epoch at each length cap."""

from collections import Counter
from loopqwen.config import config
from loopqwen.data.sampling import prepare
from loopqwen.io import save


def run():
    c = config("configs/phase2/p2_a_trace_loop.yaml")
    result = {}
    for mixture in ("answer_only", "verified_traces", "mixed_answer_and_traces"):
        data = {}
        for bucket in (True, False):
            docs, _, _, policy = prepare(c | dict(mixture=mixture, bucket=bucket))
            counts = []
            for i in range(0, len(docs), c["batch_size"]):
                group = docs[i : i + c["batch_size"]]
                counts.append(
                    max(16, max(r["input_length"] for r in group)) * len(group)
                )
            data[str(bucket)] = dict(
                positions=sum(counts),
                padding_fraction=1 - sum(r["input_length"] for r in docs) / sum(counts),
                caps={
                    s: dict(
                        Counter(r["category"] for r in docs if r["input_length"] > s)
                    )
                    for s in (64, 128, 256, 512)
                },
            )
        result[mixture] = data
    save("outputs/phase2/padding.json", result)


if __name__ == "__main__":
    run()
