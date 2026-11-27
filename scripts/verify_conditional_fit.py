"""Cold-load the actual fitted decoder and independently normalize every selected human response."""

from pathlib import Path

import mlx.core as mx

from loopqwen.conditional_reentry import ConditionalAssistant
from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.conditional_fit import NAME
from loopqwen.research.native_function_data import token_pair
from scripts.verify_conditional_features import physical

OUT = Path("docs/measurements/conditional-trained-native-replay.json")


def run():
    root = Path("research/results") / NAME
    s = read(root / "registration.json")
    actor = ConditionalAssistant(s["checkpoint"], root)
    net = actor._tool_model
    source = {r["id"]: r for r in unpack(Path(s["training"]))}
    probes = read(root / "training-probes.json")
    records = []
    for q in probes:
        p, y = token_pair(actor.tokenizer, source[q["id"]])
        state = physical(net, p + y[:-1])[len(p) - 1 :]
        views = [state[:, 0], net.readout(state)]
        nll = []
        for h in views:
            logits = net.project(net.text.model.norm(h)).astype(mx.float32)
            gold = mx.take_along_axis(logits, mx.array(y)[:, None], axis=1)[:, 0]
            nll.append(mx.mean(mx.logsumexp(logits, axis=1) - gold).item())
        rec = {
            "id": q["id"],
            "group": q["group"],
            "positions": len(y),
            "ordinary": nll[0],
            "trained": nll[1],
            "ordinary_abs_error": abs(nll[0] - q["ordinary"]),
            "trained_abs_error": abs(nll[1] - q["trained"]),
        }
        records.append(rec)
    max_error = max(
        max(r["ordinary_abs_error"], r["trained_abs_error"]) for r in records
    )
    result = {
        "weights": digest(root / "readout.safetensors"),
        "archive": digest(root / "checksums.json"),
        "turns": len(records),
        "positions": sum(r["positions"] for r in records),
        "maximum_turn_mean_abs_error": max_error,
        "records": records,
    }
    if OUT.exists():
        assert read(OUT) == result
    else:
        save(OUT, result)
    print(len(records), result["positions"], max_error, flush=True)
    if max_error > 0.0001:
        raise ValueError("Cold normalized native probabilities require investigation")


if __name__ == "__main__":
    run()
