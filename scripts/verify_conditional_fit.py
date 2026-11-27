"""Cold-load the actual fitted decoder and independently normalize every selected human response."""

from pathlib import Path

import mlx.core as mx

from loopqwen.conditional_reentry import ConditionalAssistant
from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.research.conditional_features import changed_token_spans
from loopqwen.research.conditional_fit import NAME
from loopqwen.research.native_function_data import token_pair
from scripts.verify_conditional_features import physical

OUT = Path("docs/measurements/conditional-trained-native-replay.json")


def token_costs(net, state, labels):
    values = []
    for hidden in (state[:, 0], net.readout(state)):
        logits = net.project(net.text.model.norm(hidden)).astype(mx.float32)
        selected = mx.take_along_axis(logits, labels[:, None], axis=1)[:, 0]
        values.append(mx.logsumexp(logits, axis=1) - selected)
    return values


def program_margins(net, tok, source, spec):
    parent = Path("research/results") / spec["feature_stage"]
    index = unpack(parent / "feature-index.json.gz")
    rows = []
    for rec in index:
        if not rec["negative"]:
            continue
        example = source[rec["id"]]
        p, y = token_pair(tok, example)
        states = physical(net, p + y[:-1])[len(p) - 1 :]
        chosen = token_costs(net, states, mx.array(y))
        mx.eval(*chosen)
        for alternative, other in zip(
            example["rejected"], rec["negative"], strict=True
        ):
            prefix, z = token_pair(tok, {**example, "target": alternative["target"]})
            assert p == prefix
            good, bad = changed_token_spans(y, z)
            assert (good, bad) == (
                other["positive_positions"],
                other["negative_positions"],
            )
            indices = mx.array(bad)
            negative_states = physical(net, p + z[:-1])[len(p) - 1 + indices]
            rejected = token_costs(net, negative_states, mx.array(z)[indices])
            row = {
                "id": rec["id"],
                "edit": other["edit"],
                "chosen_positions": good,
                "rejected_positions": bad,
            }
            for label, pos, neg in zip(
                ("ordinary", "trained"), chosen, rejected, strict=True
            ):
                mean_chosen = mx.mean(pos[mx.array(good)]).item()
                mean_rejected = mx.mean(neg).item()
                row[label] = {
                    "chosen_nll": mean_chosen,
                    "rejected_nll": mean_rejected,
                    "margin": mean_rejected - mean_chosen,
                }
            rows.append(row)
    assert len(rows) == 176
    return rows


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
        nll = [mx.mean(v).item() for v in token_costs(net, state, mx.array(y))]
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
        "training_sql_span_margins": program_margins(net, actor.tokenizer, source, s),
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
