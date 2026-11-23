"""Projection FLOP proxy from genuine checkpoint tensor shapes and actual padded work."""

from math import prod
from pathlib import Path
import re
from safetensors import safe_open
from loopqwen.model_loader import local_snapshot
from loopqwen.recurrence import Span
from loopqwen.checkpoint import location
from loopqwen.io import read, save


def dimensions():
    layers = [[] for _ in range(24)]
    head = None
    for shard in local_snapshot().glob("*.safetensors"):
        with safe_open(shard, framework="numpy") as f:
            for key in f.keys():
                if key == "model.language_model.embed_tokens.weight":
                    head = f.get_slice(key).get_shape()
                match = re.fullmatch(
                    r"model\.language_model\.layers\.(\d+)\..+\.([^.]+)\.weight", key
                )
                if match:
                    dims = f.get_slice(key).get_shape()
                    if len(dims) == 2:
                        layers[int(match[1])].append((match[2], dims))
    if head != [248320, 1024] or any(not l for l in layers):
        raise ValueError("Unexpected checkpoint")
    return layers, head


def measure(state, layers, head):
    c = state["config"]
    s = c["start"]
    end = s + c["length"]
    target = {
        "block": range(s, end),
        "nearby": range(max(0, s - 2), min(end + 2, 24)),
        "last8": range(16, 24),
        "all": range(24),
    }[c["target"]]
    per_layer = []
    for i, l in enumerate(layers):
        backbone = sum(prod(shape) for _, shape in l)
        adapter = sum(
            c["rank"] * sum(shape)
            for name, shape in l
            if i in target and (c["projections"] is None or name in c["projections"])
        )
        per_layer.append(2 * (backbone + adapter))
    total = 0
    for row in state["history"]:
        if "microbatches" in row:
            batches = [
                (m["k"], prod(m["shape"]), m["scored"]) for m in row["microbatches"]
            ]
        elif len(row["depths"]) == 1:
            batches = [
                (
                    row["depths"][0],
                    round(row["inputs"] / (1 - row["padding_fraction"])),
                    row["labels"],
                )
            ]
        else:
            raise ValueError("Historical run lacks per-microbatch shape accounting")
        for depth, padded, labels in batches:
            sites = Span(start=s, length=c["length"], passes=depth).sites(24)
            total += padded * sum(per_layer[i] for i, _ in sites) + 2 * labels * prod(
                head
            )
    util = [
        r["device_utilization_percent"]
        for r in state["history"]
        if r.get("device_utilization_percent") is not None
    ]
    return dict(
        forward_projection_flops=total,
        update_seconds=sum(r["seconds"] for r in state["history"]),
        device_utilization_samples=util,
        targets=state["loss_tokens"],
        inputs=state["input_tokens"],
    )


if __name__ == "__main__":
    layers, head = dimensions()
    records = {}
    for group in ("phase2_direct", "phase2_screen", "phase2_confirm"):
        for run in read(Path("experiments") / (group + ".json"))["runs"]:
            records[run] = measure(
                read(location(Path("checkpoints") / run) / "state.json"), layers, head
            )
    save("outputs/phase2/compute.json", records)
