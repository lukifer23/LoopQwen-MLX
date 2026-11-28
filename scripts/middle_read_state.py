"""Measure actual fitted residual writes and their effect after the ordinary suffix."""

import argparse

import mlx.core as mx
from mlx.utils import tree_flatten

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.middle_read import LAYER, set_reads
from loopqwen.shared_query import attend, attention_update, memory
from loopqwen.tool_cli import open_assistant
from scripts.middle_depth_budget import worker
from scripts.middle_read_bundle import DEST, REPORT
from scripts.middle_read_material import BANK
from scripts.middle_read_profile import difference, recomputed


def rms(x):
    return mx.sqrt(mx.mean(x.astype(mx.float32) ** 2)).item()


def measure():
    with worker("profile", "middle-read/trained-state") as guard:
        a = open_assistant(argparse.Namespace(checkpoint=DEST, tool_format="qwen"))
        model = a._tool_model
        layer = model.text.layers[LAYER]
        bank = unpack(BANK)
        rows = []
        for group in sorted({r["group"] for r in bank}):
            # First in the fixed optimization order, evaluated on prompt alone.
            r = next(row for row in bank if row["group"] == group)
            ids = mx.array([r["p"]])
            x = model._execute(model.text.model.embed_tokens(ids), model.sites[:LAYER])
            k, v, offset, mask = memory(layer.block, x, None, None)
            base = attend(layer.block, x, k, v, offset, mask)
            first = layer.step(
                base,
                attention_update(
                    layer.block, base, k, v, offset, mask, step=layer.step
                ),
            )
            second = layer.step(
                first,
                attention_update(
                    layer.block, first, k, v, offset, mask, step=layer.step
                ),
            )
            repeated_first = layer.step(
                base,
                attention_update(
                    layer.block, base, k, v, offset, mask, step=layer.step
                ),
            )
            reference = difference(recomputed(layer, x), second)
            set_reads(model, 1)
            one = model(ids)[:, -1]
            mx.eval(one)
            set_reads(model, 2)
            two = model(ids)[:, -1]
            rows.append(
                {
                    "group": group,
                    "row": r["id"],
                    "prompt": fingerprint(r["p"]),
                    "positions": ids.size,
                    "anchor_rms": rms(base),
                    "first_write_rms": rms(first - base),
                    "second_write_rms": rms(second - first),
                    "anchor_to_two": difference(base, second),
                    "second_using_original_state": difference(repeated_first, second),
                    "independently_recomputed_two": reference,
                    "final_logits": difference(one, two),
                }
            )
            guard(force=True)
        save(
            REPORT / "trained-state.json",
            {
                "standalone": digest(REPORT / "standalone.json"),
                "training": digest(BANK),
                "controller": read(DEST / "read-policy.json")["weights"],
                "rows": rows,
                "parameter_norms": {
                    k: mx.linalg.norm(v.astype(mx.float32)).item()
                    for k, v in tree_flatten(layer.step.parameters())
                },
                "peak_mlx": mx.get_peak_memory(),
            },
        )


if __name__ == "__main__":
    measure()
