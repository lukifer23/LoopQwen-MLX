"""Separate upstream cache partition drift from the added recurrent output computation."""

from pathlib import Path
from time import perf_counter

import mlx.core as mx

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.looped_model import LoopedModel
from loopqwen.recurrence import Span
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.shared_query_features import FEATURES, OLD, RESULTS
from loopqwen.shared_query_assistant import from_backbone
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/shared-query-partition.json")
RESULT = Path("docs/measurements/shared-query-partition.json")


def run():
    committed(PLAN)
    cfg = read(PLAN)
    root = RESULTS / FEATURES
    assert digest(root / "checksums.json") == cfg["archive"]
    s = read(root / "registration.json")
    tok = tokenizer(s)
    rows = unpack(Path(s["training"]))
    started = perf_counter()
    net = from_backbone(s["checkpoint"], root, s, read(OLD / "backbone.json"))
    dense = LoopedModel(net.text, Span(len(net.text.layers) - 1, 1, 1))
    dense.eval()
    old = read(root / "parity.json")
    measured = []
    for group, prev in zip(cfg["groups"], old, strict=True):
        w = min(
            (r for r in rows if r["group"] == group),
            key=lambda r: fingerprint([s["fit_seed"], r["id"]]),
        )
        p, _ = token_pair(tok, w)
        ids = mx.array([p])
        mid = len(p) // 2
        out = {}
        for arm, actor in (("ordinary", dense), ("loop", net)):
            for pattern in ("full", "split"):
                c = actor.make_cache()
                if pattern == "split":
                    actor.hidden(ids[:, :mid], c)
                    block = ids[:, mid:]
                else:
                    block = ids
                h = (
                    actor.hidden(block, c)[:, -1:]
                    if arm == "ordinary"
                    else actor.last_hidden(block, c)
                )
                out[arm, pattern] = actor.project(h).astype(mx.float32)
                mx.eval(out[arm, pattern])
                assert (
                    perf_counter() - started <= cfg["wall_seconds"]
                    and mx.get_peak_memory() <= cfg["peak_bytes"]
                )

        def distance(a, b, out=out):
            return mx.max(mx.abs(out[a] - out[b])).item()

        record = {
            "id": w["id"],
            "tokens": len(p),
            "prompt": fingerprint(p),
            "ordinary_partition": distance(("ordinary", "full"), ("ordinary", "split")),
            "loop_partition": distance(("loop", "full"), ("loop", "split")),
            "same_full_partition": distance(("ordinary", "full"), ("loop", "full")),
            "same_split_partition": distance(("ordinary", "split"), ("loop", "split")),
            "top": [mx.argmax(o).item() for o in out.values()],
        }
        assert record["loop_partition"] == prev["split_error"]
        measured.append(record)
    persist(
        RESULT,
        {
            "plan": digest(PLAN),
            "measurements": measured,
            "seconds": perf_counter() - started,
            "peak_bytes": mx.get_peak_memory(),
        },
    )
    print(measured)


if __name__ == "__main__":
    run()
