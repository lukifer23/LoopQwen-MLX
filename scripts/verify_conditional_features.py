"""Reconstruct cached training states from the physical layers and from left context alone."""

from pathlib import Path

import mlx.core as mx

from loopqwen.conditional_reentry import resident
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.conditional_features import NAME, feature_root
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.registered import registered

OUT = Path("docs/measurements/conditional-native-feature-replay.json")


def physical(net, ids):
    text = net.text
    h = text.model.embed_tokens(mx.array([ids]))
    depth = len(text.layers)
    h = net._execute(h, [(i, 0) for i in range(depth)])
    states = [h]
    for i in range(depth - 4, depth):
        h = net._execute(h, [(i, 1)])
        states.append(h)
    return mx.stack(states, axis=-2)[0]


def run():
    s = registered(NAME)
    root = Path("research/results") / NAME
    assert read(root / "decision.json")["passed"] and s["passes"] == 2
    tok = tokenizer(s)
    net = resident(s["checkpoint"], root, s, read(root / "backbone.json"))
    net.eval()
    material = unpack(Path(s["training"]))
    index = unpack(root / "feature-index.json.gz")
    byid = {r["id"]: r for r in material}
    specs = []
    for group in sorted({r["group"] for r in index}):
        rr = min(
            (r for r in index if r["group"] == group),
            key=lambda r: fingerprint([71592, group, r["id"]]),
        )
        specs.append((rr, None))
    for edit in ("column", "direction", "predicate"):
        eligible = [
            (r, j)
            for r in index
            for j, n in enumerate(r["negative"])
            if n["edit"] == edit
        ]
        specs.append(
            min(
                eligible,
                key=lambda pair: fingerprint([71592, edit, pair[0]["id"], pair[1]]),
            )
        )
    receipts = []
    for r, j in specs:
        row = byid[r["id"]]
        path = feature_root(s) / r["file"]
        assert digest(path) == r["sha256"]
        arrays = mx.load(str(path))
        q = row if j is None else {**row, "target": row["rejected"][j]["target"]}
        p, y = token_pair(tok, q)
        ix = (
            list(range(len(y))) if j is None else r["negative"][j]["negative_positions"]
        )
        array = arrays["positive" if j is None else f"negative_{j}"]
        end_ids = p + y[:-1]
        full = physical(net, end_ids)[mx.array(ix) + len(p) - 1]
        same = mx.array_equal(full, array).item()
        assert same, f"Cold native regeneration differed on {r['id']} edit {j}"
        tests = []
        for k in sorted({0, len(ix) // 2, len(ix) - 1}):
            at = ix[k]
            left = physical(net, end_ids[: len(p) + at])[-1]
            diff = left.astype(mx.float32) - array[k].astype(mx.float32)
            reference = mx.maximum(
                mx.sqrt(mx.mean(mx.square(left.astype(mx.float32)))), 1e-12
            )

            def dist(features, label):
                logits = net.project(net.text.model.norm(features[0])).astype(
                    mx.float32
                )
                return mx.argmax(logits).item(), (
                    mx.logsumexp(logits) - logits[label]
                ).item()

            a, b = dist(left, y[at]), dist(array[k], y[at])
            tests.append(
                {
                    "target_offset": at,
                    "input_tokens": len(p) + at,
                    "label": y[at],
                    "rms_relative": (
                        mx.sqrt(mx.mean(mx.square(diff))) / reference
                    ).item(),
                    "maximum_feature_error": mx.max(mx.abs(diff)).item(),
                    "left_top": a[0],
                    "cached_top": b[0],
                    "left_nll": a[1],
                    "cached_nll": b[1],
                }
            )
        receipts.append(
            {
                "turn": r["id"],
                "group": r["group"],
                "edit": None if j is None else r["negative"][j]["edit"],
                "exact_full_regeneration": same,
                "regenerated_positions": len(ix),
                "left_context_checks": tests,
            }
        )
    result = {
        "feature_archive": digest(root / "checksums.json"),
        "backbone": read(root / "backbone.json")["sha256"],
        "condition": "Explicit evaluation of all 24 native physical layers followed by a fresh physical visit to the final four. For each checked label a separately evaluated sequence ends at the preceding token. BF16 kernels can differ by sequence length.",
        "records": receipts,
    }
    if OUT.exists():
        assert read(OUT) == result
    else:
        save(OUT, result)
    print(
        len(receipts),
        "positive/negative complete state replays",
        sum(r["regenerated_positions"] for r in receipts),
        "positions",
        sum(len(r["left_context_checks"]) for r in receipts),
        "left-only checks",
    )


if __name__ == "__main__":
    run()
