"""Cold native reconstruction using physical sites and original source-token continuations."""

import argparse
from pathlib import Path

import mlx.core as mx

from loopqwen.conditional_reentry import ConditionalAssistant, resident
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.conditional_features import changed_token_spans, feature_root
from loopqwen.research.contrastive_features import FEATURES, FIT, OLD
from loopqwen.research.contrastive_fit import index
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.registered import registered
from scripts.restore_conditional_features import run as verify_files
from scripts.table_policy_material import persist
from scripts.verify_conditional_features import physical
from scripts.verify_conditional_fit import hidden_cost

DOC = Path("docs/measurements")


def feature(name):
    s = registered(name)
    root = Path("research/results") / name
    inventory = verify_files(name, check_rebuild=3)
    mx.clear_cache()
    tok = tokenizer(s)
    net = resident(s["checkpoint"], root, s, read(root / "backbone.json"))
    rows = {w["id"]: w for w in unpack(Path(s["training"]))}
    idx = unpack(root / "feature-index.json.gz")
    tests = []
    for g in s["groups"]:
        pool = [r for r in idx if r["group"] == g]
        ranked = sorted(
            pool, key=lambda r: fingerprint([s["fit_seed"], "cold", r["id"]])
        )
        chosen = ranked[:2]
        for reuse in (True, False):
            subset = [
                r
                for r in ranked
                if any(n["same_prefix_only"] == reuse for n in r["negative"])
            ]
            if subset and subset[0] not in chosen:
                chosen.append(subset[0])
        for rec in chosen:
            row = rows[rec["id"]]
            p, y = token_pair(tok, row)
            a = mx.load(str(feature_root(s) / rec["file"]))
            off = len(p) - 1
            full = physical(net, p + y[:-1])[off:]
            ids = sorted({0, len(y) // 2, len(y) - 1})
            assert mx.array_equal(
                a["positive"][mx.array(ids)], full[mx.array(ids)]
            ).item()
            options = []
            for j, (neg, bad) in enumerate(
                zip(rec["negative"], row.get("rejected", []), strict=True)
            ):
                op, z = token_pair(tok, {**row, "target": bad["target"]})
                good, b = changed_token_spans(y, z)
                assert (
                    op == p
                    and good == neg["positive_positions"]
                    and b == neg["negative_positions"]
                )
                other = physical(net, p + z[:-1])[off + mx.array(b)]
                stored = a[f"negative_{j}"]
                expected = full[mx.array(b)] if neg["same_prefix_only"] else other
                assert mx.array_equal(stored, expected).item()
                error = mx.sqrt(
                    mx.mean(
                        mx.square(stored.astype(mx.float32) - other.astype(mx.float32))
                    )
                ).item()
                options.append(
                    {
                        "reused_same_prefix": neg["same_prefix_only"],
                        "positions": b,
                        "full_alternate_rms_difference": error,
                    }
                )
            tests.append(
                {
                    "id": row["id"],
                    "group": g,
                    "positive_positions": ids,
                    "matched_channels": 5,
                    "negative_checks": options,
                }
            )
    out = {
        "archive": digest(root / "checksums.json"),
        "backbone": read(root / "backbone.json")["sha256"],
        "physical_steps": 28,
        "full_inventory": inventory,
        "positive_positions": sum(len(r["positive_positions"]) for r in tests),
        "cases": tests,
    }
    persist(DOC / (name + "-cold.json"), out)
    print(name, len(tests), out["positive_positions"], flush=True)


def fitted():
    root = Path("research/results") / FIT
    s = registered(FIT)
    tok = tokenizer(s)
    actor = ConditionalAssistant(s["checkpoint"], root)
    net = actor._tool_model
    rows = {
        r["id"]: r
        for r in [
            *unpack(Path(read(OLD / "registration.json")["training"])),
            *unpack(Path(s["training"])),
        ]
    }
    _, archives = index(s)
    assert archives == read(root / "fit.json")["feature_archives"]
    checks = []
    for probe in read(root / "training-probes.json"):
        r = rows[probe["id"]]
        p, y = token_pair(tok, r)
        f = physical(net, p + y[:-1])[len(p) - 1 :]
        ordinary = f[:, 0]
        fit = net.readout(f)
        collapsed = net.readout(mx.broadcast_to(ordinary[:, None, :], f.shape))
        assert mx.array_equal(ordinary, collapsed).item()
        target = mx.array(y)
        a, b = [mx.mean(hidden_cost(net, h, target)).item() for h in (ordinary, fit)]
        assert abs(a - probe["ordinary"]) < 1e-4 and abs(b - probe["trained"]) < 1e-4
        checks.append(
            {
                "id": r["id"],
                "group": r["group"],
                "positions": len(y),
                "ordinary": a,
                "trained": b,
                "collapse_returns_ordinary": True,
                "maximum_mean_error": max(
                    abs(a - probe["ordinary"]), abs(b - probe["trained"])
                ),
            }
        )
    out = {
        "archive": digest(root / "checksums.json"),
        "weights": digest(root / "readout.safetensors"),
        "positions": sum(r["positions"] for r in checks),
        "max_mean_error": max(r["maximum_mean_error"] for r in checks),
        "checks": checks,
    }
    persist(DOC / (FIT + "-cold.json"), out)
    print(len(checks), out["positions"], out["max_mean_error"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=(*FEATURES, FIT))
    args = parser.parse_args()
    if args.name == FIT:
        fitted()
    else:
        feature(args.name)
