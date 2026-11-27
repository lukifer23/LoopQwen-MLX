"""Measure the trained anchor, later-pass dependence and measured logical compute."""

import argparse
from dataclasses import replace
from pathlib import Path
from time import monotonic

import mlx.core as mx
from mlx.utils import tree_flatten

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.middle_depth import MiddleAssistant
from loopqwen.model_loader import count
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.middle_depth_development import SOURCE, instance, prompt_ids
from scripts.middle_depth_material import ROWS
from scripts.middle_depth_profile import rms


def profile(arm, policy, stage):
    with worker("profile", "trained-state/" + stage + "/" + arm) as guard:
        m = MiddleAssistant(BASE, policy)
        model, tok = m._tool_model, m.tokenizer
        row = next(r for r in read(SOURCE) if r["id"] == "bridge")
        records = []
        for world in ("original", "changed"):
            text, tools = instance(row, world, Path("."))
            ids = mx.array([prompt_ids(tok, [dict(role="user", content=text)], tools)])
            anchor = model.prefix_hidden(ids)
            first = model.shared_block(anchor)
            alpha = model.effective_alpha()
            visit = (first + alpha * (anchor - first)).astype(anchor.dtype)
            record = dict(
                world=world,
                positions=ids.size,
                anchor=rms(anchor),
                first=rms(first),
                first_change=rms(first - anchor),
                alpha=alpha.item(),
            )
            if arm == "loop":
                second = model.shared_block(visit)
                ablated = model.shared_block(anchor)
                scale = rms(first - anchor)
                record.update(
                    retained_first=rms(visit - anchor) / scale,
                    retained_anchor=rms(visit - first) / scale,
                    second_change=rms(second - first),
                    second_depends_on_first=rms(second - ablated),
                )
                if (
                    min(
                        record["retained_first"],
                        record["retained_anchor"],
                        record["second_depends_on_first"],
                    )
                    <= 0
                ):
                    raise ValueError("Collapsed trained recurrence")
            records.append(record)
        execution = []
        for depth in range(1, 4) if arm == "loop" else (1,):
            model.span = replace(model.span, passes=depth)
            guard()
            start = monotonic()
            cache = model.make_cache()
            mx.eval(model(ids, cache))
            elapsed = monotonic() - start
            # Count the actual two-dimensional projections at each visited
            # layer. Attention products vary with history length. GDN, the
            # vocab head, elementwise kernels and backward passes are separate.
            macs = 0
            for i, _ in model.sites:
                for k, v in tree_flatten(model.text.layers[i].parameters()):
                    if v.ndim == 2 and (
                        k.endswith("weight")
                        or k.endswith("lora_a")
                        or k.endswith("lora_b")
                    ):
                        macs += v.size
            execution.append(
                dict(
                    depth=depth,
                    visits=len(model.sites),
                    physical_parameters=count(model),
                    trainable_parameters=count(model, True),
                    decoder_matrix_macs_per_position=macs,
                    positions=ids.size,
                    forward_seconds=elapsed,
                )
            )
        save(
            ROOT / f"trained-mechanism-{stage}-{arm}.json",
            dict(
                weights=digest(policy / "adapters.safetensors"),
                states=records,
                execution=execution,
            ),
        )


def work(step):
    rows = {r["id"]: r for r in unpack(ROWS)}
    rec = []
    for arm in ("ordinary", "loop"):
        from scripts.middle_depth_recover import fit_directory

        p = fit_directory(arm, step)
        state = p / f"step-{step:04d}" / "state.json"
        st = read(state)
        assert st["step"] == step
        sizes = []
        visits = 0
        for u in st["log"]:
            ns = [len(rows[key]["p"]) + len(rows[key]["y"]) - 1 for key in u["ids"]]
            exits = len(u["exit_loss"]) // len(ns)
            assert 1 <= exits <= 3
            visits += sum(ns) * sum(4 * k + 8 for k in range(1, exits + 1))
            sizes.extend(ns)
        rec.append(
            dict(
                arm=arm,
                steps=st["step"],
                positions=sum(sizes),
                suffix_and_middle_forward_layer_token_visits=visits,
                optimizer_seconds=sum(r["seconds"] for r in st["log"]),
                feature_seconds=sum(r["feature_seconds"] for r in st["log"]),
                features_created=sum(r["created"] for r in st["log"]),
                peak=max(r["peak"] for r in st["log"]),
                last_saved_state=digest(state),
            )
        )
    report = ROOT / ("fit-work.json" if step == 144 else f"fit-work-{step:04d}.json")
    if report.exists() and read(report) != rec:
        raise ValueError("Previously recorded matched optimizer work changed")
    save(report, rec)


def exits(step):
    """Summarize observed pre-update losses on identical last-epoch exposures."""
    from collections import defaultdict
    from statistics import fmean
    from scripts.middle_depth_recover import fit_directory

    deck = unpack(ROWS)
    epoch = len(deck) // 2
    if step % epoch or step < epoch:
        raise ValueError("Compare complete matched epochs only")
    states = {
        arm: fit_directory(arm, step) / f"step-{step:04d}" / "state.json"
        for arm in ("loop", "ordinary")
    }
    logs = [read(path)["log"][-epoch:] for path in states.values()]
    compared = []
    for i, (loop, ordinary) in enumerate(zip(*logs, strict=True)):
        expected = deck[2 * i : 2 * i + 2]
        assert loop["ids"] == ordinary["ids"] == [row["id"] for row in expected]
        assert loop["step"] == ordinary["step"] == step - epoch + i + 1
        k = len(loop["exit_loss"]) // 2
        assert k in (2, 3) and len(ordinary["exit_loss"]) == 2
        for j, row in enumerate(expected):
            scores = loop["exit_loss"][j * k : (j + 1) * k]
            compared.append(
                {
                    "id": row["id"],
                    "source": row["cluster"],
                    "family": row["family"],
                    "group": row["group"],
                    "update": loop["step"],
                    "loop": scores,
                    "ordinary": ordinary["exit_loss"][j],
                }
            )
    by_family = {}
    for family in sorted({r["family"] for r in compared}):
        cell = [r for r in compared if r["family"] == family]
        for label, left, right in (
            ("loop2_minus_loop1", 1, 0),
            ("loop3_minus_loop2", 2, 1),
            ("loop2_minus_trained_ordinary", 1, None),
        ):
            diffs = defaultdict(list)
            for r in cell:
                if left < len(r["loop"]):
                    ref = r["ordinary"] if right is None else r["loop"][right]
                    diffs[r["source"]].append(r["loop"][left] - ref)
            by_family.setdefault(family, {})[label] = {
                "sources": len(diffs),
                "rows": sum(map(len, diffs.values())),
                "mean_source_delta": fmean(map(fmean, diffs.values())),
                "sources_with_lower_objective": sum(
                    fmean(x) < 0 for x in diffs.values()
                ),
            }
    save(
        ROOT / f"fit-exits-{step:04d}.json",
        {
            "steps": step,
            "epoch": step // epoch,
            "states": {a: digest(p) for a, p in states.items()},
            "source_weighted": by_family,
            "rows": compared,
            "scope": "Observed training-turn objectives before each optimizer update in the named epoch. Values include the registered foil penalty where present. Warmup has two loop exits. Only public free generation earns answer credit.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("profile", "work", "exits"))
    p.add_argument("--arm", choices=("ordinary", "loop"))
    p.add_argument("--policy", type=Path)
    p.add_argument("--stage", default="terminal")
    p.add_argument("--step", type=int, default=144)
    a = p.parse_args()
    if a.action == "work":
        work(a.step)
    elif a.action == "exits":
        exits(a.step)
    else:
        profile(a.arm, a.policy, a.stage)
