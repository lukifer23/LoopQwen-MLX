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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("profile", "work"))
    p.add_argument("--arm", choices=("ordinary", "loop"))
    p.add_argument("--policy", type=Path)
    p.add_argument("--stage", default="terminal")
    p.add_argument("--step", type=int, default=144)
    a = p.parse_args()
    work(a.step) if a.action == "work" else profile(a.arm, a.policy, a.stage)
