"""Train the registered real-history and simple continuations from identical parents."""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import monotonic

from mlx import nn

from loopqwen.calculator_tool import tool
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.middle_depth import architecture, donor
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.tool_session import compact
from loopqwen.training.optimizers import optimizer
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.middle_depth_fit import RUN, SPEC, checkpoint, loss, restore, update
from scripts.middle_depth_histories import PLAN, SOURCE
from scripts.middle_depth_material import ROWS
from scripts.program_language_material import answer

DECK = Path("research/sources/middle-depth/history-correction.json.gz")
REG = PLAN.with_name("history-correction.md")


def material():
    start = monotonic()
    tok = tokenizer({"checkpoint": str(BASE)})
    keys = read(PLAN)["questions"]
    inputs = {(r["question"], r["condition"]): r for r in unpack(SOURCE)}
    review = read(ROOT / "rollin-review.json")
    histories, shared, counts = [], [], Counter()
    for key in keys:
        r = inputs[key, "correct"]
        initial, action, _ = r["messages"]
        gold_p, gold_y = token_pair(
            tok, dict(messages=[initial], tools=r["tools"], target=action)
        )
        close = gold_y.index(248069)
        for arm in ("ordinary", "loop"):
            f = ROOT / "rollin" / arm / (fingerprint(key) + ".json.gz")
            raw = unpack(f)
            tr = raw["trajectory"]
            counts[arm + "_attempts"] += 1
            # The source review found one correct numeric delivery on these
            # sixteen training trajectories. It cannot be a failed prefix.
            if review[digest(f)]["correct"]:
                counts["excluded_correct"] += 1
                continue
            t = tr["turns"][0]
            assert t["prompt_ids"] == gold_p and not any(
                "observation" in t or "observations" in t for t in tr["turns"]
            )
            before = t["token_ids"][:32]
            boundaries = [i for i, v in enumerate(before) if v == 248069]
            if boundaries:
                before = before[: boundaries[min(1, len(boundaries) - 1)] + 1]
                paired, labels = gold_p + gold_y[: close + 1], gold_y[close + 1 :]
            else:
                paired, labels = gold_p, gold_y
            histories.append(
                dict(
                    id=arm + "/" + fingerprint(key),
                    report=r["report"],
                    p=gold_p + before,
                    simple=paired,
                    y=labels,
                    negatives=[],
                    trajectory=digest(f),
                    group="own",
                )
            )
        for condition in ("correct", "crossed"):
            w = inputs[key, condition]
            u, act, obs = w["messages"]
            args = act["tool_calls"][0]["function"]["arguments"]
            actual = tool().callback(**args)
            assert json.loads(obs["content"]) == actual == w["observed"]
            assert (actual == r["true"]) == (condition == "correct")
            target = answer(r["reference"]) if condition == "correct" else action
            p, y = token_pair(
                tok,
                dict(
                    messages=[u, act, dict(role="tool", content=compact(actual))],
                    tools=w["tools"],
                    target=target,
                ),
            )
            shared.append(
                dict(
                    id=condition + "/" + fingerprint(key),
                    report=r["report"],
                    p=p,
                    simple=p,
                    y=y,
                    negatives=[],
                    group="shared",
                    actual=actual,
                    submitted=args,
                )
            )
    assert len(shared) == 16 and len(histories) == 15
    histories.append(dict(histories[0], id=histories[0]["id"] + "/repeat"))
    ordinary = [
        r for r in unpack(ROWS) if r["family"] in ("chat", "reasoning", "documents")
    ]
    ordinary.sort(key=lambda r: fingerprint(["correction", r["id"]]))
    assert len(ordinary) >= 32
    deck = []
    for i, (h, s) in enumerate(zip(histories, shared, strict=True)):
        for z in (h, ordinary[2 * i], s, ordinary[2 * i + 1]):
            deck.append(dict(z, simple=z.get("simple", z["p"])))
    packed(DECK, deck)
    save(
        ROOT / "history-material.json",
        dict(
            plan=digest(REG),
            review=digest(ROOT / "rollin-review.json"),
            source=digest(SOURCE),
            deck=digest(DECK),
            rollins=dict(counts),
            real_model_callbacks=0,
            shared_executions=len(shared),
            unique_failed_prefixes=15,
            distinct_input_rows=sum(r["p"] != r["simple"] for r in deck),
            positions={k: sum(len(r[k]) for r in deck) for k in ("p", "simple", "y")},
            seconds=monotonic() - start,
        ),
    )


def fit(arm, condition):
    parent = RUN / "4619" / arm / "step-0576"
    dest = RUN / "correction" / condition / arm
    spec = read(SPEC)
    seed = spec["seed"]
    parent_state = read(parent / "state.json")
    ident = parent_state["identity"]
    depth = ident["depth"]
    rows = unpack(DECK)
    if condition == "simple":
        rows = [dict(r, p=r["simple"]) for r in rows]
    identity = dict(
        ident,
        parent=digest(parent / "state.json"),
        plan=digest(REG),
        data=digest(DECK),
        condition=condition,
        trainer=digest(Path(__file__)),
        order=fingerprint([r["id"] for r in rows]),
    )
    assert identity["data"] == read(ROOT / "history-material.json")["deck"]
    with worker("fit", f"correction/{arm}/{condition}") as guard:
        from scripts.middle_depth_recover import reconcile

        reconcile(dest)
        text, _ = donor(BASE, ident["donor"])
        model = architecture(text, depth, seed)
        if (
            digest(parent / "adapters.safetensors")
            != parent_state["hashes"]["adapters.safetensors"]
        ):
            raise ValueError("Corrupt short-fit parent")
        model.load_weights(str(parent / "adapters.safetensors"), strict=False)
        model.train()
        opt = optimizer(
            dict(learning_rate=0.00012, weight_decay=0.01, warmup=4, steps=32)
        )
        grad = nn.value_and_grad(model, loss)
        step, log = (
            restore(dest, model, opt, identity)
            if (dest / "latest.json").exists()
            else (0, [])
        )
        saved = step
        try:
            while step < 32:
                rec = update(
                    model,
                    opt,
                    grad,
                    rows[2 * step : 2 * step + 2],
                    step,
                    depth,
                    ident["feature_id"],
                    dict(spec, two_exit_warmup=0),
                    guard,
                )
                step += 1
                log.append(rec)
                if step % 8 == 0:
                    checkpoint(dest, model, opt, step, identity, log, seed, depth)
                    saved = step
        except InterruptedError:
            if step > saved:
                checkpoint(dest, model, opt, step, identity, log, seed, depth)
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("material", "fit"))
    p.add_argument("--arm", choices=("loop", "ordinary"))
    p.add_argument("--condition", choices=("actual", "simple"))
    a = p.parse_args()
    material() if a.action == "material" else fit(a.arm, a.condition)
