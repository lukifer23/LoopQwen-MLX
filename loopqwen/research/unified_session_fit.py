"""Jointly fit externally grounded actions and user answers with equal request weighting."""

import gc
import random
from collections import Counter
from functools import partial
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten

from ..io import digest, read, save
from .preference_fit import batch_inputs
from .tool_fit import features, restore


def visible_start(tok, ids):
    i = ids.index(tok.convert_tokens_to_ids("</think>")) + 1
    while i < len(ids) - 1 and not tok.decode([ids[i]]).strip():
        i += 1
    if i >= len(ids) - 1:
        raise ValueError("Expected a nonempty supervised reply")
    return i


def supervision(replies, visible_weight, yes, no):
    weights = []
    first_pos = []
    opposite = []
    binary = []
    for ids, cut, v in replies:
        length = len(ids)
        base = (1 - visible_weight) / length
        first_pos.append(len(weights) + cut)
        mark = (
            v in ("yes", "no")
            and len(ids[cut:-1]) == 1
            and ids[cut] == (yes if v == "yes" else no)
        )
        opposite.append(no if mark and v == "yes" else yes)
        binary.append(mark)
        weights.extend(
            [
                (base + (visible_weight / (length - cut) if j >= cut else 0))
                / len(replies)
                for j in range(length)
            ]
        )
    return weights, first_pos, opposite, binary


def joint_loss(
    m,
    h,
    rr,
    ss,
    targets,
    weights,
    first_pos,
    opposite,
    binary,
    *,
    first_weight,
    contrast_weight,
):
    layer = m.span.start
    first = m._execute(h, [(layer, 0)])
    second = m._execute(first, [(layer, 1)])
    logits = m.project(m.text.model.norm(second[rr, ss])).astype(mx.float32)
    final = (nn.losses.cross_entropy(logits, targets, reduction="none") * weights).sum()
    index = mx.arange(h.shape[0])
    forecast = m.project(m.text.model.norm(first[index, ss[first_pos]])).astype(
        mx.float32
    )
    y = targets[first_pos]
    anchor = nn.losses.cross_entropy(forecast, y).mean()
    opposing = nn.softplus(forecast[index, opposite] - forecast[index, y])
    contrast = (opposing * binary).sum() / mx.maximum(binary.sum(), 1)
    return final + first_weight * anchor + contrast_weight * contrast


def train(spec, root, budget):
    parent = Path(spec["public"])
    assert (
        digest(parent / "checksums.json") == spec["public_sha256"]
        and read(parent / "decision.json")["passed"]
    )
    net, tok = restore(parent)
    mx.random.seed(spec["fit_seed"])
    data = read(spec["training"])
    rows = data["examples"]
    start = perf_counter()
    bank = features(net, tok, data, budget)
    starts = [visible_start(tok, e[2]) for e in bank]
    counts = Counter(r["stratum"] for r in rows)
    groups = [
        [i for i, w in enumerate(rows) if w["stratum"] == k] for k in spec["strata"]
    ]
    assert all(groups) and sum(map(len, groups)) == len(rows)
    prepared = perf_counter() - start
    assert net.span.length == 1 and net.span.passes == 2
    yes, no = [tok.encode(s, add_special_tokens=False)[0] for s in ("yes", "no")]

    rng = random.Random(spec["fit_seed"])
    loss = partial(
        joint_loss,
        first_weight=spec["first_weight"],
        contrast_weight=spec["contrast_weight"],
    )
    vg = nn.value_and_grad(net, loss)
    opt = optimizers.AdamW(learning_rate=spec["lr"], weight_decay=0.01)
    logs = []
    seen = set()
    supervised = 0
    binary_examples = 0
    by_group = Counter()
    start = perf_counter()
    for step in range(spec["updates"]):
        indices = [rng.choice(group) for group in groups]
        rng.shuffle(indices)
        seen.update(indices)
        xs = [bank[i] for i in indices]
        h, rr, ss, targets = batch_inputs(xs)
        replies = [(bank[i][2], starts[i], rows[i]["visible"]) for i in indices]
        weights, first_pos, opposite, binary = supervision(
            replies, spec["visible_weight"], yes, no
        )
        by_group.update(rows[i]["stratum"] for i in indices)
        binary_examples += sum(binary)
        cost, grad = vg(
            net,
            h,
            rr,
            ss,
            targets,
            mx.array(weights),
            mx.array(first_pos),
            mx.array(opposite),
            mx.array(binary, mx.float32),
        )
        grad, _ = optimizers.clip_grad_norm(grad, 1.0)
        opt.update(net, grad)
        mx.eval(cost, opt.state, net.trainable_parameters())
        budget.check()
        supervised += targets.size
        if step % 100 == 0 or step + 1 == spec["updates"]:
            logs.append([step + 1, cost.item()])
            print("joint update", logs[-1], flush=True)
    params = dict(tree_flatten(net.trainable_parameters()))
    dest = root / "tool-adapters.safetensors"
    mx.save_safetensors(str(dest), params)
    save(
        root / "fit.json",
        {
            "training": digest(Path(spec["training"])),
            "weights": digest(dest),
            "passes": 2,
            "updates": spec["updates"],
            "turns": len(rows),
            "trainable": sum(v.size for v in params.values()),
            "targets": supervised,
            "contrast_examples": binary_examples,
            "prepare_seconds": prepared,
            "optimizer_seconds": perf_counter() - start,
            "loss": logs,
            "source_counts": counts,
            "sampled_by_source": by_group,
            "distinct_training_rows": len(seen),
        },
    )
    del bank, net, opt, vg
    gc.collect()
    mx.clear_cache()


def verify_sources(spec, training, material):
    """Recover all 24 human questions directly from their independently written records."""
    import json

    from huggingface_hub import hf_hub_download

    source = training["sources"]["Dolly"]
    path = Path(
        hf_hub_download(
            "databricks/databricks-dolly-15k",
            "databricks-dolly-15k.jsonl",
            repo_type="dataset",
            revision=source["revision"],
        )
    )
    assert digest(path) == source["sha256"]
    originals = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        request = (row["instruction"].strip() + "\n\n" + row["context"].strip()).strip()
        originals.setdefault(request, []).append(row)
    held = set()
    for w in material:
        if w["family"] != "user_extraction":
            continue
        q = w["question"].removesuffix("\n\nReturn only the answer.")
        match = [
            r
            for r in originals[q]
            if r["response"].strip() == w["gold"]
            and "Dolly " + r["category"] == w["source"]
        ]
        assert match and match[0]["context"].strip()
        held.add(match[0]["context"].strip())
    assert len(held) == 24
    for w in training["examples"]:
        if not w.get("source", "").startswith("Dolly "):
            continue
        q = w["messages"][-1]["content"]
        assert any(
            r["response"].strip() == w["visible"]
            and "Dolly " + r["category"] == w["source"]
            and r["context"].strip() not in held
            for r in originals[q]
        )
    prior = read("research/data/shared-session-panel.json")["rows"]
    get_ids = lambda rows: {
        r["id"] for r in rows if r["family"] in ("math", "judgment")
    }
    assert len(get_ids(material)) == 48 and get_ids(material).isdisjoint(get_ids(prior))
    assert digest(Path(spec["public"]) / "checksums.json") == spec["public_sha256"]


def verify_sampling(spec, records, meta, tok):
    lengths = []
    binary = []
    for w in records:
        text = (
            (w["thinking"] + "\n" if w["thinking"] else "")
            + "</think>\n\n"
            + w["visible"]
        )
        ids = tok.encode(text, add_special_tokens=False) + [
            tok.convert_tokens_to_ids("<|im_end|>")
        ]
        start = visible_start(tok, ids)
        lengths.append(len(ids))
        binary.append(
            w["visible"] in ("yes", "no")
            and ids[start:-1] == tok.encode(w["visible"], add_special_tokens=False)
            and len(ids[start:-1]) == 1
        )
    groups = [
        [i for i, w in enumerate(records) if w["stratum"] == g] for g in spec["strata"]
    ]
    rng = random.Random(spec["fit_seed"])
    seen = set()
    targets = 0
    contrasts = 0
    for _ in range(spec["updates"]):
        ix = [rng.choice(g) for g in groups]
        rng.shuffle(ix)
        seen.update(ix)
        targets += sum(lengths[i] for i in ix)
        contrasts += sum(binary[i] for i in ix)
    assert (len(seen), targets, contrasts) == (
        meta["distinct_training_rows"],
        meta["targets"],
        meta["contrast_examples"],
    )
