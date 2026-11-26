"""Response-only LoRA training through the shared final Qwen attention layer."""

import gc
import random
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from ..io import digest, read, save
from ..model_loader import count, load_base, local_snapshot
from ..tool_session import compact, initial_messages, prompt
from .tool_worlds import tools_for


def build(spec, passes):
    if "donor" in spec:
        from mlx_lm.utils import load_tokenizer

        from .projected_distillation import tutor

        donor, _ = tutor({"teacher": spec["donor"]})
        tok = load_tokenizer(local_snapshot())
        text = donor.text
        del donor
        from ..tool_model import quantize_prefix

        quantize_prefix(text, spec["start_layer"], spec["prefix_bits"])
        mx.eval(text.parameters())
        mx.clear_cache()
    else:
        text, tok, _ = load_base()
    mx.random.seed(spec["fit_seed"])
    from ..tool_model import tool_view

    model = tool_view(text, spec.get("start_layer", 23), spec["rank"], passes)
    return model, tok


def traces(rows):
    for row in rows:
        tools = tools_for(row)
        catalog = {t.name: t for t in tools}
        messages = initial_messages(row["question"], tools)
        for call in row["expected"]:
            target = compact(call)
            yield list(messages), target
            value = catalog[call["name"]].callback(**call["arguments"])
            messages.extend(
                (
                    dict(role="assistant", content=target),
                    dict(role="tool", content=compact(value)),
                )
            )
        yield messages, row["gold"]


def features(model, tok, rows, budget, *, directory=None):
    bank = []
    examples = (
        ((r["messages"], r["visible"], r["thinking"]) for r in rows["examples"])
        if isinstance(rows, dict)
        else ((m, v, "") for m, v in traces(rows))
    )
    for messages, visible, thinking in examples:
        budget.check()
        prefix = tok.encode(prompt(tok, messages), add_special_tokens=False)
        reply = tok.encode(
            (thinking + "\n" if thinking else "") + "</think>\n\n" + visible,
            add_special_tokens=False,
        ) + [tok.convert_tokens_to_ids("<|im_end|>")]
        ids = prefix + reply[:-1]
        h = model.prefix_hidden(mx.array([ids]))[0]
        mx.eval(h)
        if directory is None:
            value = h
        else:
            value = directory / f"{len(bank)}.safetensors"
            mx.save_safetensors(str(value), {"hidden": h})
            if len(bank) % 256 == 0:
                restored = mx.load(str(value))["hidden"]
                if restored.dtype != h.dtype or not mx.array_equal(restored, h).item():
                    raise ValueError("Serialized native prefix differs from its source")
        bank.append((value, len(prefix) - 1, reply))
        if len(bank) % 256 == 0:
            print("prepared frozen prefixes", len(bank), flush=True)
    return bank


def fit(model, tok, spec, out, budget, *, prepare=features):
    start = perf_counter()
    training = read(spec["training"])
    bank = prepare(model, tok, training, budget)
    pair_count = spec.get("pairs_per_batch", 0)
    if pair_count:
        from .reading_pairs import MatchedExamples

        paired = MatchedExamples(
            training["examples"], bank, tok, pair_count, spec["batch"]
        )
    prepared = perf_counter() - start
    if model.span.length > 1:
        from ..kernels.gdn_backend import select

        select(model, "native")
        model.train()
        model.checkpointing = True
    rng = random.Random(spec["fit_seed"])
    optimizer = optim.AdamW(learning_rate=spec["lr"], weight_decay=0.01)

    def loss_fn(net, h, rr, ss, targets, answer_locations):
        h = net._execute(h, net.sites[net.span.start :])
        final = net.text.model.norm(h[rr, ss])
        logits = net.project(final).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, targets).mean()
        if pair_count:
            loss = loss + spec["ranking_weight"] * paired.loss(logits, answer_locations)
        return loss

    vg = nn.value_and_grad(model, loss_fn)
    logs = []
    start = perf_counter()
    targets_seen = 0
    for step in range(spec["updates"]):
        budget.check()
        batch = paired.sample(rng) if pair_count else rng.sample(bank, spec["batch"])
        width = max(h.shape[0] for h, _, _ in batch)
        h = mx.stack(
            [mx.pad(x, ((0, width - x.shape[0]), (0, 0))) for x, _, _ in batch]
        )
        rr, ss, labels = [], [], []
        for i, (_, offset, ids) in enumerate(batch):
            rr.extend([i] * len(ids))
            ss.extend(range(offset, offset + len(ids)))
            labels.extend(ids)
        targets_seen += len(labels)
        loss, grads = vg(
            model,
            h,
            mx.array(rr),
            mx.array(ss),
            mx.array(labels),
            paired.targets(batch) if pair_count else mx.array([]),
        )
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state, loss)
        if step % 50 == 0 or step + 1 == spec["updates"]:
            logs.append(dict(step=step + 1, loss=loss.item()))
            print("fit", logs[-1], flush=True)
    elapsed = perf_counter() - start
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out / "tool-adapters.safetensors"), weights)
    save(
        out / "fit.json",
        dict(
            passes=model.span.passes,
            training=digest(Path(spec["training"])),
            trainable=count(model, True),
            turns=len(bank),
            updates=spec["updates"],
            targets=targets_seen,
            prepare_seconds=prepared,
            optimizer_seconds=elapsed,
            weights=digest(out / "tool-adapters.safetensors"),
            loss=logs,
        ),
    )


def restore(source, passes=None):
    source = Path(source)
    spec = read(source / "registration.json")
    meta = read(source / "fit.json")
    weights = source / "tool-adapters.safetensors"
    if digest(weights) != meta["weights"]:
        raise ValueError("Tool adapter identity changed")
    model, tok = build(spec, meta["passes"] if passes is None else passes)
    tensors = mx.load(str(weights))
    expected = dict(tree_flatten(model.trainable_parameters()))
    if tensors.keys() != expected.keys() or any(
        t.shape != expected[k].shape for k, t in tensors.items()
    ):
        raise ValueError("Tool adapter tensor set mismatch")
    model.load_weights(list(tensors.items()), strict=False)
    model.eval()
    return model, tok


def fitted(arm, spec, out, budget):
    if arm.startswith("restore:"):
        return restore(arm.split(":", 1)[1])
    passes = {"train_loop": 2, "train_dense": 1}[arm]
    if "initial_adapter" in spec:
        source = Path(spec["initial_adapter"])
        parent = read(source / "registration.json")
        if (
            digest(source / "checksums.json") != spec["initial_sha256"]
            or read(source / "fit.json")["passes"] != passes
            or any(
                parent.get(k) != spec.get(k)
                for k in ("donor", "prefix_bits", "start_layer", "rank")
            )
        ):
            raise ValueError(
                "Continued training requires the registered matching recurrent graph"
            )
        model, tok = restore(source)
        mx.random.seed(spec["fit_seed"])
        model.train()
        # Prefix weights are frozen and cached once. Inference mode uses the same
        # native gated-delta kernel that will supply the adapter at deployment.
        for layer in model.text.layers[: model.span.start]:
            layer.eval()
    else:
        model, tok = build(spec, passes)
    fit(model, tok, spec, out, budget)
    gc.collect()
    mx.clear_cache()
    model.eval()
    return model, tok
