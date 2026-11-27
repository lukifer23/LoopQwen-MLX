"""Fit the final MLP at exact supervised positions following complete natural documents."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_attention_mask

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.middle_depth import MiddleAssistant
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from scripts.middle_depth_retention import saved
from scripts.agenda_sources import events, source
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.program_language_material import answer, user

SPEC = Path("research/diagnostics/middle-depth/long-fit.json")


def texts():
    book = Path("data/research_external/narrativeqa/field-books/seven-gables.txt")
    body = book.read_text()
    original = "and now that they find the treasure, it has long\nbeen worthless."
    revision = (
        "and now that they find the treasure, the ancient claim\nis still enforceable."
    )
    assert body.count(original) == 1
    question = (
        "When Holgrave reveals what was hidden behind the portrait, does recovering it now give the "
        "Pyncheon heirs an enforceable claim according to his assessment? Identify what it conveys and explain."
    )
    for world, changed, final in [
        (
            "published",
            False,
            "No. It is an old deed conveying eastern territory to Colonel Pyncheon and his heirs. Holgrave says the deed has long been worthless.",
        ),
        (
            "revision",
            True,
            "Yes. The recovered deed conveys eastern territory to Colonel Pyncheon and his heirs, and Holgrave says the claim is still enforceable.",
        ),
    ]:
        yield (
            "gables",
            world,
            book,
            body.replace(original, revision) if changed else body,
            question,
            final,
        )
    yr = 2025
    calendar = source(yr, "ical", False)
    body = calendar.read_text()
    records = events(yr)

    # Select independently corroborated, single-line event records from the first
    # and last quarter of the complete original calendar, with unequal durations.
    def duration(r):
        return int(
            (
                datetime.fromisoformat(r["end"]) - datetime.fromisoformat(r["start"])
            ).total_seconds()
            / 60
        )

    by_position = sorted((body.index("UID:" + r["uid"] + "\n"), r) for r in records)
    a = next(r for pos, r in by_position if pos < len(body) / 4 and duration(r) >= 45)
    b = next(
        r
        for pos, r in reversed(by_position)
        if pos > 3 * len(body) / 4 and 10 <= duration(r) < duration(a)
    )
    start = body.index("UID:" + b["uid"] + "\n")
    end = body.index("END:VEVENT", start)
    section = body[start:end]
    fmt = "%Y%m%dT%H%M%S"
    before = "DTEND:" + datetime.fromisoformat(b["end"]).strftime(fmt)
    after = "DTEND:" + (
        datetime.fromisoformat(b["start"]) + timedelta(minutes=duration(a) + 10)
    ).strftime(fmt)
    assert section.count(before) == 1
    question = f"Compare the scheduled durations of “{a['title']}” and “{b['title']}”. Which session runs longer, and by how many minutes? Show both durations."
    for world, doc, db in (
        ("published", body, duration(b)),
        (
            "revision",
            body[:start] + section.replace(before, after) + body[end:],
            duration(a) + 10,
        ),
    ):
        da = duration(a)
        winner = a if da > db else b
        response = f"{winner['title']} runs longer by {abs(da - db)} minutes. The first session lasts {da} minutes and the second lasts {db} minutes."
        yield "fosdem", world, calendar, doc, question, response


def examples(tok):
    for name, world, file, doc, q, final in texts():
        prompt = doc + "\n\n" + q
        p, y = token_pair(
            tok, dict(messages=[user(prompt)], tools=[], target=answer(final))
        )
        assert 131072 <= len(p) and len(p) + len(y) < 262144
        yield (
            name,
            world,
            dict(
                source=str(file),
                source_sha=digest(file),
                question=q,
                target=final,
                input=fingerprint(p),
                length=len(p),
                targets=fingerprint(y),
            ),
            p,
            y,
        )


def freeze():
    tok = tokenizer({"checkpoint": str(BASE)})
    rows = {k + "/" + v: m for k, v, m, p, y in examples(tok)}
    save(
        SPEC,
        dict(
            rows=rows,
            steps=16,
            lr=0.00012,
            seed=4619,
            method="Use the terminal short-fit weights. Keep every causal token, fitted span weights and native caches. Update only the last decoder MLP down-projection LoRA from the actual post-attention residuals of complete supported targets. Previous output positions from this final pointwise MLP cannot affect later attention histories. Check complete-graph gradient parity on a short prefix first. Fit four whole answers in equal turn weight for 16 updates. Compare public generation before and after; a likelihood gain alone cannot promote this intervention.",
        ),
    )


def output(model, post_attention):
    last = model.text.layers[-1]
    return model.project(
        model.text.model.norm(
            post_attention + last.mlp(last.post_attention_layernorm(post_attention))
        )
    )


def residual(model, ids, cache):
    sites = model.sites
    assert sites[-1] == (23, 0)
    h = model._execute(model.text.model.embed_tokens(ids), sites[:-1], cache)
    layer = model.text.layers[-1]
    slot = cache.slots[23, 0]
    return h + layer.self_attn(
        layer.input_layernorm(h), create_attention_mask(h, slot), slot
    )


def adjoint_feature(model, ids, labels, cache):
    """Capture the residual from the native training forward on the final chunk.

    Metal can choose a different SDPA forward when recording a VJP. Only the
    last pointwise MLP will change, so this residual remains immutable across
    subsequent optimizer steps. The full document prefix is already frozen.
    """
    target = model.text.layers[-1].mlp.down_proj

    def loss_and_feature(_):
        z = residual(model, ids, cache)
        scores = output(model, z)[0, -labels.size :].astype(mx.float32)
        return nn.losses.cross_entropy(scores, labels).mean(), z

    (value, state), gradient = nn.value_and_grad(target, loss_and_feature)(target)
    mx.eval(value, state, gradient)
    return state, value.item()


def extract(model, tok, arm, weights, guard):
    spec = read(SPEC)
    rows = list(examples(tok))
    result = []
    for doc in ("gables", "fosdem"):
        pair = [r for r in rows if r[0] == doc]
        a, b = (r[3] for r in pair)
        common = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y) // 128 * 128
        ident = fingerprint([weights, model.sites, doc, fingerprint(a[:common])])
        folder = Path("outputs/middle-depth/long-training") / ident
        prefix = None
        for name, world, meta, p, y in pair:
            assert meta == spec["rows"][name + "/" + world]
            ref = folder / (world + ".json")
            if not ref.exists():
                if prefix is None:
                    import sys

                    prefix = saved(
                        model,
                        a[:common],
                        folder / "prefix",
                        ident,
                        guard,
                        [
                            ".venv/bin/python",
                            "-m",
                            "scripts.middle_depth_long_fit",
                            *sys.argv[1:],
                        ],
                    )
                disk = prefix.branch(model, p)
                # Advance to just before the first required output, honoring all
                # physical hybrid states and each pass's offset.
                start = len(p) - 1 - max(0, 64 - len(y))
                assert start >= common
                for i in range(common, start, 128):
                    guard()
                    end = min(i + 128, start)
                    disk.decode_hidden(mx.array([p[i:end]]), emit=False)
                z, initial_loss = adjoint_feature(
                    model, mx.array([p[start:] + y[:-1]]), mx.array(y), disk.cache
                )
                f = folder / (world + ".safetensors")
                mx.save_safetensors(str(f), {"residual": z, "labels": mx.array(y)})
                save(
                    ref,
                    dict(
                        sha=digest(f),
                        input=meta["input"],
                        initial_loss=initial_loss,
                        positions=len(y),
                        prefix=ident,
                        offsets=[
                            s.offset
                            for s in disk.cache.slots.values()
                            if hasattr(s, "offset")
                        ],
                    ),
                )
                del disk
            f = folder / (world + ".safetensors")
            info = read(ref)
            assert digest(f) == info["sha"] and info["input"] == meta["input"]
            assert all(v == len(p) + len(y) - 1 for v in info["offsets"])
            result.append(mx.load(str(f)))
        del prefix
    return result


def run(arm, policy):
    from .middle_depth_fit import tensors

    cfg = read(SPEC)
    with worker("long", "full-fit/" + arm) as guard:
        assistant = MiddleAssistant(BASE, policy)
        model, tok = assistant._tool_model, assistant.tokenizer
        identity = read(policy / "registration.json")
        guard()
        audit = read(ROOT / "sparse-parity.json")
        assert audit["passed"]
        rows = extract(model, tok, arm, identity["weights"], guard)
        target = model.text.layers[-1].mlp.down_proj
        params = dict(tree_flatten(model.trainable_parameters()))

        def objective(_target, r):
            return nn.losses.cross_entropy(
                output(model, r["residual"])[0, -r["labels"].size :].astype(mx.float32),
                r["labels"],
            ).mean()

        grad = nn.value_and_grad(target, objective)
        opt = optim.AdamW(cfg["lr"], weight_decay=0.01, bias_correction=True)
        log = []
        start = monotonic()
        for n in range(cfg["steps"]):
            guard()
            from mlx.utils import tree_map

            gs = None
            losses = []
            for r in rows:
                value, g = grad(target, r)
                g = tree_map(lambda v: v / len(rows), g)
                gs = g if gs is None else tree_map(lambda a, b: a + b, gs, g)
                mx.eval(value, gs)
                losses.append(value.item())
            gs, norm = optim.clip_grad_norm(gs, 1.0)
            assert 0 < norm.item() < float("inf") and all(
                v < float("inf") for v in losses
            )
            opt.update(target, gs)
            mx.eval(target.trainable_parameters(), opt.state)
            log.append(dict(step=n + 1, loss=losses, gradient=norm.item()))
        directory = Path("checkpoints/middle-depth/long") / arm
        directory.mkdir(parents=True, exist_ok=True)
        f = directory / "adapters.safetensors"
        if f.exists():
            raise FileExistsError(f)
        tensors(f, model.trainable_parameters())
        save(
            directory / "registration.json",
            dict(
                identity,
                weights=digest(f),
                short_parent=identity["weights"],
                plan=digest(SPEC),
            ),
        )
        save(
            ROOT / f"long-fit-{arm}.json",
            dict(
                weights=digest(f),
                parent=identity["weights"],
                log=log,
                positions=[r["labels"].size for r in rows],
                seconds=monotonic() - start,
                peak=mx.get_peak_memory(),
                changed_tensors=[
                    k
                    for k, v in tree_flatten(model.trainable_parameters())
                    if not mx.array_equal(v, params[k]).item()
                ],
            ),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "fit"))
    p.add_argument("--arm", choices=("loop", "ordinary"))
    p.add_argument("--policy", type=Path)
    a = p.parse_args()
    freeze() if a.action == "freeze" else run(a.arm, a.policy)
