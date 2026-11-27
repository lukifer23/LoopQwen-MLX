"""Bounded donor-parity, hybrid-state and native multi-exit gradient profile."""
import argparse
from dataclasses import replace
from time import monotonic

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from loopqwen.io import save
from loopqwen.middle_depth import architecture, donor
from loopqwen.model_loader import count
from scripts.middle_depth_budget import BASE, ROOT, worker


def exit_loss(model, ids, labels, start, depth, negatives=()):
    # Computing one depth per gradient call bounds live suffix activations.
    h = None
    for h in model.block_states(ids, depth):
        pass
    hidden = model.suffix(h)[:, start:start + labels.size]
    logits = model.project(hidden)[0].astype(mx.float32)
    loss = nn.losses.cross_entropy(logits, labels).mean()
    if negatives:
        index, foil = zip(*negatives)
        positions = mx.array(index)
        loss = loss + 0.65 * nn.softplus(
            1 + logits[positions, mx.array(foil)] - logits[positions, labels[positions]]
        ).mean()
    return loss


def rms(x):
    return mx.sqrt(mx.square(x.astype(mx.float32)).mean()).item()


def profile(which):
    out = ROOT / f"profile-{which}.json"
    if out.exists():
        raise FileExistsError(out)
    with worker("profile", "architecture/" + which) as guard:
        text, tok = donor(BASE, which)
        prompt = ("The invoice contains three lots with seven items apiece. "
                  "One item from each lot was returned. How many items remain? ")
        ids = mx.array([tok.encode(prompt * 5, add_special_tokens=False)])
        reference = text(ids)[:, -1]
        mx.eval(reference)
        model = architecture(text, 3, 4619)
        span = model.span
        model.span = replace(span, passes=1)
        error = rms(model(ids)[:, -1] - reference)
        assert error == 0, error
        records = []
        for depth in (1, 2, 3):
            guard()
            model.span = replace(span, passes=depth)
            history = {}
            def observe(where, p, x):
                history[where, p] = x
            h = model.hidden(ids, observer=observe)
            logits = model.project(h[:, -1])
            cache = model.make_cache()
            for i in range(0, ids.size, 32):
                chunk = model.hidden(ids[:, i:i+32], cache)
                mx.eval(chunk)
            actual = model.project(chunk[:, -1])
            delta = rms(actual - logits) / rms(logits)
            assert delta < 0.05 and len({id(s) for s in cache.slots.values()}) == len(model.sites)
            offsets = [s.offset for s in cache.slots.values() if hasattr(s, "offset")]
            assert offsets and all(k == ids.size for k in offsets)
            entries = [dict(entry=rms(history["entry",p]), exit=rms(history["exit",p]),
                            change=rms(history["entry",p]-history["exit",p]))
                       for p in range(depth)]
            passthrough = None
            if depth > 1:
                passthrough = rms(history["entry",1]-history["entry",0]) / rms(
                    history["exit",0]-history["entry",0])
                assert 0.15 < passthrough < 0.25
            model.train()
            tick = monotonic()
            value, grads = nn.value_and_grad(model, exit_loss)(
                model, ids, ids[0, -8:], ids.size - 9, depth)
            _, norm = optim.clip_grad_norm(grads, 1.0)
            mx.eval(value, norm)
            assert 0 < norm.item() < float("inf") and mx.isfinite(value).item()
            records.append(dict(depth=depth, slots=len(cache.slots), attention_offsets=offsets,
                                states=entries, first_state_fraction=passthrough, chunk_rms=delta,
                                nll=value.item(), gradient=norm.item(), seconds=monotonic()-tick,
                                peak=mx.get_peak_memory()))
            del grads, cache, history
            model.eval()
        guard(force=True)
        save(out, dict(donor=which, passed=True, parity=error, rows=records, length=ids.size,
                       trainable=count(model, True), physical=count(model),
                       tensor_names=[k for k,_ in tree_flatten(model.trainable_parameters())]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("donor", choices=("instruction","base"))
    profile(p.parse_args().donor)
