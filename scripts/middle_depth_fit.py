"""Train both same-donor siblings with recoverable prefix features and AdamW state."""

import argparse
import math
import signal
import tempfile
from pathlib import Path
from time import monotonic

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.middle_depth import architecture, donor
from loopqwen.training.optimizers import optimizer
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.middle_depth_material import ROWS

SPEC = Path("research/diagnostics/middle-depth/fit.json")
RUN = Path("checkpoints/middle-depth")
FEATURE = Path("outputs/middle-depth/features")


def tensors(path, values):
    mx.save_safetensors(str(path), dict(tree_flatten(values)))


def feature(model, row, identity):
    ids = row["p"] + row["y"][:-1]
    key = fingerprint([identity, ids])
    refs = FEATURE / "inputs"
    blobs = FEATURE / "objects"
    refs.mkdir(parents=True, exist_ok=True)
    blobs.mkdir(parents=True, exist_ok=True)
    ref = refs / (key + ".json")
    if ref.exists():
        item = read(ref)
        path = blobs / (item["sha256"] + ".safetensors")
        if digest(path) != item["sha256"]:
            raise ValueError("Cached prelude state differs from its digest")
        return mx.load(str(path))["state"], False
    if (
        sum(p.stat().st_size for p in blobs.glob("*.safetensors")) + len(ids) * 2048
        > 16 * 2**30
    ):
        raise InterruptedError("Feature retention cap")
    state = model.prefix_hidden(mx.array([ids]))
    mx.eval(state)
    temp = blobs / (key + ".partial.safetensors")
    mx.save_safetensors(str(temp), {"state": state})
    sha = digest(temp)
    path = blobs / (sha + ".safetensors")
    if path.exists():
        if digest(path) != sha:
            raise ValueError("Invalid existing feature object")
        temp.unlink()
    else:
        temp.replace(path)
    save(ref, dict(sha256=sha, bytes=path.stat().st_size(), positions=len(ids)))
    return state, True


def loss(model, anchor, labels, start, depth, negatives):
    h = anchor
    for p in range(depth):
        if p:
            h = (h + model.effective_alpha() * (anchor - h)).astype(h.dtype)
        h = model.shared_block(h)
    hidden = model.suffix(h)[:, start:]
    logits = model.project(hidden)[0].astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, labels).mean()
    if negatives:
        pos, token = zip(*negatives)
        ix = mx.array(pos)
        ce = (
            ce
            + 0.65
            * nn.softplus(
                1 + logits[ix, mx.array(token)] - logits[ix, labels[ix]]
            ).mean()
        )
    return ce


def checkpoint(folder, model, opt, step, identity, log, seed, depth):
    folder.mkdir(exist_ok=True, parents=True)
    staged = Path(tempfile.mkdtemp(prefix=".saving-", dir=folder))
    tensors(staged / "adapters.safetensors", model.trainable_parameters())
    tensors(staged / "optimizer.safetensors", opt.state)
    sha = digest(staged / "adapters.safetensors")
    save(
        staged / "registration.json",
        dict(
            middle_depth=True,
            weights=sha,
            seed=seed,
            trained_depth=depth,
            serving_depth=min(2, depth),
            donor=identity["donor"],
            identity=identity,
        ),
    )
    save(
        staged / "state.json",
        dict(
            step=step,
            next_rng_seed=seed + step,
            identity=identity,
            log=log,
            hashes={
                f: digest(staged / f)
                for f in (
                    "adapters.safetensors",
                    "optimizer.safetensors",
                    "registration.json",
                )
            },
        ),
    )
    final = folder / f"step-{step:04d}"
    if final.exists():
        raise FileExistsError(final)
    staged.rename(final)
    pointer = folder / "latest.json"
    previous = read(pointer)["path"] if pointer.exists() else None
    save(
        pointer,
        dict(path=final.name, previous=previous, state=digest(final / "state.json")),
    )
    return final


def restore(folder, model, opt, identity):
    pointer = read(folder / "latest.json")
    path = folder / pointer["path"]
    if digest(path / "state.json") != pointer["state"]:
        raise ValueError("Checkpoint state identity differs")
    state = read(path / "state.json")
    if state["identity"] != identity or any(
        digest(path / k) != v for k, v in state["hashes"].items()
    ):
        raise ValueError("Changed optimizer, weights, data, code or order")
    if state["next_rng_seed"] != identity["seed"] + state["step"]:
        raise ValueError("Incompatible RNG cursor")
    model.load_weights(str(path / "adapters.safetensors"), strict=False)
    opt.state = tree_unflatten(
        list(mx.load(str(path / "optimizer.safetensors")).items())
    )
    return state["step"], state["log"]


def update(model, opt, grad, rows, step, depth, feature_id, spec, guard):
    mx.random.seed(spec["seed"] + step)
    tick = monotonic()
    gs = None
    values = []
    seconds = 0.0
    built = 0
    for row in rows:
        guard()
        t = monotonic()
        anchor, created = feature(model, row, feature_id)
        mx.eval(anchor)
        seconds += monotonic() - t
        built += created
        exits = range(
            1, (min(2, depth) if step < spec["two_exit_warmup"] else depth) + 1
        )
        for d in exits:
            guard()
            value, g = grad(
                model,
                anchor,
                mx.array(row["y"]),
                len(row["p"]) - 1,
                d,
                row["negatives"],
            )
            scale = 1 / (len(rows) * len(exits))
            g = tree_map(lambda v: v * scale, g)
            gs = g if gs is None else tree_map(lambda a, b: a + b, gs, g)
            mx.eval(value, gs)
            values.append(value.item())
    gs, norm = optim.clip_grad_norm(gs, 1.0)
    magnitude = norm.item()
    if not 0 < magnitude < math.inf or not all(map(math.isfinite, values)):
        raise FloatingPointError("Nonfinite objective or gradient")
    guard(force=True)
    opt.update(model, gs)
    mx.eval(model.trainable_parameters(), opt.state)
    return dict(
        step=step + 1,
        ids=[r["id"] for r in rows],
        exit_loss=values,
        gradient=magnitude,
        alpha=model.effective_alpha().item(),
        feature_seconds=seconds,
        created=built,
        seconds=monotonic() - tick,
        peak=mx.get_peak_memory(),
    )


def run(arm, until, replay):
    spec = read(SPEC)
    seed = spec["seed"]
    depth = 3 if arm == "loop" else 1
    tag = read(ROOT / "donor-decision.json")["selected"]
    assert read(ROOT / f"profile-{tag}.json")["passed"]
    assert digest(ROWS) == spec["data"]
    deck = unpack(ROWS)
    epochs = spec["epochs"]
    order = [r for _ in range(epochs) for r in deck]
    batch = spec["batch"]
    total = len(order) // batch
    assert total * batch == len(order) and 1 <= until <= total
    folder = RUN / str(seed) / arm
    feature_id = fingerprint(
        [tag, spec["base_text"], spec["prelude_code"], "bf16/native/train", 12]
    )
    identity = dict(
        spec=digest(SPEC),
        seed=seed,
        donor=tag,
        order=fingerprint([r["id"] for r in order]),
        base_text=spec["base_text"],
        data=spec["data"],
        feature_id=feature_id,
        depth=depth,
        trainer=digest(Path(__file__)),
        architecture=digest(Path("loopqwen/middle_depth.py")),
    )
    stop = False

    def interrupt(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with worker("fit", f"{arm}/{until}/replay={replay}") as guard:
        text, _ = donor(BASE, tag)
        model = architecture(text, depth, seed)
        model.train()
        opt = optimizer(
            dict(
                learning_rate=spec["lr"],
                weight_decay=0.01,
                warmup=spec["warmup"],
                steps=total,
            )
        )
        grad = nn.value_and_grad(model, loss)
        step, log = (
            restore(folder, model, opt, identity)
            if (folder / "latest.json").exists()
            else (0, [])
        )
        if replay and until != step + 1:
            raise ValueError("Cold replay requires the following update")
        saved = step

        def persist():
            nonlocal saved
            if step > saved:
                checkpoint(folder, model, opt, step, identity, log, seed, depth)
                saved = step

        try:
            while step < until:
                guard(force=True)
                if stop:
                    break
                start = step * batch
                rec = update(
                    model,
                    opt,
                    grad,
                    order[start : start + batch],
                    step,
                    depth,
                    feature_id,
                    spec,
                    guard,
                )
                step += 1
                log.append(rec)
                if replay:
                    ref = ROOT / f"restart-{arm}.safetensors"
                    current = dict(
                        tree_flatten(
                            {
                                "model": model.trainable_parameters(),
                                "optimizer": opt.state,
                            }
                        )
                    )
                    previous = mx.load(str(ref))
                    assert previous.keys() == current.keys()
                    errors = {
                        k: mx.max(
                            mx.abs(
                                v.astype(mx.float32) - previous[k].astype(mx.float32)
                            )
                        ).item()
                        for k, v in current.items()
                    }
                    save(
                        ROOT / f"restart-{arm}.json",
                        dict(
                            step=step,
                            maximum=max(errors.values()),
                            tensors=len(errors),
                            reference=digest(ref),
                            rows=rec["ids"],
                            passed=max(errors.values()) == 0,
                        ),
                    )
                    if max(errors.values()) != 0:
                        raise ValueError("Restored update differs")
                if step % spec["checkpoint_every"] == 0 or step == until or stop:
                    persist()
                    print(arm, step, round(rec["seconds"], 2), flush=True)
        except InterruptedError:
            persist()
            raise
        if until == 1 and step == 1 and not replay:
            # Deliberately exit at the previous usable boundary. The next process
            # must reproduce this otherwise-discarded real second optimizer update.
            update(
                model,
                opt,
                grad,
                order[batch : 2 * batch],
                step,
                depth,
                feature_id,
                spec,
                guard,
            )
            tensors(
                ROOT / f"restart-{arm}.safetensors",
                {"model": model.trainable_parameters(), "optimizer": opt.state},
            )
        return step


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=("ordinary", "loop"))
    p.add_argument("--until", type=int, required=True)
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    run(a.arm, a.until, a.replay)
