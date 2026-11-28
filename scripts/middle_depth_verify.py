"""Real-donor restart, callback-prefix, causal and exact sparse-gradient verification."""

from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import save
from loopqwen.middle_depth import architecture, donor
from loopqwen.research.saved_document import SavedDocument
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.middle_depth_long_fit import adjoint_feature, output, residual
from scripts.middle_depth_material import ROWS


def sparse_gradient_comparison(model, x, y):
    """Compare full native differentiation to the cached residual at the same target positions."""
    target = model.text.layers[-1].mlp.down_proj

    def prefix_cache():
        cache = model.make_cache()
        mx.eval(model.hidden(x[:, :-64], cache))
        return cache

    def full(_, cache):
        logits = model(x[:, -64:], cache)[0, -y.size :].astype(mx.float32)
        return nn.losses.cross_entropy(logits, y).mean()

    z = residual(model, x[:, -64:], prefix_cache())
    mx.eval(z)

    def sparse(_):
        return nn.losses.cross_entropy(
            output(model, z)[0, -y.size :].astype(mx.float32), y
        ).mean()

    normal_logits = model(x[:, -64:], prefix_cache())
    sparse_logits = output(model, z)
    plain_error = mx.max(mx.abs(normal_logits - sparse_logits)).item()
    plain_full, plain_sparse = (
        full(target, prefix_cache()).item(),
        sparse(target).item(),
    )
    z, training_loss = adjoint_feature(model, x[:, -64:], y, prefix_cache())
    a, ga = nn.value_and_grad(target, full)(target, prefix_cache())
    b, gb = nn.value_and_grad(target, sparse)(target)
    a, b = a.item(), b.item()
    errors_grad = {}
    norms = {}
    relative_errors = {}
    for (key, full_gradient), (sparse_key, sparse_gradient) in zip(
        tree_flatten(ga), tree_flatten(gb), strict=True
    ):
        assert key == sparse_key
        delta = full_gradient - sparse_gradient
        errors_grad[key] = mx.max(mx.abs(delta)).item()
        norms[key] = mx.linalg.norm(sparse_gradient).item()
        difference = mx.linalg.norm(delta).item()
        relative_errors[key] = difference / norms[key] if norms[key] else None
    norm = norms["lora_b"]
    relative = relative_errors["lora_b"]
    diagnostic = dict(
        feature_training_loss=training_loss,
        normal_logit_max=plain_error,
        normal_full_loss=plain_full,
        normal_sparse_loss=plain_sparse,
        full_loss=a,
        sparse_loss=b,
        norm=norm,
        relative=relative,
        maximum=errors_grad,
        norms=norms,
        relative_errors=relative_errors,
    )
    return diagnostic


def run():
    with (
        worker("profile", "saved-prefix-and-sparse-adjoint") as guard,
        TemporaryDirectory() as tmp,
    ):
        text, tok = donor(BASE, "base")
        model = architecture(text, 2, 4619)
        r = next(r for r in unpack(ROWS) if r["group"] == "deliver")
        ids = r["p"][:256]
        folder = Path(tmp)
        count = 0

        def interrupt(i):
            nonlocal count
            guard()
            count += 1
            if count == 5:
                raise InterruptedError("Deliberate prefix boundary interruption")

        try:
            SavedDocument(model, ids, folder, "real-base-probe", interrupt)
        except InterruptedError as error:
            if str(error) != "Deliberate prefix boundary interruption":
                raise
        assert (folder / "prefix.json").is_file()
        reopened = SavedDocument(model, ids, folder, "real-base-probe", guard)
        native = model.make_cache()
        for i in range(0, len(ids), 128):
            mx.eval(model.hidden(mx.array([ids[i : i + 128]]), native))
        errors = []
        for tail in (r["p"][256:276], r["p"][256:266] + r["y"][:5]):
            branch = reopened.branch(model, ids + tail)
            plain = model.make_cache()
            for i in range(0, len(ids), 128):
                mx.eval(model.hidden(mx.array([ids[i : i + 128]]), plain))
            x = mx.array([tail])
            actual = branch.decode(x)
            ref = model(x, plain)
            errors.append(mx.max(mx.abs(actual - ref)).item())
            assert errors[-1] == 0
        x = mx.array([r["p"][:160]])
        y = mx.array(r["p"][141:161])
        diagnostic = sparse_gradient_comparison(model, x, y)
        a, b = diagnostic["full_loss"], diagnostic["sparse_loss"]
        norm, relative = diagnostic["norm"], diagnostic["relative"]
        errors_grad = diagnostic["maximum"]
        save(ROOT / "sparse-gradient-check.json", diagnostic)
        assert norm > 0 and abs(a - b) < 0.01 and relative < 0.02, diagnostic
        prior = model.project(model.hidden(x)[:, :60])
        changed = model.project(
            model.hidden(mx.concatenate([x[:, :60], x[:, 60:][:, ::-1]], axis=1))[
                :, :60
            ]
        )
        causal = mx.max(mx.abs(prior - changed)).item()
        assert causal == 0
        save(
            ROOT / "sparse-parity.json",
            dict(
                passed=True,
                model="Base initialization, two physical span visits",
                chunks=len(ids) // 128,
                full_chunk_positions=64,
                supervised_positions=y.size,
                interrupted_on_call=count,
                completed_groups=len(reopened.receipt["groups"]),
                reloaded_branch_max_logit_errors=errors,
                sites=model.sites,
                correct_causal_prefix=60,
                causal_max_logit_error=causal,
                full_loss=a,
                sparse_loss=b,
                gradient_max_errors=errors_grad,
                relative_gradient_error=relative,
                gradient_norm=norm,
                peak=mx.get_peak_memory(),
            ),
        )


if __name__ == "__main__":
    run()
