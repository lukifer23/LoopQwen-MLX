"""Keep the exact frozen embedding on disk between completed native training operations."""

from pathlib import Path

import mlx.core as mx
from mlx import nn

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.ordinary_model import ordinary
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.middle_depth_material import ROWS


class FileEmbedding(nn.Embedding):
    """Use the installed MLX Embedding operations on the certified tensor afresh."""

    def __init__(self, file):
        # The regular embedding initializer would allocate a second random
        # vocabulary. This frozen module obtains its native weight from the file.
        nn.Module.__init__(self)
        self.file = str(file)
        self.reads = 0

    @property
    def weight(self):
        self.reads += 1
        return mx.load(self.file)["model.embed_tokens.weight"]


def install(model):
    if not model.args.tie_word_embeddings:
        raise ValueError(
            "This operating point requires the certified tied Qwen vocabulary"
        )
    file = BASE / "model.safetensors"
    saved = FileEmbedding(file)
    model.text.model.embed_tokens = saved
    return saved


def qualify():
    path = ROOT / "file-backed-head.json"
    with worker("profile", "frozen-head-output-and-adjoint") as guard:
        base, tok, _ = ordinary(BASE, require_release=True, lazy=True)
        del tok
        model = base
        row = next(r for r in unpack(ROWS) if r["group"] == "deliver")
        # Check actual native embeddings, vocabulary scores, loss and hidden
        # adjoint. All of these use the same installed MLX operators in both modes.
        ids = mx.array([row["p"][-10:]])
        y = mx.array(row["y"][: ids.size])
        used = int(y.size)
        h = model.text.model.embed_tokens(ids)[:, :used]
        mx.eval(h)

        def result():
            guard()
            x = model.text.model.embed_tokens(ids)
            logits = model.project(h)

            def loss(z):
                return nn.losses.cross_entropy(
                    model.project(z)[0].astype(mx.float32), y
                ).mean()

            value, grad = mx.value_and_grad(loss)(h)
            mx.eval(x, logits, value, grad)
            return [x, logits, value, grad]

        normal = result()
        before = mx.get_active_memory()
        module = install(model)
        actual = result()
        errors = [
            mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
            for a, b in zip(normal, actual, strict=True)
        ]
        out = {
            "passed": max(errors) == 0,
            "source": row["id"],
            "errors": errors,
            "tensor_file": read(BASE / "tool-export-files.json")["model.safetensors"],
            "implementation": digest(Path(__file__)),
            "physical_bytes": 248320 * 1024 * 2,
            "reads": module.reads,
            "active_before": before,
            "active_after": mx.get_active_memory(),
        }
        save(path, out)
        if not out["passed"]:
            raise ValueError("The file-backed vocabulary changed the native layer")
        print(out, flush=True)


if __name__ == "__main__":
    qualify()
