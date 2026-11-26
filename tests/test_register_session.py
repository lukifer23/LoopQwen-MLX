import mlx.core as mx
from mlx import nn

from loopqwen.register_session import InputRows
from loopqwen.worker import acquire


def test_exact_input_embeddings_cover_unseen_generated_tokens(tmp_path):
    acquire()
    embedding = nn.Embedding(257, 64)
    embedding.weight = mx.arange(257 * 64).reshape(257, 64).astype(mx.bfloat16)
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), {"model.embed_tokens.weight": embedding.weight})
    prompt = [42, 19, 200, 42, 8]
    module = InputRows(embedding, path, prompt)
    assert module.initial_rows == 4
    for ids in (prompt, [256, 0, 42, 233, 233], [2, 256], [42, 2]):
        x = mx.array([ids])
        assert mx.array_equal(module(x), embedding(x)).item()
    assert module.rows.shape == (8, 64)
