import mlx.core as mx

from loopqwen.page_prefill import PageCache


def test_page_selection_preserves_history_and_ignores_later_queries():
    page = 2

    def select(future):
        c = PageCache(16, block=page, retrieved=1, recent=1, suffix=2)
        # Page 2 aligns best with the first query. Page 1 aligns best with the second.
        directions = [[0, 0], [0, 5], [4, 0], [1, 0], [0, 1], [2, 2]]
        for k, direction in enumerate(directions):
            keys = mx.array([[direction, direction]], dtype=mx.float32)[:, None]
            vals = mx.arange(k * page, (k + 1) * page, dtype=mx.float32).reshape(
                1, 1, page, 1
            )
            if k == 5:
                assert c.sparse(page)
            all_k, all_v = c.update_and_fetch(keys, vals)
        q = mx.array([[[[1.0, 0.0], future]]])
        _, selected = c.select(q, all_k, all_v)
        assert c.keys_and_values()[1].reshape(-1).tolist() == list(range(12))
        assert c.report()["attended"] == c.report()["possible"] - 2 * 4
        return selected.reshape(-1).tolist()

    assert select([0, -100]) == select([0, 100]) == [0, 1, 4, 5, 8, 9, 10, 11]


def test_packed_pages_restore_selected_rows_and_all_affine_factors():
    from loopqwen.page_prefill import PackedPageCache

    c = PackedPageCache(16, block=2, recent=1, retrieved=1, suffix=2)
    channels = mx.arange(64, dtype=mx.float32).reshape(1, 1, 1, 64)
    for i in range(6):
        tokens = mx.arange(i * 2, i * 2 + 2).reshape(1, 1, 2, 1)
        x = (tokens + channels / 64).astype(mx.bfloat16)
        bundles = c.update_and_fetch(x, x)
    full = c.restore(bundles)
    selected = c.restore(bundles, mx.ones((1, 1, 2, 64), dtype=mx.bfloat16))
    # The first page, highest-scoring distant page, recent and current.
    idx = mx.array([0, 1, 6, 7, 8, 9, 10, 11])
    for dense, sliced in zip(full, selected, strict=True):
        assert mx.array_equal(mx.take(dense, idx, axis=2), sliced).item()
    report = c.report()
    assert report["retained"] == report["offset"] == 12
    assert report["code_bytes"] == 2 * 12 * 64
    assert report["affine_bytes"] == 2 * 12 * 4


def test_direct_packed_query_tiles_match_full_mask_and_preserve_reserved_history():
    from mlx_lm.models.base import quantized_scaled_dot_product_attention

    from loopqwen.page_prefill import DirectPackedPageCache

    mx.random.seed(73111)
    cache = DirectPackedPageCache(27, 60)
    k, v = (mx.random.normal((1, 1, 27, 64)).astype(mx.bfloat16) for _ in range(2))
    bundles = cache.update_and_fetch(k, v)
    q = mx.random.normal((1, 2, 11, 64)).astype(mx.bfloat16)
    positions = mx.arange(16, 27)[:, None] >= mx.arange(27)[None, :]
    biased = mx.where(positions, 1.75 * (mx.arange(27) == 3), -float("inf")).astype(
        q.dtype
    )[None, None]
    for mask in ("causal", biased):
        dense = quantized_scaled_dot_product_attention(
            mx.array(q), *bundles, 0.125, mask
        )
        tiled = cache.dense_attention(mx.array(q), *bundles, 0.125, mask)
        assert mx.allclose(
            dense.astype(mx.float32), tiled.astype(mx.float32), atol=0.02, rtol=0.02
        ).item()
    report = cache.report()
    assert report["capacity"] == 60 and report["retained"] == 27
    assert report["direct_queries"] == 22 and report["direct_tiles"] == 4
    assert report["max_tile_queries"] == 8


def test_resident_register_cache_is_local_to_the_selected_request():
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    from loopqwen.evaluation.generation_eval import resident_cache
    from loopqwen.looped_model import LoopedModel
    from loopqwen.recurrence import Span
    from loopqwen.worker import acquire

    acquire()
    mx.random.seed(56213)
    cfg = TextModelArgs(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        full_attention_interval=1,
        vocab_size=128,
        tie_word_embeddings=True,
    )
    text = TextModel(cfg)
    text.set_dtype(mx.bfloat16)
    model = LoopedModel(text, Span(start=1, length=1, passes=2))
    model.eval()
    prompt = mx.array([[12, 67, 92]])
    baseline = model.hidden(prompt, model.make_cache())
    mx.eval(baseline)
    object.__setattr__(model, "packed_register_prefill", True)
    a, first = resident_cache(model, "register", 16384, 5, 128)
    mx.eval(model.hidden(prompt, a))
    _, second = resident_cache(model, "register", 16384, 7, 128)
    assert set(first) == set(second) == {(0, 0), (1, 0), (1, 1)}
    assert all(s.offset == 3 and s.capacity == 16388 for s in first.values())
    assert all(s.offset == 0 and s.capacity == 16390 for s in second.values())
    for route, length, block in (
        ("register", 16383, 128),
        ("ordinary", 16384, 128),
        ("register", 16384, 64),
    ):
        cache, slots = resident_cache(model, route, length, 5, block)
        assert not slots
        assert mx.array_equal(model.hidden(prompt, cache), baseline).item()
