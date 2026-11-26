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
