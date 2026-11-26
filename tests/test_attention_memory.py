import mlx.core as mx
from mlx import nn
from mlx_lm.models.cache import KVCache

from loopqwen.attention_memory import AttentionReader, PassageScorer, attach_bias
from loopqwen.worker import acquire


def test_window_selection_streams_the_same_features_used_for_training():
    acquire()
    mx.random.seed(5)
    embedding = nn.Embedding(200, 48)
    scorer = PassageScorer(48, width=8)
    ids = [(i * 31 + i // 17) % 200 for i in range(2189)]
    expected = mx.argmax(scorer(embedding(mx.array(ids)))).item() * scorer.stride
    assert scorer.infer(embedding, ids) == expected
    assert AttentionReader(scorer, 6, 1024).select(None, ids[:1023]) is None


def test_bias_is_causal_and_preserves_every_older_key():
    acquire()

    from types import SimpleNamespace

    c = SimpleNamespace(slots={(3, 0): KVCache(), (11, 0): KVCache()})
    attach_bias(c, {"start": 2, "stop": 5, "length": 12, "gain": 6.0})
    a = c.slots[11, 0]
    a.offset = 8
    assert a.make_mask(2) == "causal"
    mask = a.make_mask(4).squeeze().tolist()
    for q in range(4):
        for k in range(12):
            expected = (
                -float("inf") if k > 8 + q else (6.0 if q == 3 and 2 <= k < 5 else 0.0)
            )
            assert mask[q][k] == expected
    a.offset = 12
    assert a.make_mask(1).reshape(-1).tolist() == [0, 0, 6, 6, 6] + [0] * 8
    assert c.slots[3, 0].make_mask(1, return_array=False, window_size=None) is None


def test_held_out_registers_contain_hard_negatives_and_one_visible_gold():
    from mlx_lm.utils import load_tokenizer

    from loopqwen.io import read
    from loopqwen.model_loader import local_snapshot

    tok = load_tokenizer(local_snapshot())
    data = read("research/data/attention-memory-01.json")
    for row in data["panel"] + data["calibration"]:
        ids = row["prompt_ids"]
        text = tok.decode(ids)
        assert row["minimum_length"] <= len(ids) <= row["minimum_length"] + 48
        assert text.count(row["gold"]) == 1
        assert tok.decode(ids[row["gold_start"] : row["gold_stop"]]) == row["gold"]
        assert len(set(row["hard_keys"])) == 3
        for key in row["hard_keys"]:
            assert key in text
            assert (
                sum(
                    a != b for a, b in zip(row["key"].split(), key.split(), strict=True)
                )
                == 1
            )


def test_releasing_an_attention_site_frees_its_full_history_immediately():
    import gc
    import weakref
    from types import SimpleNamespace

    enabled = gc.isenabled()
    gc.disable()
    try:
        cache = SimpleNamespace(slots={(11, 0): KVCache()})
        reference = weakref.ref(cache.slots[11, 0])
        attach_bias(cache, {"start": 3, "stop": 6, "length": 10, "gain": 3.0})
        del cache.slots[11, 0]
        assert reference() is None
    finally:
        if enabled:
            gc.enable()


def test_full_context_intervention_changes_only_the_two_key_names():
    import re

    from mlx_lm.utils import load_tokenizer

    from loopqwen.io import read
    from loopqwen.model_loader import local_snapshot

    tokenizer = load_tokenizer(local_snapshot())
    data = read("research/data/attention-memory-128k.json")
    documents = []
    for item in data["panel"]:
        ids = item["prompt_ids"]
        text = tokenizer.decode(ids)
        rows = re.findall(
            r"^The filing code under ([a-z]+ [a-z]+ [a-z]+) is ([0-9]{6})\.$",
            text,
            re.MULTILINE,
        )
        assert len(ids) >= 131072
        assert len(set(key for key, _ in rows)) == len(rows)
        assert dict(rows)[item["key"]] == item["gold"]
        assert (
            tokenizer.decode(ids[item["gold_start"] : item["gold_stop"]])
            == item["gold"]
        )
        documents.append(rows)
    first, second = documents
    assert [value for _, value in first] == [value for _, value in second]
    x, y = data["swapped_keys"]
    assert [(a[0], b[0]) for a, b in zip(first, second, strict=True) if a != b] == [
        (x, y),
        (y, x),
    ]


def test_reader_bias_reaches_quantized_attention_and_changes_the_output():
    from types import SimpleNamespace

    from loopqwen.page_prefill import DirectPackedPageCache

    acquire()
    cache = DirectPackedPageCache(24, 25)
    earlier = DirectPackedPageCache(24, 25)
    attach_bias(
        SimpleNamespace(slots={(3, 0): earlier, (11, 1): cache}),
        {"start": 5, "stop": 8, "length": 24, "gain": 7.0},
    )
    k = mx.zeros((1, 1, 24, 64), mx.bfloat16)
    v = mx.zeros_like(k)
    v[:, :, 5:8, :] = 1
    mask = cache.make_mask(24)
    bundles = cache.update_and_fetch(k, v)
    q = mx.zeros((1, 1, 24, 64), mx.bfloat16)
    answer = cache.dense_attention(q, *bundles, 0.125, mask)
    plain = earlier.dense_attention(q, *earlier.update_and_fetch(k, v), 0.125, "causal")
    assert answer[0, 0, -1].mean().item() > 0.95
    assert plain[0, 0, -1].mean().item() < 0.15
    assert cache.report()["reader_biased_queries"] == 1
    assert earlier.report()["reader_biased_queries"] == 0
