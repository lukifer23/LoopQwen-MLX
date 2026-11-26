from loopqwen.io import read
from loopqwen.research.deduction_data import answer
from loopqwen.research.science_data import normalize


def test_disjoint_memory_sources_and_frozen_gold():
    first = read("research/data/deduction-stories-01.json")
    second = read("research/data/deduction-stories-02.json")
    historical = {
        r["story"] for split, rows in first.items() if split != "train" for r in rows
    }
    seen = set()
    for split, rows in second.items():
        for r in rows:
            assert r["story"] not in seen
            seen.add(r["story"])
            if split != "train":
                assert r["story"] not in historical
            assert len(set(r["supports"])) == 2
    assert answer("She is afraid of wolves.") == "wolf"
    assert answer("cats or sheep") is None


def test_passage_uses_only_a_finished_user_message_and_open_assistant_turn():
    from mlx_lm.utils import load_tokenizer

    from loopqwen.evidence_loop import passage
    from loopqwen.model_loader import local_snapshot

    t = load_tokenizer(local_snapshot())
    row = read("research/data/deduction-stories-02.json")["replication"][0]
    assert passage(t, row["prompt_ids"]) == (row["facts"], row["question"])
    assert passage(t, row["prompt_ids"] + [248069]) is None
    assert passage(t, row["prompt_ids"] + [248046, 248045, 846, 198]) is None


def test_science_supports_and_distractors_do_not_cross_splits():
    table = read("research/data/science-composition-01.json")
    seen = set()
    for rows in table.values():
        current = {
            normalize(text) for r in rows for text in [*r["facts"], r["question"]]
        }
        assert current.isdisjoint(seen)
        assert all(len(set(r["supports"])) == 2 and len(r["facts"]) == 6 for r in rows)
        seen.update(current)


def test_science_scorer_accepts_answer_text_without_grading_an_article_as_a_letter():
    from loopqwen.research.science_data import answer

    options = ["beads", "clouds", "liquid", "rocks", "rain", "crystals", "dust", "mist"]
    assert answer("The answer is a liquid.", options) == "C"
    assert answer("**Answer: B.**", options) == "B"
    assert answer("B or C", options) is None


def test_induction_scorer_reads_a_conclusion_about_the_queried_person():
    from loopqwen.research.induction_data import answer

    assert answer("Lily is gray. Greg is white.", "What color is Greg?") == "white"
    assert answer("Greg is not white.", "What color is Greg?") is None
    assert answer("The final answer: grey.", "What color is Greg?") == "gray"


def test_science_reader_registers_its_key_projection_as_a_trainable_module():
    from mlx.utils import tree_flatten

    from loopqwen.research.science_reader import ScienceReader
    from loopqwen.worker import acquire

    acquire()
    model = ScienceReader(8, 4, list(range(10)), list(range(8)))
    weights = dict(tree_flatten(model.trainable_parameters()))
    assert "fact_keys.weight" in weights and callable(model.keys)


def test_family_scorer_preserves_relation_direction_and_multiword_qualifiers():
    from loopqwen.research.kinship_data import answer

    question = "How is [Paul] related to [Mary]?"
    assert answer("Paul is Mary's father.", question) == "father"
    assert answer("Mary is Paul's daughter.", question) is None
    assert answer("Paul is Mary's father-in-law.", question) is None
    assert answer("The relationship is nephew.", question) == "nephew"


def test_native_support_supervision_matches_the_qwen_attention_operator():
    from unittest.mock import patch

    import mlx.core as mx
    from mlx_lm.models import qwen3_next
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    from loopqwen.evidence_loop import unit
    from loopqwen.looped_model import LoopedModel
    from loopqwen.native_evidence import AnchoredNativeEvidence
    from loopqwen.recurrence import Span
    from loopqwen.worker import acquire

    acquire()
    mx.random.seed(91)
    cfg = TextModelArgs(
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        full_attention_interval=1,
        vocab_size=64,
    )
    model = LoopedModel(TextModel(cfg), Span(passes=1))
    net = AnchoredNativeEvidence(model, 8, [2, 3])
    facts = mx.random.normal((1, 3, 256))
    question = mx.random.normal((1, 256))
    states, inputs = net.states(facts, question, 3)
    assert len(states) == len(inputs) == 3
    for state, h in zip(states, inputs, strict=True):
        assert mx.array_equal(h[:, :-1], inputs[0][:, :-1]).item()
        with patch.object(
            qwen3_next,
            "scaled_dot_product_attention",
            wraps=qwen3_next.scaled_dot_product_attention,
        ) as op:
            expected = unit(net.layer(h, mask=create_attention_mask(h))[:, -1])
        assert mx.array_equal(state, expected).item()
        q, k, _ = op.call_args.args[:3]
        # An identity value matrix makes native SDPA return its own attention probabilities.
        n = k.shape[2]
        indicators = mx.broadcast_to(mx.eye(n, k.shape[-1], dtype=k.dtype), k.shape)
        native = mx.fast.scaled_dot_product_attention(
            q[:, :, -1:], k, indicators, scale=net.layer.self_attn.scale
        )
        weights = mx.softmax(net.native_scores(h), axis=-1)
        assert mx.allclose(weights, native[:, :, 0, :n], atol=0.005).item()
    assert not mx.array_equal(states[0], states[-1]).item()
