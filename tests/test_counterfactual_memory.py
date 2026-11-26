from mlx_lm.utils import load_tokenizer

from loopqwen.io import read
from loopqwen.model_loader import local_snapshot
from loopqwen.research.counterfactual_memory import material


def test_unequal_token_width_is_rejected_before_full_context_inference():
    tokenizer = load_tokenizer(local_snapshot())
    spec = read("research/experiments/conditional-depth-128k.json")
    invalid = read("research/data/conditional-depth-128k.json")
    (a, b), (c, d) = invalid["key_spans"]
    assert b - a != d - c
    valid = material(tokenizer, spec)
    (a, b), (c, d) = valid["key_spans"]
    first, second = (row["prompt_ids"] for row in valid["panel"])
    assert len(first) == len(second) >= 131072
    assert valid["swapped_keys"] != invalid["swapped_keys"]
    assert b - a == d - c
    assert first[:a] + first[c:d] + first[b:c] + first[a:b] + first[d:] == second
