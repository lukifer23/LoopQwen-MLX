"""Check the released raw-trace token protocol with the pinned tokenizer."""

from pathlib import Path

from mlx_lm.utils import load_tokenizer
from loopqwen.model_loader import local_snapshot
from loopqwen.research import core
from loopqwen.research.functional import orbit
from loopqwen.training.verified_trace import request


def test_prompt_independence_and_response_boundary():
    t = load_tokenizer(local_snapshot())
    row = dict(map=[5, 4, 7, 6, 0, 2, 1, 3], start=0, order=[6, 3, 0, 2, 1, 7, 5, 4])
    prompt = request(row)
    assert request(dict(**row, states=[7, 7, 7])) == prompt
    response = " 5, 2, 7"
    doc = core.doc(t, prompt, response)
    assert t.decode(doc["ids"][: doc["prefix"]]) == prompt
    assert t.decode(doc["ids"][doc["prefix"] :]) == response + "<|im_end|>"
    for marker in ("<think>", "</think>", "<|im_end|>"):
        assert t.decode([core.atom(t, marker)]) == marker


def test_transient_into_self_loop_and_cycle():
    mapping = [1, 2, 2, 2, 5, 6, 4, 0]
    assert orbit(mapping, 7) == (3, 1)
    assert orbit(mapping, 4) == (0, 3)
