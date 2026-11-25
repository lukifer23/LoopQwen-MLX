import gc

import mlx.core as mx

from loopqwen import REVISION
from loopqwen.deliberation import attach, greedy
from loopqwen.deployment import load_trained
from loopqwen.evaluation.generation_eval import generate
from loopqwen.evaluation.language7 import base
from loopqwen.io import read
from loopqwen.research.deliberation import REG


def test_ordinary_view_matches_base_and_recurrent_switch_replays_every_token(tmp_path):
    prompt = "<|im_start|>user\nExplain seven plus six.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    bm, t, _ = base()
    ids = t.encode(prompt, add_special_tokens=False)
    expected = bm(mx.array([ids]))[:, -1]
    early = greedy(bm, t, ids, 4)["token_ids"]
    mx.eval(expected)
    del bm
    gc.collect()
    m, t, _ = load_trained(read(REG)["parent"])
    attach(m, {"version": 1, "revision": REVISION, "switch_after": 4})
    assert mx.array_equal(expected, m.ordinary_depth(mx.array([ids]))[:, -1]).item()
    assert all(tok != 248069 for tok in early)
    late = [m(mx.array([ids + early]))[:, -1].argmax().item()]
    for disk in (None, tmp_path):
        result = generate(m, t, prompt, 5, offload_dir=disk)
        assert result["token_ids"] == early + late
        assert result["recurrent_from"] == 4
