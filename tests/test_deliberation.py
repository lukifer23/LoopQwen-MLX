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


def test_learned_exit_uses_full_hidden_state_for_resident_and_disk(tmp_path):
    import pytest

    from loopqwen.io import digest

    model, t, _ = load_trained(read(REG)["parent"])
    p = t.apply_chat_template(
        [{"role": "user", "content": "How many hours are in three days?"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    ids = t.encode(p, add_special_tokens=False)
    path = tmp_path / "exit-readout.safetensors"
    weight = mx.concatenate(
        [mx.zeros((model.args.hidden_size + 1,)), mx.array([1000.0])]
    )
    mx.save_safetensors(str(path), {"weight": weight})
    policy = {
        "version": 2,
        "revision": REVISION,
        "switch_after": 2,
        "readout_sha256": digest(path),
    }
    attach(model, policy, directory=tmp_path)
    ordinary = greedy(model.ordinary_depth, t, ids, 2)["token_ids"]
    assert 248069 not in ordinary
    # Once the learned logit closes the thought, the following token again comes
    # from the unmodified recurrent vocabulary projection.
    after = model(mx.array([ids + ordinary + [248069]]))[:, -1].argmax().item()
    for disk in (None, tmp_path):
        value = greedy(model, t, ids, 4, offload_dir=disk)
        assert value["token_ids"] == ordinary + [248069, after]
    with pytest.raises(ValueError, match="weights differ"):
        attach(model, dict(policy, readout_sha256="0" * 64), directory=tmp_path)
