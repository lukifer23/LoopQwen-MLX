"""Measure the sealed residual intervention on an independent prompt, with fresh native caches."""

import signal
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx import nn

from loopqwen import qwen_tools
from loopqwen.interpolated_instruction import InterpolatedAssistant
from loopqwen.io import digest, read, save

SOURCE = Path("research/diagnostics/instruction-interpolation-activation-01.json")
DESTINATION = Path("docs/measurements/instruction-interpolation-activation.json")


def measure():
    protocol = read(SOURCE)
    policy = Path("research/results") / protocol["model_study"]
    spec = read(policy / "registration.json")
    start = perf_counter()
    mx.reset_peak_memory()
    actor = InterpolatedAssistant(spec["checkpoint"], policy)
    net = actor._tool_model
    prompt = qwen_tools.render(
        actor.tokenizer, [{"role": "user", "content": protocol["prompt"]}], []
    )
    tokens = actor.tokenizer.encode(prompt, add_special_tokens=False)
    inputs = mx.array([tokens])
    layer = net.span.start
    parameters = [
        dict(net.text.layers[i].named_modules()) for i, _ in net.sites if i == layer
    ]
    projections = spec["instruction_interpolation"]["projections"]
    assert len(parameters) == 2 and all(
        parameters[0][p] is parameters[1][p] and isinstance(parameters[0][p], nn.Linear)
        for p in projections
    )
    physical = []

    def last(strength):
        net.alpha_override = strength
        state = net.make_cache()
        out = (
            net.project(net.hidden(inputs, cache=state)[:, -1:, :])
            .astype(mx.float32)
            .reshape(-1)
        )
        mx.eval(out)
        a, b = [state.slots[layer, i] for i in (0, 1)]
        assert (
            a is not b and a.keys is not b.keys and a.offset == b.offset == len(tokens)
        )
        physical.append([a.offset, b.offset])
        return out

    strength = net.alpha_override
    active = last(strength)
    zero = last(0.0)
    repeat = last(strength)
    delta = active - zero
    peak = float(mx.max(mx.abs(delta)).item())
    rms = float(mx.sqrt(mx.mean(delta**2)).item())
    assert 0 < rms <= peak < float("inf") and bool(
        mx.array_equal(active, repeat).item()
    )
    return {
        "protocol_sha256": digest(SOURCE),
        "policy_sha256": digest(policy / "registration.json"),
        "prompt_ids": tokens,
        "shared_projections": projections,
        "physical_final_layer_offsets": physical,
        "order": [strength, 0, strength],
        "vocabulary_size": int(active.size),
        "maximum_absolute_logit_change": peak,
        "rms_logit_change": rms,
        "argmax_ids": [int(mx.argmax(v).item()) for v in (active, zero, repeat)],
        "identical_active_replay": True,
        "peak_bytes": mx.get_peak_memory(),
        "seconds": perf_counter() - start,
    }


if __name__ == "__main__":
    signal.alarm(read(SOURCE)["max_native_seconds"])
    evidence = measure()
    signal.alarm(0)
    if DESTINATION.is_file():
        prior = read(DESTINATION)
        assert all(
            prior[k] == v
            for k, v in evidence.items()
            if k not in ("seconds", "peak_bytes")
        )
    else:
        save(DESTINATION, evidence)
    print(evidence)
