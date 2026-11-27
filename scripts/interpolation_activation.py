"""Measure the sealed residual intervention on an independent prompt, with fresh native caches."""

import argparse
import signal
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx import nn

from loopqwen import qwen_tools
from loopqwen.interpolated_instruction import InterpolatedAssistant
from loopqwen.io import digest, read, save


def measure(source):
    protocol = read(source)
    policy = Path("research/results") / protocol["model_study"]
    spec = read(policy / "registration.json")
    start = perf_counter()
    mx.reset_peak_memory()
    file_evidence = spec.get("file_evidence", False)
    if protocol.get("compiled_reader"):
        from loopqwen.cross_view import CrossViewAssistant

        actor = CrossViewAssistant(spec["checkpoint"], policy)
    elif file_evidence:
        from loopqwen.file_evidence import FileEvidenceAssistant

        actor = FileEvidenceAssistant(spec["checkpoint"], policy)
    else:
        actor = InterpolatedAssistant(spec["checkpoint"], policy)
    net = actor._tool_model
    definitions = []
    history = [{"role": "user", "content": protocol["prompt"]}]
    observation = None
    if "document" in protocol:
        from loopqwen.csv_tools import tools

        document = Path(protocol["document"])
        assert digest(document) == protocol["document_sha256"]
        catalog = tools([document], keyed=True)
        definitions = qwen_tools.declarations(catalog)
        if "call" in protocol:
            from loopqwen.tool_session import compact

            call = protocol["call"]
            provider = next(t for t in catalog if t.name == call["name"])
            observation = provider.callback(**call["arguments"])
            assert observation == protocol["expected_return"]
            history.extend(
                [
                    qwen_tools.assistant_message([call]),
                    {"role": "tool", "content": compact(observation)},
                ]
            )
    prompt = qwen_tools.render(actor.tokenizer, history, definitions)
    tokens = actor.tokenizer.encode(prompt, add_special_tokens=False)
    inputs = mx.array([tokens])
    layer = net.span.start + net.span.length - 1
    if file_evidence:
        from safetensors import safe_open

        with safe_open(policy / "tool-adapters.safetensors", framework="np") as sf:
            identifiers = sorted(
                k.removesuffix(".lora_a") for k in sf.keys() if k.endswith(".lora_a")
            )
        projections = []
        for key in identifiers:
            _, _, _, index, path = key.split(".", 4)
            visits = [
                dict(net.text.layers[i].named_modules())[path]
                for i, _ in net.sites
                if i == int(index)
            ]
            assert (
                len(visits) == 2
                and visits[0] is visits[1]
                and isinstance(visits[0], nn.Linear)
            )
            projections.append(key)
        assert {int(k.split(".")[3]) for k in projections} == {22, 23}
    else:
        parameters = [
            dict(net.text.layers[i].named_modules()) for i, _ in net.sites if i == layer
        ]
        projections = spec["instruction_interpolation"]["projections"]
        assert len(parameters) == 2 and all(
            parameters[0][p] is parameters[1][p]
            and isinstance(parameters[0][p], nn.Linear)
            for p in projections
        )
    gdn_states = []
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
        if file_evidence:
            first, second = [state.slots[layer - 1, i] for i in (0, 1)]
            assert first is not second
            entries = []
            for x, y in zip(first.cache, second.cache, strict=True):
                assert (
                    x is not None
                    and y is not None
                    and x is not y
                    and x.shape == y.shape
                )
                delta = mx.sqrt(
                    mx.mean((x.astype(mx.float32) - y.astype(mx.float32)) ** 2)
                ).item()
                assert 0 < delta < float("inf")
                entries.append({"shape": list(x.shape), "visit_rms_difference": delta})
            assert len(entries) == 2
            gdn_states.append(entries)
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
    effective = float(net.effective_alpha().item()) if strength is None else strength
    return {
        **(
            {
                "actual_observation": observation,
                "compiled_weights": read(policy / "compiled.json")["weights"],
            }
            if observation is not None
            else {}
        ),
        **({"physical_gated_delta_states": gdn_states} if file_evidence else {}),
        "protocol_sha256": digest(source),
        "policy_sha256": digest(policy / "registration.json"),
        "prompt_ids": tokens,
        "shared_projections": projections,
        "physical_final_layer_offsets": physical,
        "order": [effective, 0, effective],
        "vocabulary_size": int(active.size),
        "maximum_absolute_logit_change": peak,
        "rms_logit_change": rms,
        "argmax_ids": [int(mx.argmax(v).item()) for v in (active, zero, repeat)],
        "identical_active_replay": True,
        "peak_bytes": mx.get_peak_memory(),
        "seconds": perf_counter() - start,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--role-binding", action="store_true")
    group.add_argument("--file-evidence", action="store_true")
    group.add_argument("--cross-view", action="store_true")
    args = parser.parse_args()
    stem = (
        "cross-view"
        if args.cross_view
        else "file-evidence"
        if args.file_evidence
        else "role-binding"
        if args.role_binding
        else "instruction-interpolation"
    )
    source = Path("research/diagnostics") / (stem + "-activation-01.json")
    destination = Path("docs/measurements") / (stem + "-activation.json")
    signal.alarm(read(source)["max_native_seconds"])
    evidence = measure(source)
    signal.alarm(0)
    if destination.is_file():
        prior = read(destination)
        assert all(
            prior[k] == v
            for k, v in evidence.items()
            if k not in ("seconds", "peak_bytes")
        )
    else:
        save(destination, evidence)
    print(evidence)
