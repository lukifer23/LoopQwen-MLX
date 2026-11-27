"""Regenerate two prespecified training rollout groups from their actual pre-update policies."""

import argparse
import shutil
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.reentry import resident
from loopqwen.research.table_policy_learning import NAME, PARENT
from loopqwen.worker import acquire

RESULT = Path("research/results") / NAME
WARM = Path("research/results") / PARENT
INTERMEDIATE = Path("research/diagnostics/table-policy-replay")
PREVIOUS_STEP = 32
CERTIFICATE = INTERMEDIATE / "source.json"
WEIGHTS = INTERMEDIATE / "after-32.safetensors"


def seal(log):
    (source,) = Path("checkpoints/research", NAME).glob(f"step-{PREVIOUS_STEP:06d}-*")
    state = read(source / "state.json")
    assert state["step"] == PREVIOUS_STEP and state["rollouts"] == log[:PREVIOUS_STEP]
    assert state["registration"] == digest(RESULT / "registration.json")
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    origin = source / "adapters.safetensors"
    if WEIGHTS.is_file():
        assert digest(WEIGHTS) == digest(origin)
    else:
        shutil.copyfile(origin, WEIGHTS)
    receipt = {
        "step": PREVIOUS_STEP,
        "preregistered_training": state["registration"],
        "original_checkpoint": source.name,
        "weights": digest(WEIGHTS),
        "rollouts_before": fingerprint(state["rollouts"]),
    }
    if CERTIFICATE.is_file():
        assert read(CERTIFICATE) == receipt
    else:
        save(CERTIFICATE, receipt)


def regenerate(net, prefix, draws, seed, temperature):
    n = len(draws)
    keys = mx.random.key(seed)
    cache = net.make_cache()
    ids = mx.array([prefix] * n)
    errors = [[] for _ in draws]
    regenerated = [[] for _ in draws]
    for i in range(max(len(r["tokens"]) for r in draws)):
        scores = (
            net.project(net.hidden(ids, cache)[:, -1]).astype(mx.float32) / temperature
        )
        keys, sample_key = mx.random.split(keys)
        choice = mx.random.categorical(scores, key=sample_key)
        # Witness for the original sampled token also uses an independent normalization primitive.
        active = [i < len(r["tokens"]) for r in draws]
        recorded = mx.array(
            [
                r["tokens"][i] if live else 0
                for r, live in zip(draws, active, strict=True)
            ]
        )
        logp = mx.take_along_axis(nn.log_softmax(scores), recorded[:, None], axis=-1)[
            :, 0
        ]
        mx.eval(logp, choice)
        for j, (lp, token) in enumerate(
            zip(logp.tolist(), choice.tolist(), strict=True)
        ):
            if active[j]:
                regenerated[j].append(token)
                errors[j].append(abs(lp - draws[j]["behavior_logp"][i]))
        ids = choice[:, None]
    return [
        {
            "emitted": len(y),
            "identical_token_ids": y == r["tokens"],
            "matching_positions": sum(
                a == b for a, b in zip(y, r["tokens"], strict=True)
            ),
            "recorded_sha256": fingerprint(r["tokens"]),
            "regenerated_sha256": fingerprint(y),
            "mean_abs_logp_error": sum(e) / len(e),
            "max_abs_logp_error": max(e),
        }
        for y, r, e in zip(regenerated, draws, errors, strict=True)
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal-midpoint", action="store_true")
    args = p.parse_args()
    s = read(RESULT / "registration.json")
    decision = read(RESULT / "decision.json")
    files = read(RESULT / "checksums.json")
    name = (
        "partial-state.json.gz"
        if decision.get("status") == "interrupted"
        else "on-policy-rollouts.json.gz"
    )
    assert files[name] == digest(RESULT / name) and files[
        "registration.json"
    ] == digest(RESULT / "registration.json")
    log = unpack(RESULT / name)
    if isinstance(log, dict):
        log = log["rollouts"]
    if not log:
        raise ValueError("No durable rollouts to regenerate")
    if len(log) > PREVIOUS_STEP and args.seal_midpoint:
        seal(log)
    selections = [(1, WARM / "reentry.safetensors")]
    assert (
        digest(selections[0][1]) == read(WARM / "checksums.json")["reentry.safetensors"]
    )
    if len(log) > PREVIOUS_STEP:
        certificate = read(CERTIFICATE)
        assert certificate["preregistered_training"] == files["registration.json"]
        assert certificate["rollouts_before"] == fingerprint(log[:PREVIOUS_STEP])
        assert digest(WEIGHTS) == certificate["weights"]
        selections.append((PREVIOUS_STEP + 1, WEIGHTS))
    acquire()
    net = resident(s["checkpoint"], WARM, s, read(WARM / "fit.json")["backbone"])
    results = []
    for step, weights in selections:
        row = log[step - 1]
        trained = mx.load(str(weights))
        expected = dict(tree_flatten(net.trainable_parameters()))
        assert trained.keys() == expected.keys() and all(
            t.shape == expected[k].shape and t.dtype == expected[k].dtype
            for k, t in trained.items()
        )
        net.load_weights(list(trained.items()), strict=False)
        net.eval()
        assert row["step"] == step
        for j, (prefix, batch) in enumerate(
            zip(row["prefixes"], row["draws"], strict=True)
        ):
            seed = s["fit_seed"] + step * 4 + j
            results.append(
                {
                    "update": step,
                    "question": row["question"],
                    "stage": row["stage"],
                    "conditional_prompt_sha256": fingerprint(prefix),
                    "prompt_length": len(prefix),
                    "sampling_seed": seed,
                    "weights": digest(weights),
                    "draws": regenerate(net, prefix, batch, seed, s["temperature"]),
                }
            )
    all_draws = [d for group in results for d in group["draws"]]
    doc = {
        "sample_temperature": s["temperature"],
        "sealed_rollouts": files[name],
        "groups": results,
        "sampled_tokens": sum(d["emitted"] for d in all_draws),
        "matching_tokens": sum(d["matching_positions"] for d in all_draws),
        "all_ids_identical": all(d["identical_token_ids"] for d in all_draws),
        "max_abs_logp_error": max(d["max_abs_logp_error"] for d in all_draws),
    }
    path = Path("docs/measurements/table-policy-native-sampling.json")
    if path.is_file():
        assert read(path) == doc
    else:
        save(path, doc)
    print(
        doc["matching_tokens"],
        "of",
        doc["sampled_tokens"],
        "native IDs reproduced; maximum logp error",
        doc["max_abs_logp_error"],
        flush=True,
    )
    assert doc["all_ids_identical"] and doc["max_abs_logp_error"] <= 1e-4


if __name__ == "__main__":
    main()
