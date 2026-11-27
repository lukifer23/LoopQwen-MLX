"""Independently load the content-trained policy, regenerate inputs and replay the interventions."""

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.conditional_features import CACHE
from loopqwen.research.content_read_fit import NAME, RESULTS, evidence_for, probes
from loopqwen.research.shared_query_features import arrays
from loopqwen.shared_query_assistant import SharedQueryAssistant
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed


def run(name=NAME):
    root = RESULTS / name
    committed(root / "checksums.json")
    if not (root / "step.safetensors").is_file():
        return
    s = read(root / "registration.json")
    src = s.get("feature_source", name)
    folder = CACHE / src / digest(RESULTS / src / "registration.json")
    idx = unpack(root / "feature-index.json.gz")
    rows = unpack(Path(s["training"]))
    ev = evidence_for(rows)
    net = SharedQueryAssistant(s["checkpoint"], root)._tool_model
    earlier = mx.load(str(RESULTS / s["parent"] / "step.safetensors"))
    weights = dict(tree_flatten(net.step.parameters()))
    assert set(weights) - set(earlier) == {"address.weight", "write.weight"} and all(
        mx.array_equal(weights[k], x).item() for k, x in earlier.items()
    )
    proof = []
    for x in idx:
        assert digest(folder / x["file"]) == x["sha256"]
    for x in sorted(idx, key=lambda r: fingerprint(["cold", r["id"]]))[:4]:
        r = next(w for w in rows if w["id"] == x["id"])
        z = arrays(net, r["p"], r["y"])
        old = mx.load(str(folder / x["file"]))
        assert z.keys() == old.keys() and all(
            mx.array_equal(v, old[k]).item() for k, v in z.items()
        )
        proof.append(x["id"])
    old = read(root / "probes.json")
    errors = []
    for mode in ("parent", "one", "trained"):
        results = probes(net, s, idx, folder, ev, mode)
        for k, r in results.items():
            errors.extend(abs(v - old[mode][k][field]) for field, v in r.items())
    receipt = dict(
        passed=max(errors) < 0.00001,
        fit=digest(root / "checksums.json"),
        paths=3,
        feature_hashes=len(idx),
        regenerated=proof,
        max_error=max(errors),
        frozen_tensors=len(earlier),
    )
    persist(Path("docs/measurements") / (name + "-cold.json"), receipt)
    print(receipt)


if __name__ == "__main__":
    run()
