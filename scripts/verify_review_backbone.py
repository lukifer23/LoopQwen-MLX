"""Check every released Base tensor and every shared frozen array in the review views."""

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from loopqwen.io import digest, read, save
from loopqwen.model_loader import load_base
from loopqwen.observation_review import ObservationAssistant

POLICY = Path("research/results/observation-review-01")


def main():
    spec = read(POLICY / "registration.json")
    directory = Path(spec["checkpoint"])
    base, _, meta = load_base()
    text = dict(tree_flatten(base.parameters()))
    path = directory / "model.safetensors"
    released = mx.load(str(path))
    assert text.keys() == released.keys()
    for name, a in text.items():
        b = released[name]
        assert (
            a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b).item()
        ), name
    source_count = len(text)
    del base, text, released, a, b
    mx.clear_cache()
    actor = ObservationAssistant(directory, POLICY)
    actor.load_tools()
    left = dict(tree_flatten(actor._tool_model.text.parameters()))
    right = {id(x) for _, x in tree_flatten(actor.review.text.parameters())}
    assert all(id(x) in right for x in left.values())
    record = {
        "revision": meta["revision"],
        "export_sha256": digest(path),
        "exact_text_tensors": source_count,
        "unique_base_parameters": meta["unique_parameters"],
        "shared_quantized_and_unquantized_tensors": len(left),
        "shared_bytes": sum(
            a.nbytes for a in {id(v): v for v in left.values()}.values()
        ),
        "review_adapter_sha256": read(POLICY / "fit.json")["weights"],
        "opening_sites": [list(s) for s in actor._tool_model.sites],
        "review_sites": [list(s) for s in actor.review.sites],
    }
    dest = Path("docs/measurements/observation-review-base-identity.json")
    if dest.exists():
        assert read(dest) == record
    else:
        save(dest, record)
    print(source_count, len(left), record["unique_base_parameters"])


if __name__ == "__main__":
    main()
