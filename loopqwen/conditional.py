"""Select trained recurrent adapters from input evidence while sharing the dense decoder."""

from copy import deepcopy

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from .context_policy import features
from .deliberation import attach, ordinary_view
from .io import digest, read


def assistant_thinking(ids):
    """Recognize an open assistant thought in the pinned Qwen chat protocol."""
    for offset in range(len(ids) - 1, -1, -1):
        token = ids[offset]
        if token in (248044, 248046, 248069):
            return False
        if token == 248045:
            return list(ids[offset + 1 : offset + 4]) == [74455, 198, 248068]
    return False


class ConditionalDepth:
    def __init__(self, ordinary, experts, classifier, labels, confidence):
        self.ordinary = ordinary
        self.experts = experts
        self.classifier = classifier
        self.labels = labels
        self.confidence = confidence

    def views(self):
        return {"ordinary": self.ordinary, **self.experts}

    def choose(self, model, ids, cap):
        if not assistant_thinking(ids):
            return self.ordinary, "ordinary"
        evidence = features(model.text.model.embed_tokens, ids)
        distribution = mx.softmax(self.classifier(evidence)).tolist()
        index = max(range(len(distribution)), key=distribution.__getitem__)
        label = self.labels[index]
        if distribution[index] >= self.confidence:
            if label == "register" and len(ids) >= 1024:
                return model, "register"
            if label == "arithmetic" and len(ids) < 1024:
                if cap <= 256:
                    return model, "short_arithmetic"
                if "arithmetic" in self.experts:
                    return self.experts["arithmetic"], "arithmetic"
            if label == "entailment" and "entailment" in self.experts:
                return self.experts["entailment"], "entailment"
        return self.ordinary, "ordinary"


def load(model, folder, spec):
    labels = spec["labels"]
    if (
        labels != ["ordinary", "arithmetic", "register", "entailment"]
        or not 0.5 < spec["confidence"] < 1.0
    ):
        raise ValueError("Unsupported conditional depth policy")
    if any(name not in ("arithmetic", "entailment") for name in spec["experts"]):
        raise ValueError("Unknown recurrent specialization")
    expected_files = {
        "adapters.safetensors",
        "attention-reader.json",
        "attention-reader.safetensors",
        "selector.safetensors",
    }
    for name in spec["experts"]:
        expected_files.add(name + "/adapters.safetensors")
        if name == "arithmetic":
            expected_files.update(
                {"arithmetic/deliberation.json", "arithmetic/exit-readout.safetensors"}
            )
    if set(spec["files"]) != expected_files:
        raise ValueError("Incomplete conditional checkpoint inventory")
    shared = {id(v): v for _, v in tree_flatten(model.parameters())}
    ordinary = ordinary_view(model)
    experts = {}
    for name in spec["experts"]:
        bank = deepcopy(model, shared.copy())
        if getattr(bank, "attention_reader", None) is not None:
            object.__delattr__(bank, "attention_reader")
        path = folder / name / "adapters.safetensors"
        weights = mx.load(str(path))
        expected = dict(tree_flatten(bank.trainable_parameters()))
        if weights.keys() != expected.keys() or any(
            v.shape != expected[k].shape for k, v in weights.items()
        ):
            raise ValueError("Specialist adapter differs from the shared decoder")
        bank.load_weights(list(weights.items()), strict=False)
        bank.eval()
        if name == "arithmetic":
            policy = read(path.parent / "deliberation.json")
            if policy["adapter_sha256"] != digest(path):
                raise ValueError("Wrong finisher adapter")
            attach(bank, policy, directory=path.parent)
        experts[name] = bank
    classifier = nn.Linear(2 * model.args.hidden_size, len(labels))
    classifier.load_weights(str(folder / "selector.safetensors"), strict=True)
    classifier.eval()
    object.__setattr__(
        model,
        "context_policy",
        ConditionalDepth(ordinary, experts, classifier, labels, spec["confidence"]),
    )
