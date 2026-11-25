"""Load an inference adapter while preserving the historical training implementation."""

from . import checkpoint
from .io import checkpoint_location, read


def load_trained(root):
    generation = checkpoint_location(root)
    state = read(generation / "state.json")
    if "config" in state:
        return checkpoint.load_trained(generation)
    deployment = state["spec"]["deployment"]
    model, tokenizer, _ = checkpoint.setup(deployment["config"])
    checkpoint.restore(generation, model)
    model.alpha_override = deployment.get("alpha_override")
    model.eval()
    return model, tokenizer, state
