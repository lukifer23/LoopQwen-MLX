"""Load an inference adapter while preserving the historical training implementation."""

from . import checkpoint
from .io import checkpoint_location, digest, read


def load_trained(root):
    generation = checkpoint_location(root)
    state = read(generation / "state.json")
    if "config" in state:
        return checkpoint.load_trained(generation)
    deployment = state["spec"]["deployment"]
    model, tokenizer, _ = checkpoint.setup(deployment["config"])
    checkpoint.restore(generation, model)
    if "backbone_mix" in deployment:
        from .backbone_mix import restore

        restore(model, deployment["backbone_mix"])
    model.alpha_override = deployment.get("alpha_override")
    model.eval()
    sidecar = generation / "attention-reader.json"
    if sidecar.is_file():
        from .attention_memory import load_reader

        if (
            digest(generation / "adapters.safetensors")
            != read(sidecar)["adapter_sha256"]
        ):
            raise ValueError("Attention reader is bound to a different adapter")
        load_reader(model, generation)
    policy = generation / "deliberation.json"
    if policy.is_file():
        from .deliberation import attach

        config = read(policy)
        if digest(generation / "adapters.safetensors") != config["adapter_sha256"]:
            raise ValueError("Deliberation policy belongs to another checkpoint")
        attach(model, config, directory=generation)
    if (generation / "context-policy.json").is_file():
        from .context_policy import attach as attach_context

        attach_context(model, generation)
    return model, tokenizer, state
