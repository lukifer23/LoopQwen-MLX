"""Native weight-only quantization for long recurrent requests after route selection."""

from copy import deepcopy
from time import perf_counter

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear

from .looped_model import LoopedModel


class RegisterView(LoopedModel):
    def project(self, hidden):
        object.__setattr__(self, "projection_calls", self.projection_calls + 1)
        return self.output_embedding.as_linear(hidden)


class RegisterPolicy:
    def __init__(self, original, view, receipt, minimum):
        self.original = original
        self.view = view
        self.receipt = receipt
        self.minimum = minimum

    def choose(self, model, ids, cap):
        selected, route = self.original.choose(model, ids, cap)
        if selected is model and route == "register" and len(ids) >= self.minimum:
            return self.view, route
        return selected, route

    def views(self):
        return self.original.views()


def build_view(model):
    if not model.args.tie_word_embeddings:
        raise ValueError("The original output head must use the input embedding")
    start = perf_counter()
    mx.synchronize()
    mx.reset_peak_memory()
    shared = {id(a): a for _, a in tree_flatten(model.text.parameters())}
    text = deepcopy(model.text, shared)
    bits = 8
    for layer in text.layers:
        nn.quantize(layer.mlp, group_size=64, bits=bits)
        mx.eval(layer.mlp.parameters())
    view = RegisterView(text, model.span)
    if model.span.has_gate:
        view.alpha_logit = model.alpha_logit
    view.alpha_override = model.alpha_override
    view.output_embedding = text.model.embed_tokens.to_quantized(
        group_size=64, bits=bits
    )
    view.eval()
    mx.eval(view.parameters())
    mx.clear_cache()
    if hasattr(model, "attention_reader"):
        object.__setattr__(view, "attention_reader", model.attention_reader)
    for key in ("packed_register_prefill", "fused_packed_decode"):
        object.__setattr__(view, key, getattr(model, key, False))
    object.__setattr__(view, "projection_calls", 0)
    paths = []
    adapted_paths = []
    for i, (a, b) in enumerate(zip(text.layers, model.text.layers, strict=True)):
        for name in ("gate_proj", "up_proj", "down_proj"):
            compact = getattr(a.mlp, name)
            source = getattr(b.mlp, name)
            path = f"layers.{i}.mlp.{name}"
            if isinstance(source, LoRALinear):
                if not (
                    isinstance(compact, LoRALinear)
                    and compact.lora_a is source.lora_a
                    and compact.lora_b is source.lora_b
                    and compact.scale == source.scale
                ):
                    raise ValueError("The original low-rank update must be shared")
                compact, source = compact.linear, source.linear
                adapted_paths.append(path)
            if not isinstance(compact, nn.QuantizedLinear) or not isinstance(
                source, nn.Linear
            ):
                raise TypeError(
                    "Only the new register view may convert the original feed-forward weights"
                )
            paths.append(path)
    converted = {
        id(a): a.nbytes
        for _, a in tree_flatten(view.parameters())
        if id(a) not in shared and a is not getattr(model, "alpha_logit", None)
    }
    receipt = {
        "bits": bits,
        "group_size": 64,
        "projections": paths,
        "shared_adapter_projections": adapted_paths,
        "shared_input_embedding": text.model.embed_tokens.weight
        is model.text.model.embed_tokens.weight,
        "new_resident_tensor_bytes": sum(converted.values()),
        "setup_s": perf_counter() - start,
        "setup_peak": mx.get_peak_memory(),
    }
    return view, receipt


def install(model, *, minimum=32768):
    policy = getattr(model, "context_policy", None)
    if policy is None or isinstance(policy, RegisterPolicy):
        raise ValueError("This precision policy needs the original conditional decoder")
    view, receipt = build_view(model)
    object.__setattr__(
        model, "context_policy", RegisterPolicy(policy, view, receipt, minimum)
    )
