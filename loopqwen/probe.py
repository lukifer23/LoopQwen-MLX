from dataclasses import replace
from time import perf_counter

import mlx.core as mx
from mlx.utils import tree_flatten

from .evaluation.loss_eval import language_loss
from .io import save
from .looped_model import LoopedModel
from .model_loader import count, load_base


def probe(span):
    base, tok, info = load_base()
    x = mx.array(
        [tok.encode("At the end of the experiment we measured the final mass.")]
    )
    wrapper = LoopedModel(base, replace(span, passes=1))
    wrapper.eval()
    parity = mx.max(mx.abs(base(x) - wrapper(x))).item()
    if parity != 0:
        raise AssertionError(f"K=1 parity failed: {parity}")
    original = {id(a) for _, a in tree_flatten(base.parameters())}
    rows = []
    for k in dict.fromkeys([1, span.passes]):
        wrapper.span = replace(span, passes=k)
        mx.eval(wrapper(x))
        t = perf_counter()
        for _ in range(5):
            mx.eval(wrapper(x))
        rows.append(
            dict(
                k=k,
                executions=len(wrapper.sites),
                full_forward_ms=(perf_counter() - t) * 200,
                loss=language_loss(wrapper, tok),
            )
        )
    wrapped = {id(a) for _, a in tree_flatten(wrapper.parameters())}
    gates = int(span.mode == "stabilized")
    assert original <= wrapped and len(wrapped - original) == gates
    result = dict(
        unique=count(base),
        wrapped_unique=count(wrapper),
        added_scalars=gates,
        adapter_parameters=0,
        k1_max_abs=parity,
        sites=wrapper.sites,
        shared_pretrained_tensors=len(original),
        rows=rows,
    )
    save("outputs/probe.json", result)
    return result
