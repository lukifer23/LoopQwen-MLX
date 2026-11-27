"""Measure one complete native query derivative on a previously charged training example."""

import argparse
from pathlib import Path
from time import monotonic

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from loopqwen.content_read import ContentRead
from loopqwen.io import read, digest
from loopqwen.kernels.frozen_document import document_attention
from loopqwen.research.shared_query_fit import cited_loss
from loopqwen.shared_query_assistant import SharedQueryAssistant
from scripts import full_document_adjoint as prior
from scripts.full_document_features import DIR, OUT as FEATURES, PLAN as EXTRACTION
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/full-document-kernel.json")
OUT = Path("docs/measurements/full-document-kernel.json")


def run(freeze):
    for p in (prior.OUT, FEATURES, prior.PLAN):
        committed(p)
    old = read(prior.OUT)
    (ref,) = old["history"]
    assert not old["passed"] and ref["index"] in read(prior.PLAN)["train"]
    src = Path("loopqwen/kernels/frozen_document.py")
    s = dict(
        source=digest(src),
        script=digest(Path(__file__)),
        prior=digest(prior.OUT),
        features=digest(FEATURES),
        index=ref["index"],
        protocol="Engineering only. Use the first and only completed update of the closed full-document run as a reference before optimization. Same source version, whole document, initialization, supervised labels, marked span, two native visits and trainable matrices. Compute one full gradient with the Metal per-partition softmax moments, after passing dense nested two-visit forward/derivative tests in FP32 and FP16 for single and causal multi-query suffixes. Compare the gradient norm, unmodified target likelihood and evidence likelihood. This task does not fit weights and does not evaluate held names. Stop if more than 120 seconds or 3 GiB, check system availability before and after the operation. A qualifying implementation must achieve under 3 GiB peak, absolute NLL differences below .003, finite positive gradients in aggregate and relative norm error under 1%. Separate training on new documents requires a new registration.",
    )
    if freeze:
        persist(PLAN, s)
        return
    committed(PLAN)
    assert read(PLAN) == s and not OUT.exists()
    import psutil
    from subprocess import check_output

    def safe():
        return (
            psutil.virtual_memory().available >= 1.5 * 2**30
            and check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            == "1"
        )

    assert safe()
    mx.reset_peak_memory()
    start = monotonic()
    actor = SharedQueryAssistant(
        "exports/tool-language-packed-05", prior.RESULT / prior.FIT
    )
    init = DIR / digest(prior.PLAN) / "initial.safetensors"
    pars = read(prior.RESULT / prior.FIT / "registration.json")
    net = ContentRead(
        actor._tool_model.text,
        revisits=2,
        rank=pars["rank"],
        gate_rank=pars["gate_rank"],
    )
    net.step.load_weights(str(init), strict=True)
    net.step.freeze()
    net.step.address.unfreeze()
    net.step.write.unfreeze()
    net.eval()
    folder = DIR / digest(EXTRACTION)
    source = read(FEATURES)
    i = ref["index"]
    row = source["rows"][i]
    path = folder / (str(i) + ".safetensors")
    full = folder / "document.safetensors"
    assert digest(path) == row["sha256"] and digest(full) == source["document_kv"]
    z = mx.load(str(path))
    common = mx.load(str(full))
    k = mx.concatenate([common["k"], z["tail_keys"]], axis=2)
    v = mx.concatenate([common["v"], z["tail_values"]], axis=2)
    labels = ref_labels = old["detail"]["before_train"][str(i)]["labels"]
    lead = z["labels"].size - ref_labels
    pos = row["offset"] + lead
    pre = z["pre"][:, lead:]
    h = z["ordinary"][:, lead:]
    y = z["labels"][lead:]
    assert (
        labels == y.size and k.shape[2] == v.shape[2] == pos + labels and pos > 131072
    )

    def loss(step):
        qq = []
        zz = net.refine(
            pre,
            h,
            k,
            v,
            pos,
            "causal",
            step=step,
            differentiate=document_attention,
            observe_content=qq.append,
        )
        logits = net.project(net.text.model.norm(zz)).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, y[None, :], reduction="mean")
        addr = cited_loss(
            qq[0], v, pos, [(0, [row["source"]])], net.step.dimension**-0.5
        )
        return ce + addr, ce, addr

    (_, ce, addr), g = nn.value_and_grad(net.step, loss)(net.step)
    norm = mx.sqrt(
        sum(mx.sum(t.astype(mx.float32).square()) for _, t in tree_flatten(g))
    ).item()
    elapsed = monotonic() - start
    peak = mx.get_peak_memory()
    gates = dict(
        norm=norm > 0 and abs(norm / ref["gradient"] - 1) < 0.01,
        nll=abs(ce.item() - ref["nll"]) < 0.003,
        address=abs(addr.item() - ref["address"]) < 0.003,
        peak=peak < 3 * 2**30,
        time=elapsed < 120,
        system=safe(),
    )
    persist(
        OUT,
        dict(
            passed=all(gates.values()),
            gates=gates,
            plan=digest(PLAN),
            initial=digest(init),
            index=i,
            positions=labels,
            context=k.shape[2],
            seconds=elapsed,
            peak_bytes=peak,
            gradient=norm,
            nll=ce.item(),
            address=addr.item(),
            prior_peak=old["peak_bytes"],
            reference=ref,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "measure"))
    run(p.parse_args().action == "freeze")
