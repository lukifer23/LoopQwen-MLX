"""Qualify full-vocabulary supervised updates on the exact document memory."""

import argparse
import math
import subprocess
from pathlib import Path
from time import monotonic

import mlx.core as mx
import psutil
from mlx import nn, optimizers
from mlx.utils import tree_flatten

from loopqwen.content_read import ContentRead
from loopqwen.frozen_memory_attention import frozen_attention
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.shared_query_fit import cited_loss
from loopqwen.research.table_reentry_fit import apply
from loopqwen.shared_query_assistant import SharedQueryAssistant
from scripts.full_document_features import DIR, OUT as FEATURES, PLAN as EXTRACTION
from scripts.learner_prefix_material import FIT, RESULT
from scripts.long_branch_sources import OUT as SOURCES
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/full-document-adjoint.json")
OUT = Path("docs/measurements/full-document-adjoint.json")
SEED = 973111


def spec():
    for p in (FEATURES, EXTRACTION, SOURCES, RESULT / FIT / "checksums.json"):
        committed(p)
    features = read(FEATURES)
    assert features["passed"] and len(features["rows"]) == 12
    order = sorted(
        range(12), key=lambda i: fingerprint([SEED, features["rows"][i]["key"]])
    )
    return dict(
        seed=SEED,
        features=digest(FEATURES),
        model=digest(RESULT / FIT / "checksums.json"),
        doc=features["document_kv"],
        implementation={
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path("loopqwen/frozen_memory_attention.py"),
                Path("loopqwen/content_read.py"),
                Path("loopqwen/shared_query.py"),
            )
        },
        train=order[:8],
        held=order[8:],
        epochs=8,
        lr=0.0007,
        weight=1.0,
        limits=dict(seconds=900, peak_gib=3, available_gib=1.5),
        question="Can native gradients learn to address an unchanged full-length document and predict complete source versions, including for different questions about the same document?",
        protocol="Numerical and training feasibility only. Require passing complete-prefix extraction and the independent grouped and causal derivative parity test. Use the frozen original two-read repair donor; train only two shared content matrices. Keep all 147432 document entries at native precisions. Use every visible version token and the turn terminator under full causal teacher forcing, plus log probability of its original source span at the first such decision. Fit exactly 8 deterministic epochs of 8 independent package questions. Four disjoint package names on the same unchanged file are withheld until the weights are sealed. Test all 12 in immutable order. Require all 64 finite nonzero updates, mean training source NLL gain >=0.1 and target NLL gain >0.02; separately require >=3/4 held improvements in target NLL and positive mean held address gain. No free-generation, semantic accuracy or release credit. Do not rerun this panel after a missed gate. Terminate at the resource cap.",
    )


def run(register):
    s = spec()
    if register:
        persist(PLAN, s)
        return
    committed(PLAN)
    assert s == read(PLAN) and not OUT.exists()
    start = monotonic()
    next_check = 0

    def guard():
        nonlocal next_check
        now = monotonic()
        if now < next_check:
            return
        next_check = now + 0.75
        if (
            now - start > s["limits"]["seconds"]
            or mx.get_peak_memory() > s["limits"]["peak_gib"] * 2**30
            or psutil.virtual_memory().available < 1.5 * 2**30
            or subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            != "1"
        ):
            raise RuntimeError("Full-document update resource limit")

    guard()
    mx.reset_peak_memory()
    actor = SharedQueryAssistant("exports/tool-language-packed-05", RESULT / FIT)
    parent = actor._tool_model
    tok = actor.tokenizer
    mx.random.seed(SEED)
    pars = read(RESULT / FIT / "registration.json")
    net = ContentRead(
        parent.text, revisits=2, rank=pars["rank"], gate_rank=pars["gate_rank"]
    )
    net.eval()
    net.step.load_weights(tree_flatten(parent.step.parameters()), strict=False)
    net.step.freeze()
    net.step.address.unfreeze()
    net.step.write.unfreeze()
    proof = read(FEATURES)
    folder = DIR / digest(EXTRACTION)
    prefix = folder / "document.safetensors"
    assert digest(prefix) == s["doc"]
    mem = mx.load(str(prefix))
    src = read(SOURCES)["sources"]["deno"]["documents"][1]["questions"]

    def inputs(i):
        r = proof["rows"][i]
        file = folder / (str(i) + ".safetensors")
        assert digest(file) == r["sha256"] and src[i]["key"] == r["key"]
        z = mx.load(str(file))
        ids = z["labels"].tolist()
        txt = tok.decode(ids)
        val = src[i]["answer"]
        out = tok._tokenizer(txt, add_special_tokens=False, return_offsets_mapping=True)
        assert out["input_ids"] == ids
        at = txt.index(val, txt.index("</think>") + 8)
        lead = next(j for j, (a, b) in enumerate(out["offset_mapping"]) if a <= at < b)
        k = mx.concatenate([mem["k"], z["tail_keys"]], axis=2)
        v = mx.concatenate([mem["v"], z["tail_values"]], axis=2)
        assert k.shape == v.shape and k.shape[2] == r["offset"] + len(ids)
        return (
            z["pre"][:, lead:],
            z["ordinary"][:, lead:],
            k,
            v,
            r["offset"] + lead,
            z["labels"][lead:],
            r["source"],
        )

    def loss(step, args, backward):
        pre, h, k, v, pos, y, cite = args
        qq = []
        z = net.refine(
            pre,
            h,
            k,
            v,
            pos,
            "causal",
            step=step,
            differentiate=frozen_attention if backward else False,
            observe_content=qq.append,
        )
        logits = net.project(net.text.model.norm(z)).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, y[None, :], reduction="mean")
        address = cited_loss(qq[0], v, pos, [(0, [cite])], net.step.dimension**-0.5)
        return ce + s["weight"] * address, ce, address

    def measure(indices):
        data = {}
        for i in indices:
            guard()
            args = inputs(i)
            _, ce, ad = loss(net.step, args, False)
            data[str(i)] = dict(nll=ce.item(), address=ad.item(), labels=args[-2].size)
        return data

    history = []
    detail = {}
    dest = DIR / digest(PLAN)
    dest.mkdir(exist_ok=True, parents=True)
    try:
        detail["before_train"] = measure(s["train"])
        initial = dest / "initial.safetensors"
        mx.save_safetensors(str(initial), dict(tree_flatten(net.step.parameters())))
        opt = optimizers.AdamW(learning_rate=s["lr"], weight_decay=0.01)
        div = nn.value_and_grad(net.step, lambda step, args: loss(step, args, True))
        for epoch in range(s["epochs"]):
            order = sorted(s["train"], key=lambda i: fingerprint([SEED, epoch, i]))
            for i in order:
                guard()
                args = inputs(i)
                (total, ce, ad), g = div(net.step, args)
                norm = apply(net.step, opt, g)
                vals = [total.item(), ce.item(), ad.item(), norm]
                assert all(math.isfinite(x) for x in vals) and norm > 0
                history.append(
                    dict(
                        epoch=epoch,
                        index=i,
                        gradient=norm,
                        loss=vals[0],
                        nll=vals[1],
                        address=vals[2],
                    )
                )
                del args, g
            print("full-length updates", len(history), flush=True)
        trained = dest / "step.safetensors"
        mx.save_safetensors(str(trained), dict(tree_flatten(net.step.parameters())))
        identity = digest(trained)
        detail["after"] = measure(range(12))
        net.step.load_weights(str(initial), strict=True)
        detail["before_held"] = measure(s["held"])

        def gains(which, metric):
            base = detail["before_" + which]
            return [v[metric] - detail["after"][i][metric] for i, v in base.items()]

        from statistics import mean

        a = gains("held", "nll")
        gates = dict(
            updates=len(history) == 64,
            train_target=mean(gains("train", "nll")) > 0.02,
            train_address=mean(gains("train", "address")) >= 0.1,
            held_target=sum(x > 0 for x in a) >= 3,
            held_address=mean(gains("held", "address")) > 0,
        )
        result = dict(passed=all(gates.values()), gates=gates, weights=identity)
    except RuntimeError as err:
        if str(err) != "Full-document update resource limit":
            raise
        result = dict(passed=False, reason=str(err))
    persist(
        OUT,
        dict(
            **result,
            plan=digest(PLAN),
            history=history,
            detail=detail,
            seconds=monotonic() - start,
            peak_bytes=mx.get_peak_memory(),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "measure"))
    run(parser.parse_args().action == "register")
