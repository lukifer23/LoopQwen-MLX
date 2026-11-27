"""Train a tied two-read decoder on natural edits in two complete Spanish locale files."""

import argparse
import math
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import mlx.core as mx
import psutil
from mlx import nn, optimizers
from mlx.utils import tree_flatten

from loopqwen.bounded_read import BoundedContentRead
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.kernels.frozen_document import document_attention
from loopqwen.long_training_memory import DocumentMemory
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.shared_query_fit import cited_loss
from loopqwen.research.table_reentry_fit import apply
from loopqwen.shared_query_assistant import SharedQueryAssistant
from scripts.full_document_features import DIR
from scripts.full_document_kernel import OUT as KERNEL
from scripts.learner_prefix_material import FIT, RESULT
from scripts.long_branch_sources import CACHE, OUT as SOURCE, SPECS
from scripts.program_language_material import answer, user
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/changed-locale-fit.json")
OUT = Path("docs/measurements/changed-locale-fit")
SEED = 981337


def registration(docs):
    parent = RESULT / FIT / "checksums.json"
    for path in (parent, SOURCE, KERNEL):
        committed(path)
    assert read(KERNEL)["passed"]
    keys = [r["key"] for r in docs[0]["questions"]]
    order = sorted(range(len(keys)), key=lambda i: fingerprint([SEED, keys[i]]))
    return dict(
        seed=SEED,
        source=digest(SOURCE),
        model=digest(parent),
        kernel=digest(KERNEL),
        implementation=digest(Path(__file__)),
        train=order[:6],
        held=order[6:],
        epochs=8,
        lr=0.0004,
        limits=dict(seconds=1800, peak_gib=3, free_gib=1.5),
        protocol="Different experiment on two original Spanish CLDR upstream versions with naturally changed answers. No Deno held name or previously closed release panel. Prepare the entire native history of each document once, using all ten independently verified questions per version. Recheck the first branch after the others. Freeze the seven parent repair-query tensors, train only the two new shared bounded-read matrices. For each correct version supervise all visible answer tokens and the genuine turn terminator, the original evidence span, and a binary preference over the two real versions' first differing target token after their identical answer prefix. Loss = full-vocabulary NLL + source NLL + binary softplus loss. Each epoch covers all 12 training cases in seeded hash order. Stop at eight epochs, the resource limit, nonfinite loss, or numerical error. Save weights before any evaluation of held fields. Advance from training only if 96 updates complete, mean source NLL improves by at least .10, mean answer NLL improves by at least .03, and at least two of six training fields have positive likelihood margins on both source versions. A failed prerequisite closes evaluation. For the four previously unused field names on these same two documents require at least two correct dual-version margins, positive mean second-visit answer likelihood gain and positive mean evidence gain against initialization. Treat this as teacher-forced evidence dependence only. Free generation, product quality, generalization to unrelated files and ordinary Base comparison require separate direct evaluation. Count all native time, including both prefills.",
    )


def feature_sets(parent, tok, docs, directory, guard):
    sets = []
    for version, doc in enumerate(docs):
        raw = CACHE / SPECS["es"][3][version]
        assert digest(raw) == doc["sha256"]
        body = raw.read_bytes()
        prefix = f"Here is the complete {SPECS['es'][1]} file.\n\n{body.decode()}\nEnd of file.\n\n"
        items = []
        for q in doc["questions"]:
            p, y = token_pair(
                tok,
                dict(
                    messages=[user(prefix + q["question"])],
                    tools=[],
                    target=answer(q["answer"]),
                ),
            )
            assert (
                fingerprint(p) == q["input_sha256"]
                and body[slice(*q["source_bytes"])].decode() == q["answer"]
            )
            text = tok.decode(y)
            at = text.index(q["answer"], text.index("</think>") + 8)
            enc = tok._tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True
            )
            assert enc["input_ids"] == y
            lead = next(
                i for i, (a, b) in enumerate(enc["offset_mapping"]) if a <= at < b
            )
            items.append((p, y, lead))
        n = doc["common_prompt_tokens"]
        ids = items[0][0][:n]
        assert fingerprint(ids) == doc["common_ids"] and n > 131072
        rows = []
        with TemporaryDirectory(dir=directory) as temp:
            shared = DocumentMemory(parent, ids, temp, guard)
            lengths = [
                slot.physical_length()
                for slot in shared.disk.cache.slots.values()
                if hasattr(slot, "physical_length")
            ]
            assert lengths == [n] * 6
            for i, (p, y, lead) in enumerate([*items, items[0]]):
                guard()
                z = shared.branch(p, y)
                file = directory / f"v{version}-{i}.safetensors"
                mx.save_safetensors(
                    str(file),
                    {
                        k: z[k]
                        for k in (
                            "pre",
                            "ordinary",
                            "labels",
                            "tail_keys",
                            "tail_values",
                        )
                    },
                )
                if not i:
                    common = directory / f"v{version}-prefix.safetensors"
                    shutil.copyfile(z["prefix_file"], common)
                    assert digest(common) == digest(z["prefix_file"])
                if i == len(items):
                    assert digest(file) == rows[0]["sha256"]
                    file.unlink()
                else:
                    rows.append(
                        dict(
                            sha256=digest(file),
                            offset=z["offset"],
                            lead=lead,
                            labels=y,
                            span=doc["questions"][i]["source_tokens"],
                            opening=fingerprint(p),
                        )
                    )
        sets.append(dict(prefix=digest(common), histories=lengths, rows=rows))
        print("staged complete Spanish version", version, n, flush=True)
    return sets


def train(s, docs, folder, guard, result):
    mx.random.seed(SEED)
    actor = SharedQueryAssistant("exports/tool-language-packed-05", RESULT / FIT)
    parent = actor._tool_model
    features = feature_sets(parent, actor.tokenizer, docs, folder, guard)
    result["features"] = features
    cfg = read(RESULT / FIT / "registration.json")
    mx.random.seed(SEED)
    net = BoundedContentRead(
        parent.text, revisits=2, rank=cfg["rank"], gate_rank=cfg["gate_rank"]
    )
    net.step.load_weights(tree_flatten(parent.step.parameters()), strict=False)
    net.step.freeze()
    net.step.address.unfreeze()
    net.step.write.unfreeze()
    net.eval()
    initial = folder / "initial.safetensors"
    mx.save_safetensors(str(initial), dict(tree_flatten(net.step.parameters())))
    result["initial"] = digest(initial)
    common = [mx.load(str(folder / f"v{v}-prefix.safetensors")) for v in (0, 1)]

    def inputs(v, i):
        row = features[v]["rows"][i]
        file = folder / f"v{v}-{i}.safetensors"
        assert digest(file) == row["sha256"]
        z = mx.load(str(file))
        y = row["labels"]
        alt = features[1 - v]["rows"][i]["labels"]
        j = next(j for j, (a, b) in enumerate(zip(y, alt)) if a != b)
        lead = row["lead"]
        # A BPE token can join a leading newline to a value. Include it in the answer loss.
        assert lead <= j and y[:j] == alt[:j]
        k = mx.concatenate([common[v]["k"], z["tail_keys"]], axis=2)
        values = mx.concatenate([common[v]["v"], z["tail_values"]], axis=2)
        pos = row["offset"] + lead
        labels = z["labels"][lead:]
        assert k.shape == values.shape and k.shape[2] == pos + labels.size
        return (
            z["pre"][:, lead:],
            z["ordinary"][:, lead:],
            k,
            values,
            pos,
            labels,
            row["span"],
            j - lead,
            alt[j],
        )

    def loss(step, args, visits, backward):
        pre, h, k, v, pos, y, span, j, other = args
        query = []
        state = net.refine(
            pre,
            h,
            k,
            v,
            pos,
            "causal",
            step=step,
            visits=visits,
            differentiate=document_attention if backward else False,
            observe_content=query.append,
        )
        logits = net.project(net.text.model.norm(state)).astype(mx.float32)[0]
        ce = nn.losses.cross_entropy(logits, y, reduction="mean")
        addr = cited_loss(query[0], v, pos, [(0, [span])], net.step.dimension**-0.5)
        margin = logits[j, y[j]] - logits[j, other]
        return ce + addr + mx.logaddexp(mx.array(0.0), -margin), ce, addr, margin

    def evaluate(keys, visits=2):
        rows = []
        for i in keys:
            for v in (0, 1):
                guard()
                _, ce, addr, margin = loss(net.step, inputs(v, i), visits, False)
                rows.append(
                    dict(
                        key=i,
                        version=v,
                        nll=ce.item(),
                        address=addr.item(),
                        margin=margin.item(),
                    )
                )
        return rows

    def gain(old, new, key):
        assert [(r["key"], r["version"]) for r in old] == [
            (r["key"], r["version"]) for r in new
        ]
        return sum(a[key] - b[key] for a, b in zip(old, new, strict=True)) / len(old)

    def both(rows):
        return sum(
            all(r["margin"] > 0 for r in rows if r["key"] == i)
            for i in {r["key"] for r in rows}
        )

    before = evaluate(s["train"])
    opt = optimizers.AdamW(learning_rate=s["lr"], weight_decay=0.01)
    grad = nn.value_and_grad(net.step, lambda step, args: loss(step, args, 2, True))
    for epoch in range(s["epochs"]):
        pairs = [(v, i) for v in (0, 1) for i in s["train"]]
        for v, i in sorted(pairs, key=lambda p: fingerprint([SEED, epoch, *p])):
            guard()
            (total, ce, addr, margin), g = grad(net.step, inputs(v, i))
            norm = apply(net.step, opt, g)
            values = [total.item(), ce.item(), addr.item(), margin.item(), norm]
            if not all(map(math.isfinite, values)) or norm <= 0:
                raise ArithmeticError("nonfinite objective or invalid gradient")
            result["history"].append(
                dict(
                    epoch=epoch,
                    version=v,
                    key=i,
                    loss=values[0],
                    nll=values[1],
                    address=values[2],
                    margin=values[3],
                    gradient=norm,
                )
            )
        print("changed full document epoch", epoch + 1, flush=True)
    dest = folder / "step.safetensors"
    mx.save_safetensors(str(dest), dict(tree_flatten(net.step.parameters())))
    result["weights"] = digest(dest)
    after = evaluate(s["train"])
    result["training"] = dict(
        before=before,
        after=after,
        both=both(after),
        address_gain=gain(before, after, "address"),
        answer_gain=gain(before, after, "nll"),
    )
    result["training_passed"] = bool(
        len(result["history"]) == 96
        and both(after) >= 2
        and gain(before, after, "address") >= 0.1
        and gain(before, after, "nll") >= 0.03
    )
    if result["training_passed"]:
        net.step.load_weights(str(initial))
        baseline = evaluate(s["held"])
        net.step.load_weights(str(dest))
        one = evaluate(s["held"], 1)
        two = evaluate(s["held"])
        result["held"] = dict(
            initial=baseline,
            one=one,
            two=two,
            both=both(two),
            address_gain=gain(baseline, two, "address"),
            second_gain=gain(one, two, "nll"),
        )
        result["passed"] = (
            both(two) >= 2
            and gain(baseline, two, "address") > 0
            and gain(one, two, "nll") > 0
        )


def run(freeze):
    docs = read(SOURCE)["sources"]["es"]["documents"]
    s = registration(docs)
    if freeze:
        persist(PLAN, s)
        return
    committed(PLAN)
    assert read(PLAN) == s and not (OUT / "receipt.json").exists()
    folder = DIR / digest(PLAN)
    folder.mkdir(exist_ok=True, parents=True)
    start = monotonic()
    next_check = 0.0
    mx.reset_peak_memory()

    def guard(_=None):
        nonlocal next_check
        now = monotonic()
        if now < next_check:
            return
        next_check = now + 0.5
        if (
            now - start > s["limits"]["seconds"]
            or mx.get_peak_memory() > s["limits"]["peak_gib"] * 2**30
            or psutil.virtual_memory().available < s["limits"]["free_gib"] * 2**30
            or subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            != "1"
        ):
            raise InterruptedError("registered native resource limit")

    result = dict(plan=digest(PLAN), passed=False, training_passed=False, history=[])
    try:
        guard()
        train(s, docs, folder, guard, result)
        next_check = 0.0
        guard()
    except (ArithmeticError, InterruptedError, RuntimeError, AssertionError) as err:
        result.update(passed=False, stop=repr(err))
        if not isinstance(err, InterruptedError):
            import traceback

            result["traceback"] = traceback.format_exc()
    persist(
        OUT / "receipt.json",
        dict(**result, seconds=monotonic() - start, peak_bytes=mx.get_peak_memory()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "measure"))
    run(p.parse_args().action == "freeze")
