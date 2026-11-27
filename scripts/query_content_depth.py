"""Measure exactly which generated training tokens benefit from the second physical read."""

import argparse
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic

import mlx.core as mx
import psutil

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.conditional_features import feature_root
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.shared_query_fit import parts
from loopqwen.shared_query_assistant import SharedQueryAssistant
from scripts.assistance_deliberation import write_csv
from scripts.register_repair_query import FIT, RESULT
from scripts.shared_query_evidence import offsets
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/query-content-depth.json")
REPORT = Path("docs/measurements/query-content-depth.json")


def channels(tok, ids, target):
    text, ranges = offsets(tok, ids)
    stop = text.index("</think>")
    labels = ["format"] * len(text)
    labels[:stop] = ["reasoning"] * stop
    answer = target.get("content", "").strip()
    if answer:
        start = text.index(answer, stop + len("</think>"))
        labels[start : start + len(answer)] = ["visible"] * len(answer)
    for m in re.finditer(r"<function=([^>]+)>", text):
        i, j = m.span(1)
        labels[i:j] = ["function name"] * (j - i)
    for m in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", text, re.DOTALL):
        for idx, key in ((1, "argument name"), (2, "argument value")):
            i, j = m.span(idx)
            labels[i:j] = [key] * (j - i)
    classes = []
    for i, j in ranges:
        kinds = {labels[k] for k in range(i, j) if not text[k].isspace()}
        if not kinds:
            kind = "whitespace"
        elif len(kinds) > 1:
            kind = "mixed boundary"
        else:
            kind = kinds.pop()
        classes.append(kind)
    assert len(classes) == len(ids)
    return classes


def material():
    folder = RESULT / FIT
    spec = read(folder / "registration.json")
    fs = read(RESULT / spec["feature_stage"] / "registration.json")
    tok = tokenizer(spec)
    training = {r["id"]: r for r in unpack(Path(spec["training"]))}
    probes = read(folder / "probes.json")
    index = {
        r["id"]: r
        for r in unpack(RESULT / spec["feature_stage"] / "feature-index.json.gz")
    }
    cases = {}
    for p in probes["after"]:
        k = p["id"]
        t = training[k]
        _, y = token_pair(tok, t)
        labels = channels(tok, y, t["target"])
        assert fingerprint(y) == index[k]["labels"] and len(labels) == p["positions"]
        cases[k] = labels
    plan = {
        "archive": digest(folder / "checksums.json"),
        "probes": digest(folder / "probes.json"),
        "feature_archive": digest(RESULT / spec["feature_stage"] / "checksums.json"),
        "code": digest(Path(__file__)),
        "labels_sha256": fingerprint(cases),
        "positions": dict(Counter(v for row in cases.values() for v in row)),
        "protocol": "Use all 120 preselected training probes. At the exact same saved weights score every natural target token with either one or two physical reads. Preserve causal memory, logits and gold IDs. Freeze token classification from the original rendered suffix and offsets before inference. Non-whitespace characters crossing a role boundary are mixed. Separate reasoning-channel content, visible answers, function names, parameter names, parameter values, syntax and whitespace. Report target-token NLL and actual greedy top-1 token accuracy, with one-to-two gains and losses for every class. Require all 5415 positions and all archived complete-example losses within 0.00002 nats for a valid diagnostic. Credit likelihood support beyond syntax only if reasoning, visible answers and tool argument values each improve. This is a within-training local intervention; it cannot promote a checkpoint or change product gates.",
        "limits": {"seconds": 120, "peak_gib": 2.5, "available_gib": 1.5},
    }
    return spec, fs, probes, index, cases, plan


def run(freeze):
    spec, fs, probes, index, cases, plan = material()
    if freeze:
        committed(RESULT / FIT / "checksums.json")
        persist(PLAN, plan)
        return
    committed(PLAN)
    assert read(PLAN) == plan
    start = monotonic()
    mx.reset_peak_memory()
    net = SharedQueryAssistant(spec["checkpoint"], RESULT / FIT)._tool_model
    folder = feature_root(fs)
    raw = []
    groups = defaultdict(Counter)
    maxerr = 0.0
    latest = 0
    for one, two in zip(probes["first_visit"], probes["after"], strict=True):
        k = two["id"]
        assert k == one["id"]
        r = index[k]
        where = 0
        sums = [0.0, 0.0]
        assert digest(folder / r["file"]) == r["sha256"]
        for pre, h, keys, vals, pos, gold, _ in parts(r, folder, {}, spec):
            if monotonic() - latest > 1:
                latest = monotonic()
                if (
                    latest - start > 120
                    or mx.get_peak_memory() > 2.5 * 2**30
                    or psutil.virtual_memory().available < 1.5 * 2**30
                    or subprocess.check_output(
                        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                        text=True,
                    ).strip()
                    != "1"
                ):
                    raise RuntimeError(
                        "Native content probe exceeded its registered resource limit"
                    )
            metrics = []
            for j in (1, 2):
                z = net.refine(
                    pre,
                    h,
                    keys,
                    vals,
                    pos,
                    "causal" if h.shape[1] > 1 else None,
                    visits=j,
                )
                logits = net.project(net.text.model.norm(z)).astype(mx.float32)[0]
                loss = (
                    mx.logsumexp(logits, axis=-1)
                    - mx.take_along_axis(logits, gold[:, None], axis=-1)[:, 0]
                )
                prediction = mx.argmax(logits, axis=-1)
                mx.eval(loss, prediction)
                metrics.append((loss.tolist(), prediction.tolist()))
            labels = gold.tolist()
            for i, y in enumerate(labels):
                n = where + i
                c = cases[k][n]
                a, b = [(m[0][i], m[1][i]) for m in metrics]
                raw.append(
                    {
                        "example": k,
                        "group": r["group"],
                        "position": n,
                        "kind": c,
                        "target_id": y,
                        "one_id": a[1],
                        "two_id": b[1],
                        "one_nll": a[0],
                        "two_nll": b[0],
                    }
                )
                g = groups[c]
                g["n"] += 1
                g["one_correct"] += a[1] == y
                g["two_correct"] += b[1] == y
                g["gained"] += a[1] != y and b[1] == y
                g["lost"] += a[1] == y and b[1] != y
                g["one_nll"] += a[0]
                g["two_nll"] += b[0]
                sums[0] += a[0]
                sums[1] += b[0]
            where += len(labels)
        assert where == len(cases[k])
        maxerr = max(
            maxerr, abs(sums[0] / where - one["ce"]), abs(sums[1] / where - two["ce"])
        )
    assert len(raw) == 5415 and maxerr <= 0.00002
    for v in groups.values():
        v["mean_nll_gain"] = (v["one_nll"] - v["two_nll"]) / v["n"]
    write_csv(Path("paper/query-content-depth.csv"), raw)
    persist(
        REPORT,
        {
            "plan": digest(PLAN),
            "positions": len(raw),
            "groups": groups,
            "max_mean_error": maxerr,
            "beyond_syntax": all(
                groups[k]["mean_nll_gain"] > 0
                for k in ("reasoning", "visible", "argument value")
            ),
            "seconds": monotonic() - start,
            "peak_bytes": mx.get_peak_memory(),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("freeze", "measure"))
    run(p.parse_args().mode == "freeze")
