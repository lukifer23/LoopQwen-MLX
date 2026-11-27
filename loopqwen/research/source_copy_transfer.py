"""Measure matched continuations and free-running conversations on both pinned models."""

import gc
import re
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from ..calculator_tool import calculate, tool
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..qwen_tools import render, declarations
from ..shared_query_assistant import SharedQueryAssistant
from ..tool_compare import DenseAssistant
from ..tool_session import run as dialogue
from .conditional_primary import ARMS, costs
from .core import Budget
from .file_evidence_environment import capture
from .linked_rollin import exact_sign_flip
from .native_function_study import executions
from .registered import registered
from .report_calculation import operation_evidence
from .table_reentry_fit import root_for
from scripts.register_source_copy import FIT, PAIR, RESULT, PROTOCOL, common

# Scale of the publisher's first result. The second is a dimensionless ratio
# except for the four further monetary operations identified below.
SCALES = {
    "JPM": 6,
    "BLL": 0,
    "IPG": 6,
    "CAT": 6,
    "ETR": 6,
    "STT": 6,
    "UNP": 6,
    "HUM": None,
    "CME": 9,
    "ADI": 0,
    "ETFC": 0,
    "GPN": 3,
    "AES": 0,
    "LMT": 6,
    "MRO": 0,
    "PNC": 6,
    "PM": 6,
    "IP": 6,
    "MAS": 6,
    "AMT": 3,
    "OKE": 3,
    "HII": 6,
    "LKQ": 6,
    "GS": 6,
}
SECOND_AMOUNTS = {"GPN", "AES", "PNC", "PM"}
SUFFIXES = {"thousand": 3, "million": 6, "billion": 9, "trillion": 12}
NUM = re.compile(
    r"(?<![\w.])([+−–-]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?:\s*(%|percent(?:age points?)?|thousand|million|billion|trillion))?",
    re.I,
)


def words(text):
    return " ".join(
        w for w in re.findall(r"\w+", text.casefold()) if w not in ("the", "a", "an")
    )


def displayed(text):
    # The last sentence or line carries the delivered conclusion. Multiple
    # different values there remain ambiguous under the strict scorer.
    parts = re.split(r"(?:\n+|(?<=[.!?])\s+)", text.strip())
    for line in reversed(parts):
        nums = []
        for m in NUM.finditer(line):
            value = Fraction(
                re.sub(r"\s|,", "", m[1]).replace("−", "-").replace("–", "-")
            )
            unit = (m[2] or "").lower()
            if value > 0 and re.search(
                r"\b(?:decreas(?:e|ed)|declin(?:e|ed)|fell|reduction|lower|drop(?:ped)?|down)\b",
                line,
                re.I,
            ):
                value = -value
            nums.append((value, unit))
        if nums:
            return nums[0] if all(n == nums[0] for n in nums) else None
    return None


def finance(w, i, tr):
    ref = Fraction(w["turns"][i]["observed"]["value"])
    events = executions(tr)
    scale = SCALES[w["cluster"]] if i == 0 or w["cluster"] in SECOND_AMOUNTS else None
    values = []
    final_substantive = False
    for event in events:
        obs = event["observation"]
        final_substantive = False
        if "value" not in obs:
            continue
        steps = event["call"]["arguments"]["steps"]
        assert event["call"]["name"] == "calculate" and calculate(steps) == obs
        if any(
            operation_evidence(expr, steps[:j], set())[0]
            for j, expr in enumerate(steps)
        ):
            values.append(Fraction(obs["value"]))
            final_substantive = True
    got = displayed(tr["answer"] or "")

    def near(a, b, t):
        return abs(a - b) <= t

    ratio = scale is None
    tol = max(Fraction(1, 100), abs(ref) / 1000)
    delivered = False
    valid = False
    if got:
        value, unit = got
        if ratio:
            percent = unit == "%" or unit.startswith("percent")
            raw = value / 100 if percent else value
            tolerance = Fraction(1, 200) if percent else tol
            delivered = (percent or not unit) and near(raw, ref, tolerance)
            if final_substantive:
                val = values[-1]
                valid = near(val, ref, tolerance) or (
                    percent and near(val / 100, ref, tolerance)
                )
        else:
            power = SUFFIXES.get(unit, 0)
            raw = value * Fraction(10) ** (power - scale)
            delivered = not unit.startswith(("%", "percent")) and near(raw, ref, tol)
            if final_substantive:
                val = values[-1]
                # A calculator can operate in table units or in the displayed units.
                valid = near(val, ref, tol) or near(
                    val * Fraction(10) ** (power - scale), ref, tol
                )
    return dict(
        success=bool(tr["status"] == "answered" and delivered and valid),
        delivered=bool(delivered),
        supporting_return=bool(valid),
        executed=len(events),
        arithmetic=len(values),
        display=[str(got[0]), got[1]] if got else None,
    )


def score(w, i, tr):
    if w["family"] == "report":
        return finance(w, i, tr)
    text = words(tr["answer"] or "")
    refs = {words(a) for a in w["turns"][i]["answers"]}
    return dict(
        success=tr["status"] == "answered" and text in refs,
        reference_match=text in refs,
    )


def measure(actor, s, w, arm, guard, arm_deadline):
    source = w["body"] + "\n\n" + w["turns"][0]["question"]
    live = [dict(role="system", content=w["system"])]
    fixed, q = common(w)
    providers = [tool()] if w["family"] == "report" else []
    timing = []
    for label, h, question in (
        ("first", live, source),
        ("live", live, q),
        ("matched", fixed, q),
    ):
        mx.synchronize()
        mx.reset_peak_memory()
        start = perf_counter()
        if actor is None:
            if arm == "candidate":
                actor = SharedQueryAssistant(s["checkpoint"], RESULT / FIT)
            else:
                actor = DenseAssistant(True, "qwen", s["checkpoint"])
                assert actor.language.base_source == "certified_release_text"
        net = actor._tool_model
        n0 = (net.output_positions, net.output_revisits) if arm == "candidate" else None
        opening = actor.tokenizer.encode(
            render(
                actor.tokenizer,
                [*h, dict(role="user", content=question)],
                declarations(providers),
            ),
            add_special_tokens=False,
        )
        first = None

        def callback(count):
            nonlocal first
            if first is None and count == len(opening):
                first = perf_counter() - start
            guard.check()
            if perf_counter() > arm_deadline:
                raise InterruptedError("Arm budget")

        tr = dialogue(
            net,
            actor.tokenizer,
            question,
            providers,
            _messages=h,
            max_tokens=s["max_tokens"],
            max_turns=s["max_turns"],
            progress=callback,
            prefill_cache_state=True,
            reuse_prefix=True,
            tool_format="qwen",
            record_kv_history=True,
        )
        mx.synchronize()
        elapsed = perf_counter() - start
        assert first is not None and tr["turns"][0]["prompt_ids"] == opening
        if label == "first" and h[-1]["role"] == "user":
            # Unfinished thoughts have no delivered assistant text. Keep the empty
            # visible turn so the second human utterance follows a real failed reply.
            h.append(dict(role="assistant", content=""))
        timing.append(
            dict(
                history=label,
                trajectory=tr,
                first_s=first,
                wall_s=elapsed,
                peak=mx.get_peak_memory(),
                opening=fingerprint(opening),
                score=score(w, int(label != "first"), tr),
                history_after=deepcopy(h),
            )
        )
        if n0 is not None:
            n = net.output_positions - n0[0]
            visits = net.output_revisits - n0[1]
            assert visits == 2 * n == 2 * sum(len(t["token_ids"]) for t in tr["turns"])
            timing[-1].update(positions=n, visits=visits)
    return actor, dict(id=w["id"], family=w["family"], requests=timing)


def outcome(panel, runs):
    denominators = {}
    totals = {a: costs([q for r in runs[a] for q in r["requests"]]) for a in ARMS}
    a, b = (totals[k] for k in ARMS)
    gates = {k: a[k] < b[k] for k in ("wall_s", "first_s", "peak")}
    gates.update({k: a[k] > b[k] for k in ("opening_tps", "decode_tps")})
    for fam in ("reading", "report"):
        split = {}
        for label in ("first", "live", "matched", "all"):
            pairs = []
            done = []
            for i, w in enumerate(panel):
                if w["family"] != fam:
                    continue
                sets = [
                    [
                        q
                        for q in runs[arm][i]["requests"]
                        if q["history"] == label or label == "all"
                    ]
                    for arm in ARMS
                ]
                pairs.append(
                    [int(all(q["score"]["success"] for q in qq)) for qq in sets]
                )
                done.append(
                    [
                        int(all(q["trajectory"]["status"] == "answered" for q in qq))
                        for qq in sets
                    ]
                )
            d = [a - b for a, b in pairs]
            split[label] = dict(
                n=len(pairs),
                total=list(map(sum, zip(*pairs))),
                completed=list(map(sum, zip(*done))),
                gains=d.count(1),
                losses=d.count(-1),
                p=exact_sign_flip(d),
            )
        r = split["all"]
        gates[fam] = (
            r["gains"] - r["losses"] >= 5 and r["losses"] == 0 and r["p"] <= 0.05
        )
        gates[fam + "_completion"] = r["completed"][0] >= r["completed"][1]
        denominators[fam] = split
    return dict(
        passed=all(gates.values()), gates=gates, counts=denominators, costs=totals
    )


def verify(root, d):
    s = read(root / "registration.json")
    assert s == registered(PAIR)
    if d.get("reason"):
        assert d["passed"] is False
        return
    panel = unpack(Path(s["material"]))
    runs = {a: unpack(root / (a + ".json.gz")) for a in ARMS}
    order = read(PROTOCOL)["order"]
    for arm in ARMS:
        assert len(panel) == len(runs[arm]) == len(order)
        for w, r, spec in zip(panel, runs[arm], order, strict=True):
            assert w["id"] == r["id"] == spec["id"]
            assert [q["history"] for q in r["requests"]] == ["first", "live", "matched"]
            for i, q in enumerate(r["requests"]):
                tr = q["trajectory"]
                if i != 1:
                    assert q["opening"] == spec["first" if i == 0 else "fixed"]
                assert q["opening"] == fingerprint(tr["turns"][0]["prompt_ids"])
                assert score(w, int(i > 0), tr) == q["score"]
                if arm == "candidate":
                    assert q["visits"] == 2 * sum(
                        len(t["token_ids"]) for t in tr["turns"]
                    )
    assert outcome(panel, runs) == d


def run():
    s = registered(PAIR)
    root = root_for(s)
    parent = RESULT / FIT
    if not read(parent / "decision.json")["passed"]:
        save(
            root / "decision.json",
            dict(
                passed=False,
                reason="Registered fit prerequisite failed",
                fit=digest(parent / "checksums.json"),
            ),
        )
        return
    cold = Path("docs/measurements/source-copy-fit-01-cold.json")
    assert read(cold)["passed"] and read(cold)["fit"] == digest(
        parent / "checksums.json"
    )
    panel = unpack(Path(s["material"]))
    capture(
        root,
        (
            "loopqwen/research/source_copy_transfer.py",
            "loopqwen/source_copy.py",
            "scripts/register_source_copy.py",
            "loopqwen/tool_session.py",
        ),
    )
    save(
        root / "weights.json",
        dict(
            learned=digest(parent / "step.safetensors"),
            archive=digest(parent / "checksums.json"),
            base=s["base_text_sha256"],
            cold=digest(cold),
        ),
    )
    guard = Budget(PAIR, s["limits"])
    runs = {}
    try:
        for arm in ARMS:
            start = perf_counter()
            actor = None
            rows = []
            runs[arm] = rows
            for w in panel:
                guard.check()
                if perf_counter() - start >= s["seconds_per_arm"]:
                    raise InterruptedError("Arm budget")
                actor, r = measure(
                    actor, s, w, arm, guard, start + s["seconds_per_arm"]
                )
                rows.append(r)
                packed(root / (arm + ".json.gz"), rows)
                print(arm, len(rows), flush=True)
                if perf_counter() - start > s["seconds_per_arm"]:
                    raise InterruptedError("Arm budget")
            del actor
            gc.collect()
            mx.clear_cache()
        result = outcome(panel, runs)
    except (InterruptedError, MemoryError, TimeoutError) as e:
        result = dict(
            passed=False, reason=str(e), completed={k: len(v) for k, v in runs.items()}
        )
    finally:
        guard.close()
    save(root / "decision.json", result)


if __name__ == "__main__":
    run()
