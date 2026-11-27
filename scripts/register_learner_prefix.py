"""Bind learner-generated training states, corrected continuations and the untouched test pair."""

from collections import defaultdict

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from scripts.learner_prefix_material import (
    DATA,
    DIR,
    RECEIPT,
    RECORDS,
    RESULT,
    SEED,
)
from scripts.learner_prefix_material import (
    FIT as PARENT,
)
from scripts.learner_prefix_material import (
    PLAN as CAPTURE,
)
from scripts.learner_prefix_panel import PANEL
from scripts.learner_prefix_panel import PLAN as DIRECT
from scripts.register_repair_query import EX
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

FIT = "learner-prefix-fit-01"
PRIMARY = "learner-prefix-transfer-01"
TARGETS = DIR / "supervised-tokens.json.gz"


def targets():
    for p in (DATA, RECORDS, RECEIPT, CAPTURE, PANEL, DIRECT):
        committed(p)
    receipt = read(RECEIPT)
    assert receipt["passed"] and receipt["sha256"] == digest(RECORDS)
    raw = unpack(RECORDS)
    assert [r["id"] for r in raw] == read(CAPTURE)["ordered"]
    old = {r["id"]: r for r in unpack(DATA)}
    spec = read(RESULT / PARENT / "registration.json")
    tok = tokenizer(spec)
    close = tok.convert_tokens_to_ids("</think>")
    end = tok.convert_tokens_to_ids("<|im_end|>")
    stops = {close, end, *tok.eos_token_ids}
    pairs = []

    def add(key, group, p, y, region):
        assert y and y[-1] == end and close in y
        if region == "learner":
            assert p[-1] not in stops
        pairs.append({"id": key, "group": group, "region": region, "p": p, "y": y})

    for r in raw:
        w = old[r["id"]]
        target = dict(w["target"])
        if w["group"] == "strategy":
            # Human supplied factual answers to intermediate reasoning steps, in source order.
            conclusion, facts = target["content"].split(". ", 1)
            assert conclusion in ("Yes", "No") and facts.strip()
            target["reasoning_content"] = facts.strip()
        elif w["group"] == "report_action":
            steps = target["tool_calls"][0]["function"]["arguments"]["steps"]
            target["reasoning_content"] = "Calculate " + " then ".join(steps) + "."
        p, y = token_pair(tok, {**w, "target": target})
        assert r["opening"] == fingerprint(p) and r["visits"] == 2 * len(r["tokens"])
        assert not stops.intersection(r["tokens"][:-1])
        assert r["closed"] == (r["tokens"][-1] in stops)
        add(r["id"] + "/clean", w["group"], p, y, "clean")
        # Only correct a still-open deliberation. Never insert reasoning into a visible
        # answer, submitted callback, or a completed assistant message.
        if r["closed"]:
            continue
        prefix = r["tokens"]
        if not tok.decode(prefix).strip():
            continue
        assert len(prefix) == r["limit"]
        sep = tok.encode("\n", add_special_tokens=False)
        add(r["id"] + "/learner", w["group"], p + prefix, sep + y, "learner")
    protected = (
        "instruction",
        "explanation",
        "arithmetic",
        "legacy",
        "qualitative",
        "service_action",
        "program",
        "full_rows",
    )
    for group in protected:
        choices = sorted(
            (w for w in old.values() if w["group"] == group),
            key=lambda w: fingerprint([SEED, "retain", w["id"]]),
        )
        for w in choices[:24]:
            p, y = token_pair(tok, w)
            add(w["id"] + "/retain", group, p, y, "retain")
    assert len({w["id"] for w in pairs}) == len(pairs)
    if TARGETS.exists():
        assert unpack(TARGETS) == pairs
    else:
        packed(TARGETS, pairs)
    return pairs


def register():
    pairs = targets()
    parent = RESULT / PARENT
    base = read(parent / "registration.json")
    sources = {
        str(p): digest(p)
        for p in (
            parent / "checksums.json",
            DATA,
            RECORDS,
            RECEIPT,
            TARGETS,
            CAPTURE,
            PANEL,
            DIRECT,
        )
    }
    buckets = defaultdict(list)
    for w in pairs:
        buckets[w["group"] + "/" + w["region"]].append(w["id"])
    probes = [
        k
        for group, ids in sorted(buckets.items())
        for k in sorted(ids, key=lambda i: fingerprint([SEED, "probe", i]))[:4]
    ]
    s = {
        k: base[k]
        for k in (
            "checkpoint",
            "base_text_sha256",
            "config_sha256",
            "donor",
            "passes",
            "shared_query",
            "revisits",
            "rank",
            "gate_rank",
            "compute_precision",
            "micro_positions",
        )
    }
    s.update(
        id=FIT,
        fit_seed=SEED,
        parent=PARENT,
        training=str(TARGETS),
        sources=sources,
        probes=probes,
        epochs=3,
        lr=0.00012,
        teacher_kl=0.12,
        update_positions=256,
        gate_rank=base["gate_rank"],
        expected_positions=3 * sum(len(w["y"]) for w in pairs),
        limits={"wall_seconds": 2400, "peak_gib": 3, "available_gib": 1.5},
        question="Does correcting the learner's own unfinished deliberations improve source answers and calculator actions while preserving ordinary language and the learned second read?",
        protocol="Start from the sealed trained two-read controller and update only its tied parameters through physical MLX attention. Use exact self-generated opening prefixes and annotated expert solutions. Include all clean turns, all nonempty cap-reaching prefixes and protected language turns. Three deterministic epochs, checkpoints at epoch boundaries. Cross entropy gives equal weight to target positions apart from the thought-close and end-of-turn positions, each weighted three times. Add a full-vocabulary KL to the fixed original two-read parent. Require positive mean probe NLL improvements on both clean and learner-prefix strategy and report actions, no greater than 0.05 nats average degradation across protected groups, mean second-read advantage >0.002 and exact exposure with finite gradients. Stop on failed gate or native resource cap. Only a passing fit enters the committed new pair. Neither the roll-in nor any expert text is used at inference.",
    )
    persist(EX / (FIT + ".json"), s)
    persist(
        EX / (PRIMARY + ".json"),
        {
            **{
                k: s[k]
                for k in (
                    "checkpoint",
                    "base_text_sha256",
                    "config_sha256",
                    "donor",
                    "passes",
                    "revisits",
                )
            },
            "id": PRIMARY,
            "fit_stage": FIT,
            "material": str(PANEL),
            "sources": {
                **sources,
                str(EX / (FIT + ".json")): digest(EX / (FIT + ".json")),
            },
            "limits": {"wall_seconds": 2400, "peak_gib": 3, "available_gib": 1.5},
            "seconds_per_arm": 1200,
            "max_tokens": 384,
            "max_turns": 3,
            "context_turns": 1,
            "question": read(DIRECT)["question"],
            "protocol": read(DIRECT)["design"] + " " + read(DIRECT)["gate"],
        },
    )
    print(
        len(pairs),
        "examples;",
        s["expected_positions"],
        "supervised positions across three epochs",
    )


if __name__ == "__main__":
    register()
