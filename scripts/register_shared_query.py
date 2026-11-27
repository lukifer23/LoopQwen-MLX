"""Precommit the tied-query learner, direct comparison and source-visible review."""

from pathlib import Path

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.qwen_tools import render
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.shared_query_features import FEATURES, FIT, OLD, PRIMARY
from scripts.contrastive_transfer_panel import PANEL as PREVIOUS
from scripts.shared_query_material import ROOT, TOOLS
from scripts.table_policy_material import persist

PANEL = ROOT / "transfer-panel.json.gz"
PLAN = Path("research/diagnostics/shared-query-reading.json")
EX = Path("research/experiments")


def build():
    source = read(ROOT / "source-inventory.json")
    ev = read(ROOT / "evidence-inventory.json")
    old = EX / "conditional-program-fit-01.json"
    existing = read(old)
    tok = tokenizer(existing)
    rows = []
    for w in unpack(ROOT / "held.json.gz"):
        question = w["request"]
        defs = TOOLS if w["family"] == "report" else []
        p = tok.encode(
            render(tok, [{"role": "user", "content": question}], defs),
            add_special_tokens=False,
        )
        rows.append(
            {
                **w,
                "questions": [question],
                "defs": defs,
                "initial_ids": p,
                "initial_sha256": fingerprint(p),
            }
        )
    long = [w for w in unpack(PREVIOUS)["rows"] if w["family"] == "locale"]
    rows.extend(long)
    rows.sort(key=lambda w: fingerprint([864371, "primary", w["id"]]))
    assert len(rows) == 84 and len(long) == 4 and len({w["id"] for w in rows}) == 84
    material = {
        "rows": rows,
        "sources": {str(x): digest(x) for x in (PREVIOUS, ROOT / "held.json.gz")},
    }
    if PANEL.exists():
        assert unpack(PANEL) == material
    else:
        packed(PANEL, material)
    plan = {
        "material": digest(PANEL),
        "scope": "All 48 multi-step questions and 32 financial-report questions. Every full visible output is graded before totals or identities are opened. The project coding assistant is a single reviewer.",
        "rating": "On strategy questions, require the correct conclusion plus a reasonable explanation grounded in the actual supplied passages. A naked yes or no is partial. Accept a qualified answer if the passage cannot establish the human premise. On reports require an unambiguous supported amount with the relevant units and sign. For a grounded usable report, the answer must also follow a genuine successful calculator execution of substantive arithmetic. Read original reports, authored decompositions and facts, expert programs and executed observations. Mark omissions and unsupported additional claims.",
        "gate": "On each family require at least six net additional fully usable answers, two-sided exact paired p at most 0.05, zero losses of fully usable items, and no increase in unsupported answers. Report whole source pairs, all individual losses and ties. Completing the matched-depth and independent-seed comparisons and the retained broad qualification remains necessary after a pass.",
    }
    persist(PLAN, plan)
    common = {
        k: existing[k]
        for k in ("checkpoint", "base_text_sha256", "config_sha256", "donor", "passes")
    }
    paths = [
        old,
        OLD / "backbone.json",
        OLD / "checksums.json",
        EX / "contrastive-world-fit-01.json",
        EX / "contrastive-world-transfer-01.json",
        *[
            ROOT / x
            for x in (
                "training.json.gz",
                "held.json.gz",
                "source-inventory.json",
                "independent-replay.json",
                "evidence.json.gz",
                "evidence-inventory.json",
            )
        ],
        PANEL,
        PLAN,
    ]
    common.update(
        shared_query=True,
        revisits=2,
        rank=160,
        gate_rank=32,
        fit_seed=864371,
        training=str(ROOT / "training.json.gz"),
        evidence=str(ROOT / "evidence.json.gz"),
        sources={str(p): digest(p) for p in paths},
        limits={"wall_seconds": 4380, "peak_gib": 8.25, "available_gib": 1.5},
    )
    opening = sum(x["opening"] for x in source["groups"].values())
    targets = sum(x["target"] for x in source["groups"].values())
    feature = {
        **common,
        "id": FEATURES,
        "examples": 3136,
        "target_positions": targets,
        "opening_positions": opening,
        "question": "Can all ordinary Qwen positions and their full final attention memories be exposed for differentiable repeated source queries without losing cached-token parity?",
        "protocol": "Run only after the preceding increment-based chain is archived and misses qualification. A completed primary must receive its blinded review first. Test three hash-selected original prompts for unmodified and split-cache top-token agreement, maximum logit errors <=0.55, 24 unique memory writers and six full attention histories. Require every original token label, degree of exposure and SHA-256. Stop on a missed prerequisite or any registered resource limit.",
    }
    persist(EX / (FEATURES + ".json"), feature)
    fit = {
        **common,
        "id": FIT,
        "epochs": 3,
        "lr": 0.00022,
        "teacher_kl": 0.24,
        "evidence_weight": 8.0,
        "micro_positions": 96,
        "update_positions": 512,
        "expected_positions": targets * 3,
        "expected_citations": sum(ev["queries"].values()) * 3,
        "question": "Can gradients through actual tied attention reads improve the predicted answer and assign more probability to human-cited source evidence?",
        "protocol": "Train all 3136 complete targets on three deterministic shuffled passes under native MLX. No test targets enter fitting. The full next-token likelihood, forward KL from the ordinary frozen instruction sibling and negative log probability assigned to cited source tokens are measured independently. Apply one exact loss per token and eight token-equivalents per annotated query. Report every update and four fixed training probes per group. Require finite positive gradients, all specified exposure, mean per-group NLL improvement >=0.04 nats, both new reasoning and report-action improvements, mean additional NLL gain >0.002 from the second read relative to the first at unchanged learned weights, and >0.02 nats better citation mass. Failure closes direct evaluation for those weights.",
    }
    fit["sources"] = {
        **fit["sources"],
        str(EX / (FEATURES + ".json")): digest(EX / (FEATURES + ".json")),
    }
    persist(EX / (FIT + ".json"), fit)
    primary = {
        **common,
        "id": PRIMARY,
        "material": str(PANEL),
        "max_tokens": 384,
        "max_turns": 3,
        "context_turns": 1,
        "question": "Does a trained tied output query improve real calculations and source-based multi-step reasoning and preserve full 128K-plus processing against the pinned untouched ordinary Base?",
        "protocol": "One frozen loop/Base pair on 84 ordered prompts with identical official Qwen thinking-enabled tokens, tools, greedy policy and limits. The 32 report and 48 qualitative requests each have a different issuer or topic within their family. The four unabridged locale files have evidence after token 131072. Require >=6 net gains and zero losses per short family and exact two-sided p<=0.05. Require every long result correct, a strictly higher long score than Base, no family decrease in completion, lower peak, total first-token and wall times, faster opening and decode. All 80 short responses have an additional preregistered blinded source-based assessment. Only a joint pass qualifies the broad retained comparison, independent seed, equal-exposure ordinary-depth training, and final clean installation.",
    }
    primary["sources"] = {
        **primary["sources"],
        str(EX / (FIT + ".json")): digest(EX / (FIT + ".json")),
    }
    persist(EX / (PRIMARY + ".json"), primary)
    print(
        "Frozen", len(rows), "source requests and", targets * 3, "training positions."
    )


if __name__ == "__main__":
    build()
