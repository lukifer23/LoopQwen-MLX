"""Register the evidence-sensitive learner and its single held direct comparison."""

from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from loopqwen.research.contrastive_features import FEATURES, FIT
from scripts.contrastive_transfer_panel import PANEL, PLAN
from scripts.contrastive_world_material import OUT
from scripts.table_policy_material import persist

PRIMARY = "contrastive-world-transfer-01"
EX = Path("research/experiments")
QUAL = OUT / "qualification.json"
OLD = Path("research/results/conditional-features-01")
REG = EX / "conditional-program-fit-01.json"


def build():
    prev = read(REG)
    common = {
        k: prev[k]
        for k in (
            "checkpoint",
            "base_text_sha256",
            "config_sha256",
            "donor",
            "conditional_reentry",
            "passes",
            "rank",
            "gate_rank",
        )
    }
    q = read(QUAL)
    common.update(
        fit_seed=781345,
        increment_readout=True,
        training=str(OUT / "training.json.gz"),
        sources={
            str(p): digest(p)
            for p in (
                QUAL,
                OUT / "training.json.gz",
                OUT / "withheld.json.gz",
                OUT / "independent-replay.json",
                OLD / "checksums.json",
                OLD / "backbone.json",
                REG,
                EX / "conditional-program-primary-01.json",
                PANEL,
                PLAN,
            )
        },
        limits={"wall_seconds": 4380, "available_gib": 1.5, "peak_gib": 8.25},
    )
    features = []
    for name, groups in zip(
        FEATURES,
        (
            ["operator", "inspect", "resolved", "full_rows"],
            ["qualitative", "service_action", "service_reply"],
        ),
        strict=True,
    ):
        expected = {
            k: sum(q["groups"][g].get(m, 0) for g in groups)
            for k, m in (
                ("examples", "turns"),
                ("positions", "response_positions"),
                ("alternatives", "preferences"),
                ("negative_positions", "rejected_positions"),
            )
        }
        feature = {
            **common,
            "id": name,
            "groups": groups,
            "expected": expected,
            "question": "Can the native second visit expose every causal state and every agreed changed answer in the reserved "
            + ", ".join(groups)
            + " training sources?",
            "protocol": "Start only if the completed preceding 340-request study misses its joint automatic and masked gate. Keep its weights closed. Use the same immutable four-bit instruction backbone with separate physical histories on each visit. Require initial ordinary logits, cached parity, all 28 physical sites and every enrolled positive and negative target. Both disjoint extraction stages are prerequisites to optimization. Stop on time, memory, parity, integrity or exposure failure.",
        }
        dest = EX / (name + ".json")
        persist(dest, feature)
        features.append(dest)
    repeats = {"operator": 2, "qualitative": 2, "full_rows": 2, "service_reply": 2}
    old_index = unpack(OLD / "feature-index.json.gz")
    epoch_labels = sum(
        v["response_positions"] * repeats.get(k, 1) for k, v in q["groups"].items()
    ) + sum(r["positions"] for r in old_index)
    epoch_pref = sum(
        v.get("preferences", 0) * repeats.get(k, 1) for k, v in q["groups"].items()
    ) + sum(bool(r["negative"]) for r in old_index)
    s = {
        **common,
        "id": FIT,
        "feature_stages": [OLD.name, *FEATURES],
        "epochs": 2,
        "micro_positions": 192,
        "micro_accumulation": 4,
        "lr": 0.00032,
        "margin": 1.0,
        "preference_weight": 12.0,
        "teacher_kl": 0.18,
        "repeat_by_group": repeats,
        "expected_positions": 2 * epoch_labels,
        "expected_preferences": 2 * epoch_pref,
        "free_sql_cap": 192,
        "free_answer_cap": 192,
        "question": "Can a readout driven exclusively by physical layer increments learn evidence-changing replies, keep broad language fit and freely execute source-dependent SQL?",
        "protocol": "Train one shared readout on every original turn and every new turn for two immutable seeded epochs. Learn from real conditional target distributions and supervised alternatives in both directions. Preserve the ordinary instruction distribution with full forward KL on all positive positions, coefficient 0.18. Native log likelihood, chosen-minus-rejected changed-token odds and KL have different units and are independently reported. Four new groups receive twice the exposure per epoch. The old cache must match the original physical backbone and its checked five-state representation. Every supervised optimizer update must be finite and nonzero. Require exact print-through of the ordinary upstream exit after replacing all five states by that exit on all 112 training probes at fitted weights, mean NLL gain >=0.08 across fourteen groups and no average regression larger than 0.02 nats in human instruction, arithmetic, explanation or earlier assistance. On 24 hash-selected enrolled SQL questions, generate from a real schema with no gold action available and execute on ALL three to nine human worlds, requiring at least eight complete questions and six different successful programs. On twelve hash-selected enrolled qualitative pairs and twelve enrolled grounded service pairs, freely generate on BOTH actual positive contexts with the whole vocabulary. Require at least three completely correct qualitative pairs and three completely correct service pairs. Score a qualitative reply by a clear choice and a service reply by at least two distinct actual field values from one valid result row. Any missed fit prerequisite closes the held experiment for these weights. On success run one direct comparison only.",
    }
    s["sources"] = {**common["sources"], **{str(p): digest(p) for p in features}}
    fit_path = EX / (FIT + ".json")
    persist(fit_path, s)
    direct = {
        **common,
        "id": PRIMARY,
        "material": str(PANEL),
        "max_tokens": 384,
        "max_turns": 3,
        "context_turns": 1,
        "prefill_cache_state": True,
        "feature_stage": FIT,
        "sources": {**s["sources"], str(fit_path): digest(fit_path)},
        "question": "Does changing an entirely unseen source change the model’s answer correctly, including after 131072 raw tokens, while retaining latency and grounded tool use?",
        "protocol": "Compare exactly two deployed native models, the newly fitted recurrent checkpoint and untouched BF16 Qwen Base. Freeze 148 shared original user openings, physical inputs, order, tokenizer, thinking-enabled greedy policy, callbacks and scorer. Require all 148 per arm even after a quality miss. Four complete CLDR documents from 152185 to 164271 raw tokens appear in this first comparison; native cached prefill actually consumes every prompt ID and the sought literal occurs beyond 131072. No truncation, retrieval proxy, externally inserted answer, evaluator-guided correction or reranking. One locale turn of 384. Elsewhere up to three turns of 384 each. 48 novel questions with genuine SQL on changed tables, 48 paired qualitative reasoning questions, and 48 held customer conversations with actual SQLite queries over their published search results. Original service logs are simulator-backed. Both arms choose and execute their own database calls. Review the 48 complete customer responses blind to arm and aggregate, accepting all source-supported options. Freeze individual reasons before unmasking. Report quality, completion, first-token wait, wall, opening and post-first throughput and actual peak together. Require each of the three 24-pair families to achieve at least six whole successes, net at least five versus Base and source-level exact paired two-sided p<=0.05, no individually correct Base request lost in each family. Require all four raw locale inputs consumed fully, both outputs correct on both languages, at least one additional exact response against Base, noninferior completion for each family, no increase in unsupported service replies, strictly shorter overall and first-token wall times, smaller peak and higher opening and post-first rates across 148. Stop if hardware budget or scientific prerequisite fails, without a release claim. This is a fresh transfer gate. A joint pass requires the unchanged 340-request retention comparison, independent seed, equally trained matched K1 on the same donor, and a cold actual installed long-context replay before overall superiority can be claimed.",
    }
    persist(EX / (PRIMARY + ".json"), direct)
    print(s["expected_positions"], s["expected_preferences"], flush=True)


if __name__ == "__main__":
    build()
