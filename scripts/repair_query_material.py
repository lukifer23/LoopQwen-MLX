"""Assemble the full supervised repair and reasoning corpus with original per-turn identities."""

from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from scripts.program_language_material import norm
from scripts.provider_field_mutations import LOCAL
from scripts.provider_recovery_field import PANEL
from scripts.shared_query_material import ROOT as SHARED
from scripts.table_policy_material import persist

REPAIR = PANEL.parent
ROOT = Path("research/sources/repair-query")
PARTS = (
    ("", SHARED / "training.json.gz"),
    ("sql-error", REPAIR / "training.json.gz"),
    ("csv-key", REPAIR / "csv-training.json.gz"),
    ("csv-role", REPAIR / "csv-role-training.json.gz"),
    ("wrong-scalar", REPAIR / "semantic-training.json.gz"),
)
TRAINING = ROOT / "training.json.gz"


def build():
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    rows = []
    keys = set()
    groups = defaultdict(Counter)
    sources = {}
    reserved = [
        r
        for panel in (PANEL, LOCAL)
        for r in unpack(panel)["rows"]
        if r["condition"] == "unassisted"
    ]
    original_questions = {norm(r["question"]): r["id"] for r in reserved}
    overlaps = defaultdict(list)
    sequences = defaultdict(list)
    prompts = defaultdict(set)
    token_lengths = {}
    for namespace, path in PARTS:
        sources[str(path)] = digest(path)
        for w in unpack(path):
            row = {**w, "id": namespace + "/" + w["id"] if namespace else w["id"]}
            assert row["id"] not in keys
            keys.add(row["id"])
            text = {norm(m["content"]) for m in row["messages"] if m["role"] == "user"}
            for word in text & original_questions.keys():
                overlaps[original_questions[word]].append(row["id"])
            p, y = token_pair(tok, row)
            sequences[fingerprint([p, y])].append(row["id"])
            prompts[fingerprint(p)].add(fingerprint(y))
            token_lengths[row["id"]] = (len(p), len(y))
            g = groups[row["group"]]
            g["turns"] += 1
            g["target_positions"] += len(y)
            g["opening_positions"] += len(p)
            rows.append(row)
    assert len(rows) == 6872 and len(keys) == len(rows)
    # Preserve the original selections and make any shared training examples visible.
    duplicates = [sorted(ids) for ids in sequences.values() if len(ids) > 1]
    if TRAINING.exists():
        assert unpack(TRAINING) == rows
    else:
        packed(TRAINING, rows)
    out = {
        "inputs": sources,
        "training": digest(TRAINING),
        "examples": len(rows),
        "groups": groups,
        "target_positions": sum(g["target_positions"] for g in groups.values()),
        "opening_positions": sum(g["opening_positions"] for g in groups.values()),
        "duplicate_training_prompt_target_sequences": duplicates,
        "exact_reserved_user_message_matches": dict(overlaps),
        "evidence": digest(SHARED / "evidence.json.gz"),
        "scope": "Complete targets from all five original corpora, with stable distinct IDs. Any identical prompt and target used in two selected curricula remains visible in the duplicate ledger. No held request, output or scorer-generated correction was used to select training examples. The overlap count tests exact normalized user-message equality; it cannot determine donor pretraining exposure.",
    }
    persist(ROOT / "inventory.json", out)
    # The two expert action records share every visible source token and demand
    # different numeric conclusions. Exclude their entire action/reply trajectories.
    incompatible = [k for k, values in prompts.items() if len(values) > 1]
    assert len(incompatible) == 1
    contradictory = {"finqa/EW/2016/page_92.pdf-3", "finqa/EW/2016/page_92.pdf-4"}
    removed = {r["id"] for r in rows if r["id"].rsplit("/", 1)[0] in contradictory}
    assert len(removed) == 4
    actions = [r for r in rows if r["id"] in removed and r["group"] == "report_action"]
    assert all(fingerprint(token_pair(tok, r)[0]) == incompatible[0] for r in actions)
    assert {
        r["target"]["content"]
        for r in rows
        if r["id"] in removed and r["group"] == "report_reply"
    } == {"90.73", "107.73"}
    evidence = unpack(SHARED / "evidence.json.gz")
    keep = set()
    for ids in sequences.values():
        choices = [k for k in ids if k not in removed]
        if choices:
            keep.add(min(choices, key=lambda k: (k not in evidence, k)))
    unique = [r for r in rows if r["id"] in keep]
    annotations = {k: v for k, v in evidence.items() if k in keep}
    qualified = ROOT / "verified-training.json.gz"
    evidence_path = ROOT / "verified-evidence.json.gz"
    for dest, value in ((qualified, unique), (evidence_path, annotations)):
        if dest.exists():
            assert unpack(dest) == value
        else:
            packed(dest, value)
    totals = defaultdict(Counter)
    for r in unique:
        g = totals[r["group"]]
        p, y = token_lengths[r["id"]]
        g["opening_positions"] += p
        g["target_positions"] += y
        g["turns"] += 1
    qualified_stats = {
        "raw": digest(TRAINING),
        "training": digest(qualified),
        "evidence": digest(evidence_path),
        "examples": len(unique),
        "groups": totals,
        "opening_positions": sum(g["opening_positions"] for g in totals.values()),
        "target_positions": sum(g["target_positions"] for g in totals.values()),
        "human_cited_queries": sum(len(a["evidence"]) for a in annotations.values()),
        "identical_turns_removed": len(rows) - len(unique) - len(removed),
        "conflicting_complete_sources_removed": sorted(removed),
        "reason": "One identical full financial report and user question about 2016 has expert answers of both 107.73 and 90.73. The former program uses 2015 repurchase numbers. Exclude both complete action/reply trajectories. Deduplicate identical prompt AND target IDs across the independently prepared datasets and retain original human evidence for enrolled examples.",
    }
    persist(ROOT / "verified-inventory.json", qualified_stats)
    print(
        "Qualified unique turns",
        len(unique),
        qualified_stats["target_positions"],
        qualified_stats["human_cited_queries"],
    )
    print(
        out["examples"],
        out["target_positions"],
        out["opening_positions"],
        "duplicate token sequences",
        len(duplicates),
        "exact reserved questions",
        len(overlaps),
    )


if __name__ == "__main__":
    build()
