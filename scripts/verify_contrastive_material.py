"""Rebuild a source-pair ledger independently and check held separation, labels and interventions."""

from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from scripts.contrastive_world_material import OLD, OUT, REASON, SERVICES, TABLES
from scripts.program_language_material import norm
from scripts.table_policy_material import persist


def run():
    import json

    spec = read(OUT / "qualification.json")
    for name, sha in spec["sources"].items():
        assert digest(Path(name)) == sha
    rows = unpack(OUT / "training.json.gz")
    withheld = unpack(OUT / "withheld.json.gz")
    assert (
        digest(OUT / "training.json.gz") == spec["training"]
        and digest(OUT / "withheld.json.gz") == spec["withheld"]
    )
    table = {w["id"]: w for w in unpack(TABLES / "training.json.gz")}
    reason = {w["id"]: w for w in unpack(REASON / "training.json.gz")}
    service = {w["id"]: w for w in unpack(SERVICES / "training.json.gz")}
    service_pairs = {
        fingerprint(p["ids"]): p
        for p in read(SERVICES / "paired-replies.json")["pairs"]
    }
    old = unpack(OLD / "training.json.gz")
    held_questions = {norm(w["question"]) for w in withheld["tables"]} | {
        norm(m["question"]) for w in withheld["qualitative"] for m in w["members"]
    }
    # Match full source questions inside user prompts, and long original human answers inside full targets.
    held_strings = {s for s in held_questions if len(s.split()) >= 5}
    held_strings.update(
        norm(w["reply"])
        for p in withheld["services"]
        for w in p
        if len(norm(w["reply"]).split()) >= 8
    )
    collisions = defaultdict(list)
    for row in [*old, *rows]:
        fields = [m.get("content", "") for m in [*row["messages"], row["target"]]]
        for content in fields:
            words = " " + norm(content) + " "
            for s in held_strings:
                if " " + s + " " in words:
                    collisions[s].append(row["id"])
    held_contexts = {w["context"] for w in withheld["tables"]}
    held_bgs = {w["source_background"] for w in withheld["qualitative"]}
    held_dialogues = {w["dialogue"] for pair in withheld["services"] for w in pair}
    counts = Counter()
    paired = defaultdict(list)
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    for row in rows:
        group = row["group"]
        key = row["pair"]
        if group in ("operator", "inspect", "resolved", "full_rows"):
            w = table[key]
            assert w["context"] not in held_contexts
            if group == "operator":
                assert (
                    row["target"]["tool_calls"][0]["function"]["arguments"]["statement"]
                    == w["statement"]
                )
            if group in ("resolved", "full_rows"):
                obs = json.loads(row["messages"][-1]["content"])
                assert not obs["truncated"] and obs["referenced_tables"] == ["records"]
                if group == "resolved":
                    assert (
                        str(obs["rows"][0][0]).casefold()
                        == str(row["target"]["content"]).casefold()
                    )
                else:
                    matches = [
                        v
                        for v in w["versions"]
                        if [obs["columns"], *obs["rows"]] == v["rows"]
                    ]
                    assert any(v["answer"] == row["target"]["content"] for v in matches)
        if group == "qualitative":
            w = reason[key]
            m = w["members"][row["member"]]
            assert (
                w["source_background"] not in held_bgs
                and row["target"]["content"] == m["correct"]
                and m["question"] in row["messages"][0]["content"]
            )
        if group in ("service_reply", "service_action"):
            name = (
                key
                if group == "service_action"
                else service_pairs[key]["ids"][row["member"]]
            )
            w = service[name]
            assert (
                w["dialogue"] not in held_dialogues
                and row["messages"][: len(w["messages"])] == w["messages"]
            )
            if group == "service_action":
                assert row["target"] == w["target"]
            else:
                assert (
                    row["target"]["content"] == w["reply"]
                    and json.loads(row["messages"][-1]["content"]) == w["observation"]
                )
        if "rejected" in row:
            paired[group, key].append(row)
        _p, y = token_pair(tok, row)
        counts[group] += len(y)
    for members in paired.values():
        a, b = sorted(members, key=lambda r: r["member"])
        assert a["member"] == 0 and b["member"] == 1 and a["messages"] != b["messages"]
        assert (
            a["target"] == b["rejected"][0]["target"]
            and b["target"] == a["rejected"][0]["target"]
        )
    assert all(spec["groups"][g]["response_positions"] == n for g, n in counts.items())
    result = {
        "training": spec["training"],
        "withheld": spec["withheld"],
        "verified_positive_turns": len(rows),
        "complete_incompatible_pairs": len(paired),
        "positive_target_positions": sum(counts.values()),
        "by_group": dict(Counter(k[0] for k in paired)),
        "excluded_held_groups": {
            "table_files": len(held_contexts),
            "qualitative_backgrounds": len(held_bgs),
            "source_service_dialogues": len(held_dialogues),
        },
        "held_complete_phrases_tested": len(held_strings),
        "held_phrases_in_any_old_or_new_training_field": dict(collisions),
    }
    assert not collisions, dict(collisions)
    persist(OUT / "independent-replay.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    run()
