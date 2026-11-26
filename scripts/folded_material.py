"""Freeze the label-free layer targets, new human problems and historical execution checks."""

import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from contrast_material import REV, key
from huggingface_hub import hf_hub_download
from mlx_lm.utils import load_tokenizer
from operand_material import external
from program_reasoning_material import former, norm, sha

from loopqwen.evaluation import executable_tools
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import initial_messages, prompt

DATA = Path("research/data")
PRIOR = "0a9613a"
SEED = 847231


def main():
    tok = load_tokenizer(local_snapshot())
    rng = random.Random(SEED)
    raw = DATA / "selective-raw-training.json"
    tool = DATA / "tool-dialogue-typed-training.json"
    pools = {"raw": {}, "dialogue": {}}
    for r in read(raw)["rows"]:
        pools["raw"][r["id"]] = r["ids"]
    for r in sorted(read(tool)["examples"], key=lambda r: sha("fold-turn" + r["id"])):
        msgs = r["messages"]
        question = next(m["content"] for m in msgs if m["role"] == "user")
        gid = sha(msgs[0]["content"] + "\n" + question)
        if gid not in pools["dialogue"]:
            text = prompt(tok, msgs) + r["thinking"] + "\n</think>\n\n" + r["visible"]
            pools["dialogue"][gid] = tok.encode(text, add_special_tokens=False)
    rows = []
    for domain, g in pools.items():
        seen = set()
        candidates = []
        for gid, original in sorted(
            g.items(), key=lambda t: sha("fold-context" + t[0])
        ):
            ids = original[-256:]
            identity = fingerprint(ids)
            if len(ids) < 32 or identity in seen:
                continue
            seen.add(identity)
            candidates.append((gid, ids))
        assert len(candidates) >= 544, (domain, len(candidates))
        for i, (gid, ids) in enumerate(candidates[:544]):
            rows.append(
                {
                    "id": gid,
                    "domain": domain,
                    "split": "train" if i < 512 else "validation",
                    "ids": ids,
                    "positions": sorted(rng.sample(range(len(ids)), 24)),
                }
            )
    save(
        DATA / "folded-layer-training.json",
        {"sources": {str(p): digest(p) for p in (raw, tool)}, "rows": rows},
    )
    panel = []
    ext = external()
    filename = "data/validation-00000-of-00001.parquet"
    validation = Path(
        hf_hub_download("google/boolq", filename, repo_type="dataset", revision=REV)
    )
    assert (
        digest(validation)
        == "52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8"
    )
    bools = pq.read_table(validation).to_pylist()
    excluded = former(
        {q for _, q, _ in ext} | {r["passage"] for r in bools}, prior=PRIOR
    )
    blocked = {tuple(norm(q)[:8]) for q in excluded}
    counts = Counter()
    for src, q, gold in sorted(ext, key=lambda r: sha("fold-new-math" + r[1])):
        opening = tuple(norm(q)[:8])
        if gold is None or opening in blocked or counts[src] >= 10:
            continue
        question = q + "\nGive the final numeric answer."
        panel.append(
            {
                "id": sha(q),
                "family": "math",
                "source": src,
                "question": question,
                "gold": gold,
            }
        )
        counts[src] += 1
        blocked.add(opening)
    assert sorted(counts.values()) == [10, 10]
    truths = Counter()
    for r in sorted(
        bools, key=lambda r: sha("fold-reading" + r["passage"] + r["question"])
    ):
        passage = r["passage"]
        opening = tuple(norm(passage)[:8])
        truth = r["answer"]
        if opening in blocked or truths[truth] >= 10:
            continue
        user = f"Passage\n{passage}\n\nQuestion\n{r['question']}\nAnswer yes or no based on the passage."
        panel.append(
            {
                "id": key(passage),
                "family": "judgment",
                "source": "BoolQ validation",
                "question": user,
                "gold": "yes" if truth else "no",
            }
        )
        truths[truth] += 1
        blocked.add(opening)
        if len(panel) == 40:
            break
    assert sorted(truths.values()) == [10, 10]
    from loopqwen.evaluation.mixed_field import catalog

    field = read(DATA / "release-mixed-field-01.json")
    for w in field["rows"]:
        if w["family"] in ("package", "database", "file"):
            panel.append({**w, "source": "field retention"})
    for w in read(DATA / "tool-exec-bfcl-01.json")["rows"]:
        user = w["question"][0][0]["content"] + executable_tools.INSTRUCTION
        panel.append(
            {**w, "family": "public", "source": "BFCL retention", "user_request": user}
        )
    rng.shuffle(panel)
    for w in panel:
        group = w["family"]
        tools = (
            []
            if group in ("math", "judgment")
            else (
                executable_tools.tools_for(w)
                if group == "public"
                else catalog(w, field, replay=True)
            )
        )
        user = w.get("user_request", w["question"])
        msgs = (
            initial_messages(user, tools)
            if tools
            else [{"role": "user", "content": user}]
        )
        ids = tok.encode(prompt(tok, msgs), add_special_tokens=False)
        if "initial_ids" in w:
            assert ids == w["initial_ids"]
        w["initial_ids"] = ids
    assert len(panel) == 92 and len({w["id"] for w in panel}) == 92
    save(
        DATA / "folded-layer-panel.json",
        {
            "rows": panel,
            "field_sha256": digest(DATA / "release-mixed-field-01.json"),
            "excluded_through": PRIOR,
            "boolq_revision": REV,
            "boolq_file": filename,
            "boolq_sha256": digest(validation),
        },
    )
    print(Counter(w["domain"] for w in rows), Counter(w["family"] for w in panel))


if __name__ == "__main__":
    main()
