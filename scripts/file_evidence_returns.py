"""Locate the changed evidence relative to the reused prefix and replay the first file pair from scratch."""

import signal
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import unpack
from loopqwen.file_evidence import FileEvidenceAssistant
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from loopqwen.research.file_evidence import catalog, write_files
from loopqwen.tool_session import compact
from scripts.file_evidence_audit import NAME, PANEL
from scripts.verify_runtime_wheel import trace


def verify():
    root = Path("research/results") / NAME
    spec = read(root / "registration.json")
    actor = FileEvidenceAssistant(spec["checkpoint"], root)
    groups = {}
    for row, record in zip(
        read(PANEL)["rows"], unpack(root / "candidate.json.gz"), strict=True
    ):
        if row["family"] == "file":
            groups.setdefault(row["cluster"], []).append(
                (row, record["requests"][0]["trajectory"])
            )
    exposed = []
    for pair in groups.values():
        a, b = pair
        assert a[1]["turns"][0]["prompt_ids"] == b[1]["turns"][0]["prompt_ids"]
        assert a[1]["turns"][0]["token_ids"] == b[1]["turns"][0]["token_ids"]
        turns = []
        for row, tr in pair:
            for before, after in zip(tr["turns"], tr["turns"][1:]):
                obs = {"value": row["evidence"]}
                if (
                    before.get("observation") == obs
                    and before.get("call") in row["calls"]
                ):
                    ids = after["prompt_ids"]
                    reused = after["reused_prompt_tokens"]
                    wire = compact(obs)
                    assert wire in actor.tokenizer.decode(ids[reused:])
                    turns.append((ids, reused))
                    break
        if len(turns) == 2:
            (left, x), (right, y) = turns
            assert x == y and left[:x] == right[:y]
            diff = next(i for i, (v, w) in enumerate(zip(left, right)) if v != w)
            assert diff >= x
            exposed.append(
                {
                    "article": a[0]["page"],
                    "reused_tokens": x,
                    "first_changed_token": diff,
                    "full_input_lengths": [len(left), len(right)],
                }
            )
    pair = next(iter(groups.values()))
    replayed = []
    with TemporaryDirectory() as folder:
        write_files([w for w, _ in pair], folder)
        for w, archived in pair:
            tr = actor.conversation(catalog(w, folder)).respond(
                w["questions"][0],
                max_tokens=spec["max_tokens"],
                max_turns=spec["max_turns"],
                reuse_prefix=False,
                record_kv_history=True,
            )
            equal = trace(tr) == trace(archived)
            replayed.append(
                {
                    "id": w["id"],
                    "identical": equal,
                    "actual_trace": fingerprint(trace(tr)),
                    "primary_trace": fingerprint(trace(archived)),
                    "fresh_answer": tr["answer"],
                    "primary_answer": archived["answer"],
                    "maximum_reuse": max(
                        t.get("reused_prompt_tokens", 0) for t in tr["turns"]
                    ),
                }
            )
    dest = Path("docs/measurements/file-evidence-return-path.json")
    result = {
        "model": NAME,
        "same_initial_call_on_file_pairs": len(groups),
        "both_real_returns_in_uncached_suffix": exposed,
        "first_pair_full_replay": replayed,
    }
    if dest.exists():
        assert read(dest) == result
    else:
        save(dest, result)
    print(len(exposed), "paired returns in fresh suffix;", replayed)


if __name__ == "__main__":
    signal.alarm(90)
    verify()
