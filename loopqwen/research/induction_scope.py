"""Exercise the deployed reader on the same narratives with different requested answers."""

import sys
from pathlib import Path

from ..deployment import load_trained
from ..evaluation.generation_eval import generate
from ..evaluation.mechanism import packed
from ..identity import fingerprint
from ..io import digest, read, save
from . import core
from .deduction_data import prompt


def questions(story, extended=False):
    subject = story["question"].split()[-1][:-1]
    probes = [
        f"What kind of animal is {subject}?",
        f"Does the story mention {subject}?",
        "How many different individuals are mentioned?",
    ]
    if extended:
        probes += [
            f"Which animal kind does {subject} belong to?",
            f"Can you find {subject} in these statements?",
            "How many creatures are named in this story?",
        ]
    return probes


def measure(s, out, b, new, prior, t):
    records = []
    data = read(s["material"])
    for stage in ("primary", "replication"):
        for row in data[stage][: s["stories_per_panel"]]:
            for query in questions(row, s.get("extended", False)):
                b.check()
                text = prompt(t, row["facts"], query)
                ids = t.encode(text, add_special_tokens=False)
                _, old_route = prior.context_policy.choose(
                    prior, ids, s["generation_cap"]
                )
                _, new_route = new.context_policy.choose(new, ids, s["generation_cap"])
                item = {
                    "story": row["id"],
                    "question": query,
                    "input_sha256": fingerprint(ids),
                    "prior": old_route,
                    "new": new_route,
                }
                if old_route != new_route or len(records) < s["generation_replays"]:
                    for arm, model in (("prior", prior), ("new", new)):
                        val = generate(
                            model,
                            t,
                            text,
                            s["generation_cap"],
                            progress=lambda _: b.check(),
                            stop_ids=(248046,),
                        )
                        item[arm + "_tokens"] = val["token_ids"]
                        item[arm + "_seconds"] = val["elapsed_s"]
                        item[arm + "_text"] = t.decode(val["token_ids"])
                records.append(item)
                packed(out / "scope.json.gz", {"records": records})
                if old_route != new_route or item.get("new_tokens") != item.get(
                    "prior_tokens"
                ):
                    raise ValueError(
                        "Production route or emitted tokens changed for a different user intent"
                    )
    return {
        "requests": len(records),
        "replays": sum("new_tokens" in r for r in records),
        "passed": True,
        "status": "completed",
    }


def run(name):
    reg = Path("research/experiments") / (name + ".json")
    s = read(reg)
    if (
        digest(reg)
        != read("research/registry.json")["experiments"][name]["registration"]
    ):
        raise ValueError("Scope study registration differs")
    for parent, expected in s["prerequisites"].items():
        path = Path("research/results") / parent
        if (
            digest(path / "checksums.json") != expected
            or not read(path / "decision.json")["passed"]
        ):
            raise ValueError("Scope study prerequisite differs")
    out = core.OUT / name
    out.mkdir(parents=True, exist_ok=False)
    b = core.Budget(name, s["limits"])
    decision = {"registration": digest(reg), "passed": False}
    try:
        new, t, _ = load_trained(Path(s["new"]) / "model")
        prior, _, _ = load_trained(Path(s["prior"]) / "model")
        decision.update(measure(s, out, b, new, prior, t))
        decision.update(passed=True, status="completed")
    except (ValueError, InterruptedError, TimeoutError, MemoryError) as e:
        decision.update(
            status="failed" if isinstance(e, ValueError) else "interrupted",
            reason=str(e),
        )
    finally:
        b.close()
    from ..evaluation.mechanism import unpack

    records = (
        unpack(out / "scope.json.gz")["records"]
        if (out / "scope.json.gz").exists()
        else []
    )
    decision.update(
        requests=len(records), replays=sum("new_tokens" in r for r in records)
    )
    save(out / "decision.json", decision)
    print(decision, flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
