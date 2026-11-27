"""Blind complete visible answers for source review and join locked grades to costs."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.middle_depth_budget import ROOT
from scripts.middle_depth_development import SOURCE


def records(stage):
    tasks = {r["id"]: r for r in read(SOURCE)}
    for file in sorted((ROOT / "short" / stage).glob("*/*/*.json.gz")):
        row = unpack(file)
        source = row["source"]
        world = row["world"]
        r = tasks[source]
        key = (
            "changed_reference"
            if world == "changed"
            else "followup_reference"
            if world in ("own", "shared")
            else "reference"
        )
        yield file, row, source, world, r[key]


def review(stage):
    out = ROOT / "review" / stage
    prompts = {}
    for f, row, source, world, reference in records(stage):
        tr = row["trace"]
        if tr["status"] != "answered":
            continue
        # Identical answers to the same question with identical real callbacks
        # are graded once. Identity, timing and hidden reasoning are withheld.
        calls = [
            {
                k: v
                for k, v in t.items()
                if k in ("call", "calls", "observation", "observations")
            }
            for t in tr["turns"]
            if "call" in t or "calls" in t
        ]
        item = dict(
            source=source,
            world=world,
            reference=reference,
            answer=tr["answer"],
            calls=calls,
        )
        prompts[fingerprint(item)] = item
    save(out / "blinded.json", {k: prompts[k] for k in sorted(prompts)})
    return prompts


def aggregate(stage):
    prompts = review(stage)
    rates = read(ROOT / "review" / stage / "grades.json")
    if set(prompts) != set(rates):
        raise ValueError("Missing or extra source-review grades")
    results = []
    for f, row, source, world, reference in records(stage):
        tr = row["trace"]
        calls = [
            {
                k: v
                for k, v in t.items()
                if k in ("call", "calls", "observation", "observations")
            }
            for t in tr["turns"]
            if "call" in t or "calls" in t
        ]
        if tr["status"] != "answered":
            grade = dict(correct=False, first_error=tr["status"], unsupported=None)
        else:
            k = fingerprint(
                dict(
                    source=source,
                    world=world,
                    reference=reference,
                    answer=tr["answer"],
                    calls=calls,
                )
            )
            grade = rates[k]
        tps = sum(len(t["token_ids"]) - 1 for t in tr["turns"])
        decode = sum(t["seconds"] - t["ttft_s"] for t in tr["turns"])
        # Trace progress reports the consumed physical position after every
        # greedy token. Locate the first actual visible non-whitespace token.
        visible = None
        from loopqwen.research.cross_view_study import tokenizer
        from scripts.middle_depth_budget import BASE

        if not hasattr(aggregate, "tokenizer"):
            aggregate.tokenizer = tokenizer({"checkpoint": str(BASE)})
        tok = aggregate.tokenizer
        timing = iter(row["timing"])
        for turn in tr["turns"]:
            current_start = len(turn["prompt_ids"])
            generated = turn["token_ids"]
            for i, token in enumerate(generated):
                when = next(v for pos, v in timing if pos == current_start + i)
                if token == 248069:
                    after = i + 1
                    break
            else:
                continue
            for j in range(after, len(generated)):
                when = next(v for pos, v in timing if pos == current_start + j)
                if tok.decode([generated[j]]).strip() not in ("", "<|im_end|>"):
                    visible = when
                    break
            if visible is not None:
                break
        results.append(
            dict(
                file=str(f.relative_to(ROOT)),
                sha=digest(f),
                arm=f.parent.name,
                cap=int(f.parent.parent.name),
                source=source,
                world=world,
                family=row["family"],
                grade=grade,
                status=tr["status"],
                seconds=tr["seconds"],
                time_to_correct=tr["seconds"] if grade["correct"] else None,
                first_token=tr["turns"][0]["ttft_s"],
                first_visible=visible,
                token_count=sum(len(t["token_ids"]) for t in tr["turns"]),
                input_tokens=len(tr["turns"][0]["prompt_ids"]),
                decode_tps=tps / decode if decode else None,
                opening_tps=len(tr["turns"][0]["prompt_ids"])
                / tr["turns"][0]["ttft_s"],
                peak=max(t["peak_bytes"] for t in tr["turns"]),
            )
        )
    save(ROOT / "review" / stage / "scored.json", results)
    groups = defaultdict(list)
    for r in results:
        if r["world"] not in ("own", "shared"):
            groups[r["arm"], r["cap"], r["family"]].append(r)
    summary = [
        dict(
            arm=a,
            cap=c,
            family=f,
            episodes=len(z),
            success=sum(r["grade"]["correct"] for r in z),
            by_eight_seconds=sum(
                r["grade"]["correct"] and r["seconds"] <= 8 for r in z
            ),
            seconds=sum(r["seconds"] for r in z),
        )
        for (a, c, f), z in groups.items()
    ]
    save(ROOT / "review" / stage / "summary.json", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("blind", "aggregate"))
    p.add_argument("stage")
    args = p.parse_args()
    review(args.stage) if args.action == "blind" else aggregate(args.stage)
