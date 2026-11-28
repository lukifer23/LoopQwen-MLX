"""Blind complete visible answers for source review and join locked grades to costs."""

import argparse
from collections import defaultdict

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
        item = {
            "source": source,
            "world": world,
            "reference": reference,
            "answer": tr["answer"],
            "calls": calls,
        }
        prompts[fingerprint(item)] = item
    save(out / "blinded.json", {k: prompts[k] for k in sorted(prompts)})
    return prompts


def pending(directory, stage, prior, review_conflicts=False):
    """Carry a locked human/source judgment only for identical review content."""
    target = directory / stage
    deck = read(target / "blinded.json")
    known, origins = {}, {}
    variants = defaultdict(dict)
    conflicts = set()
    for name in (*prior, stage):
        folder = directory / name
        grade_path = folder / "grades.json"
        if not grade_path.exists():
            continue
        graded = read(grade_path)
        originals = read(folder / "blinded.json")
        sha = digest(grade_path)
        for key in deck.keys() & graded.keys():
            if originals.get(key) != deck[key] or fingerprint(deck[key]) != key:
                raise ValueError("A prior grade belongs to different source material")
            if review_conflicts:
                variants[key][name] = {"grade": graded[key], "grade_file": sha}
            if key in known and known[key] != graded[key]:
                if not review_conflicts:
                    raise ValueError(
                        "Conflicting locked judgments require source review"
                    )
                conflicts.add(key)
            known[key] = graded[key]
            origins.setdefault(key, []).append({"stage": name, "grade_file": sha})
    if review_conflicts:
        for key in conflicts:
            if stage not in variants[key]:
                del known[key], origins[key]
        save(
            target / "conflicting-prior.json",
            {k: variants[k] for k in sorted(conflicts)},
        )
    save(target / "carried.json", {k: known[k] for k in sorted(known)})
    save(target / "carried-sources.json", {k: origins[k] for k in sorted(origins)})
    new = {k: v for k, v in deck.items() if k not in known}
    save(target / "pending.json", new)
    print(
        len(known),
        "identical locked judgments and",
        len(new),
        "responses requiring source review",
    )


def lock(directory, stage):
    folder = directory / stage
    deck = read(folder / "blinded.json")
    carried = read(folder / "carried.json")
    path = folder / "new-grades.json"
    new = read(path) if path.exists() else {}
    if set(carried) & set(new) or set(carried) | set(new) != set(deck):
        raise ValueError("Every new answer must have its own source judgment")
    grades = carried | new
    for key, grade in grades.items():
        if fingerprint(deck[key]) != key or type(grade["correct"]) is not bool:
            raise ValueError("Invalid judgment identity or correctness")
        if grade["correct"] and (grade["unsupported"] or grade["first_error"]):
            raise ValueError("An unsupported or incorrect answer cannot receive credit")
    dest = folder / "grades.json"
    if dest.exists() and any(grades.get(k) != v for k, v in read(dest).items()):
        raise ValueError("Cannot silently replace an earlier locked grade")
    save(dest, {k: grades[k] for k in sorted(grades)})


def consistency(stages, name):
    """Compare primary credit and ancillary flags only on byte-identical review content."""
    folder = ROOT / "review"
    seen = defaultdict(list)
    sources = {}
    for stage in stages:
        grades = folder / stage / "grades.json"
        deck = read(folder / stage / "blinded.json")
        source = read(grades)
        if set(source) != set(deck):
            raise ValueError("Only a fully locked review can enter the comparison")
        sources[stage] = digest(grades)
        for key, grade in source.items():
            if fingerprint(deck[key]) != key:
                raise ValueError("Changed blinded reply")
            seen[key].append({"stage": stage, **grade})
    shared = {k: v for k, v in seen.items() if len(v) > 1}
    conflicts = {
        field: {k: v for k, v in shared.items() if len({x[field] for x in v}) > 1}
        for field in ("correct", "unsupported")
    }
    save(
        ROOT / "direct" / (name + ".json"),
        {
            "grades": sources,
            "unique_responses": len(seen),
            "shared_across_stages": len(shared),
            "occurrences": sum(len(v) for v in seen.values()),
            "conflicts": conflicts,
            "scope": "Repeated identical source, world, reference, answer and tool history across locked development reviews. This checks label consistency, not independent inter-rater agreement.",
        },
    )


def first_visible(turns, timestamps, tok):
    """Time of the first visible prose token in the actual public conversation."""
    closing = tok.convert_tokens_to_ids("</think>")
    visible = None
    timing = iter(timestamps)
    for turn in turns:
        generated = turn["token_ids"]
        start = len(turn["prompt_ids"])
        stamps = [
            next(t for pos, t in timing if pos == start + i)
            for i in range(len(generated))
        ]
        if visible is not None or closing not in generated:
            continue
        after = generated.index(closing) + 1
        # Only natural-language text visible before an action can count.
        # Token-limit traces may still contain the beginning of a reply.
        for i in range(after, len(generated)):
            piece = tok.decode([generated[i]]).strip()
            if piece in ("<tool_call>", "<|im_end|>", "<think>"):
                break
            if piece and piece != "</think>":
                visible = stamps[i]
                break
    return visible


def aggregate(stage):
    from loopqwen.research.cross_view_study import tokenizer
    from scripts.middle_depth_budget import BASE

    tok = tokenizer({"checkpoint": str(BASE)})
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
            grade = {"correct": False, "first_error": tr["status"], "unsupported": None}
        else:
            k = fingerprint(
                {
                    "source": source,
                    "world": world,
                    "reference": reference,
                    "answer": tr["answer"],
                    "calls": calls,
                }
            )
            grade = rates[k]
        tps = sum(len(t["token_ids"]) - 1 for t in tr["turns"])
        decode = sum(t["seconds"] - t["ttft_s"] for t in tr["turns"])
        visible = first_visible(tr["turns"], row["timing"], tok)
        results.append(
            {
                "file": str(f.relative_to(ROOT)),
                "sha": digest(f),
                "arm": f.parent.name,
                "cap": int(f.parent.parent.name),
                "source": source,
                "world": world,
                "family": row["family"],
                "grade": grade,
                "status": tr["status"],
                "seconds": tr["seconds"],
                "time_to_correct": tr["seconds"] if grade["correct"] else None,
                "first_token": tr["turns"][0]["ttft_s"],
                "first_visible": visible,
                "token_count": sum(len(t["token_ids"]) for t in tr["turns"]),
                "input_tokens": len(tr["turns"][0]["prompt_ids"]),
                "decode_tps": tps / decode if decode else None,
                "opening_tps": len(tr["turns"][0]["prompt_ids"])
                / tr["turns"][0]["ttft_s"],
                "peak": max(t["peak_bytes"] for t in tr["turns"]),
            }
        )
    save(ROOT / "review" / stage / "scored.json", results)
    groups = defaultdict(list)
    for r in results:
        if r["world"] not in ("own", "shared"):
            groups[r["arm"], r["cap"], r["family"]].append(r)
    summary = [
        {
            "arm": a,
            "cap": c,
            "family": f,
            "episodes": len(z),
            "success": sum(r["grade"]["correct"] for r in z),
            "by_eight_seconds": sum(
                r["grade"]["correct"] and r["seconds"] <= 8 for r in z
            ),
            "seconds": sum(r["seconds"] for r in z),
        }
        for (a, c, f), z in groups.items()
    ]
    save(ROOT / "review" / stage / "summary.json", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "action", choices=("blind", "aggregate", "pending", "lock", "consistency")
    )
    p.add_argument("stage")
    p.add_argument("--prior", nargs="*", default=[])
    p.add_argument("--review-conflicts", action="store_true")
    args = p.parse_args()
    if args.action == "consistency":
        consistency(args.prior, args.stage)
    elif args.action == "pending":
        review(args.stage)
        pending(ROOT / "review", args.stage, args.prior, args.review_conflicts)
    elif args.action == "lock":
        lock(ROOT / "review", args.stage)
    elif args.action == "blind":
        review(args.stage)
    else:
        aggregate(args.stage)
