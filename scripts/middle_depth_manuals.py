"""Freeze and replay a two-manual transfer test for changed-source recall."""

import argparse
import hashlib
import re
import tarfile
from pathlib import Path
from urllib.request import urlopen

from loopqwen.calculator_tool import tool
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.qwen_tools import declarations, render
from loopqwen.research.cross_view_study import tokenizer
from scripts.middle_depth_budget import BASE, BUDGET, ROOT
from scripts.middle_depth_documents import SPECS

PLAN = SPECS / "manual-recall.md"
FROZEN = SPECS / "manual-openings.json"
LOCAL = Path("data/research_external/gnu-manuals")
RAW = Path("outputs/middle-depth/manual-trajectories")
STAGE = "manual-recall"
LIMIT = 2100
WORLD_ORDER = ("original", "edited", "irrelevant")


def material(name):
    spec = read(SPECS / "manual-sources.json")[name]
    path = LOCAL / spec["url"].rsplit("/", 1)[1]
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(spec["url"], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != spec["archive_sha"]:
            raise ValueError(
                "Downloaded distribution differs from the registered source"
            )
        path.write_bytes(data)
    if digest(path) != spec["archive_sha"]:
        raise ValueError(f"Source archive changed at {path}")
    with tarfile.open(path) as archive:
        info = spec["member"]
        master = archive.extractfile(info).read().decode()
        nodes = re.findall(r"^Node: (.*?)\x7f", master, re.MULTILINE)
        parts = re.findall(rf"^({name}\.info-\d+): \d+$", master, re.MULTILINE)
        if not parts:
            raise ValueError("Expected a split GNU Info manual")
        seen, pieces = [], []
        for i, part in enumerate(parts):
            member = str(Path(info).with_name(part))
            intro, *records = (
                archive.extractfile(member).read().decode().split("\x1f\n")
            )
            if i == 0:
                pieces.append(intro)
            for record in records:
                header = re.match(r"File: [^\n]+?  Node: (.*?)(?:,  |\n)", record)
                if not header:
                    raise ValueError(f"Unexpected non-node content in {member}")
                seen.append(header[1])
            pieces.append("\n".join(records))
    if seen != nodes or len(nodes) != len(set(nodes)):
        raise ValueError("The plain text omits, reorders or duplicates a manual node")
    return spec, "".join(pieces), nodes


def evidence(name, text):
    if name == "make":
        rules = re.findall(
            r"^     objects = (.+?\\\n +[^\n]+)\n\n     edit : \$\(objects\)",
            text,
            re.MULTILINE,
        )
        objects = [re.findall(r"[\w.-]+\.o\b", line) for line in rules]
        if not objects or any(row != objects[0] for row in objects):
            raise ValueError("Ambiguous tutorial object list")
        count = len(set(objects[0]))
        (jobs,) = re.findall(r"jobserver unavailable: using -j(\d+)\.", text)
        jobs = int(jobs)
        return {
            "objects": count,
            "jobs": jobs,
            "compile_rounds": (count + jobs - 1) // jobs,
            "total_slots": (count + jobs - 1) // jobs + 1,
        }
    (blues,) = re.findall(
        r"^ +-rw-r--r-- gray/staff +(\d+) [^\n]* blues$", text, re.MULTILINE
    )
    out, same = re.findall(
        r"^ +genfile --length (\d+) (?:>|--file) outfile$", text, re.MULTILINE
    )
    if out != same:
        raise ValueError("The equivalent outfile commands disagree")
    diff = int(blues) - int(out)
    return {
        "blues_bytes": int(blues),
        "outfile_bytes": int(out),
        "larger": "blues" if diff > 0 else "outfile",
        "difference_bytes": abs(diff),
    }


def openings(name, tok):
    spec, source, nodes = material(name)
    built, references = {}, {}
    tools = [tool()]
    for world in WORLD_ORDER:
        body = source
        for before, after in spec.get(world, []):
            if body.count(before) != 1:
                raise ValueError(f"Ambiguous {name}/{world} edit")
            body = body.replace(before, after)
        refs = evidence(name, body)
        if refs != spec["answers"]["edited" if world == "edited" else "original"]:
            raise ValueError("Hand and executable references disagree")
        references[world] = refs
        question = body + "\n\n" + spec["question"]
        ids = tok.encode(
            render(tok, [{"role": "user", "content": question}], declarations(tools)),
            add_special_tokens=False,
        )
        if not spec["band_min"] <= len(ids) <= spec["band_max"]:
            raise ValueError("Physical opening falls outside the registered band")
        built[world] = (question, ids)
    common = next(
        i
        for i, triple in enumerate(zip(*(p for q, p in built.values())))
        if len(set(triple)) > 1
    )
    ids = built["original"][1][: common // 128 * 128]
    row = {
        "source": hashlib.sha256(source.encode()).hexdigest(),
        "node_order": fingerprint(nodes),
        "nodes": len(nodes),
        "common": len(ids),
        "common_sha": fingerprint(ids),
        "worlds": {
            w: {"input": fingerprint(p), "length": len(p), "reference": references[w]}
            for w, (q, p) in built.items()
        },
    }
    return row, (spec, tools, built, ids)


def freeze(check=False):
    tok = tokenizer({"checkpoint": str(BASE)})
    record = {
        "registration": digest(PLAN),
        "sources": digest(SPECS / "manual-sources.json"),
        "manuals": {name: openings(name, tok)[0] for name in ("make", "tar")},
    }
    if check or FROZEN.exists():
        if read(FROZEN) != record:
            raise ValueError("Registered manual openings changed")
    else:
        save(FROZEN, record)


def run_pair():
    from loopqwen.evaluation.mechanism import unpack
    from scripts.middle_depth_standalone import MODELS, check
    from scripts.middle_depth_wave import dispatch

    frozen = read(FROZEN)
    if frozen["registration"] != digest(PLAN) or frozen["sources"] != digest(
        SPECS / "manual-sources.json"
    ):
        raise ValueError("Registration changed")
    charges = [read(p) for p in (ROOT / "charges").glob("*.json")]
    used = sum(r["seconds"] for r in charges if r["stage"] == "long")
    manual_used = sum(r["seconds"] for r in charges if r["job"].endswith("/" + STAGE))
    if BUDGET["long"] - used < LIMIT - manual_used:
        save(
            ROOT / STAGE / "dispatch.json",
            {
                "status": "blocked-budget",
                "allocation": "long",
                "spent": used,
                "ceiling": BUDGET["long"],
                "remaining_manual_envelope": LIMIT - manual_used,
                "registration": digest(PLAN),
            },
        )
        raise RuntimeError(
            "Insufficient remaining long-context allowance for both manuals"
        )
    for arm in ("base", "loop"):
        check(MODELS / arm, arm)
    for i, name in enumerate(("make", "tar")):
        for j, world in enumerate(WORLD_ORDER):
            for arm in ("loop", "base") if (i + j) % 2 == 0 else ("base", "loop"):
                path = RAW / STAGE / name / arm / "512" / (world + "-first.json.gz")
                receipt = ROOT / STAGE / name / arm / (world + ".json")
                if not path.exists():
                    dispatch(
                        "manuals",
                        "request",
                        "--name",
                        name,
                        "--world",
                        world,
                        "--arm",
                        arm,
                    )
                raw = unpack(path)
                ids = raw["trace"]["turns"][0]["prompt_ids"]
                expected = frozen["manuals"][name]
                weights = (
                    "base-dc7cdfe"
                    if arm == "base"
                    else read(MODELS / arm / "registration.json")["weights"]
                )
                if (
                    raw["frozen"] != digest(FROZEN)
                    or raw["weights"] != weights
                    or raw["source"] != expected["source"]
                    or fingerprint(ids) != expected["worlds"][world]["input"]
                ):
                    raise ValueError("Archived trajectory identity mismatch")
                save(
                    receipt,
                    {
                        "raw": str(path),
                        "sha256": digest(path),
                        "source": raw["source"],
                        "input": fingerprint(ids),
                        "input_tokens": len(ids),
                        "weights": weights,
                        "status": raw["trace"]["status"],
                        "cold_prefill_seconds": raw["cold_build_seconds"],
                        "serving_seconds": raw["trace"]["seconds"],
                        "scratch_bytes": raw["scratch_bytes"],
                    },
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "run", "request"))
    parser.add_argument("--name", choices=("make", "tar"))
    parser.add_argument("--world", choices=WORLD_ORDER)
    parser.add_argument("--arm", choices=("base", "loop"))
    a = parser.parse_args()
    if a.action in ("freeze", "verify"):
        freeze(check=a.action == "verify")
    elif a.action == "run":
        run_pair()
    else:
        from scripts.middle_depth_documents import measure

        if not all((a.name, a.arm, a.world)):
            parser.error("request requires --name, --world and --arm")
        measure(
            argparse.Namespace(
                book=a.name,
                arm=a.arm,
                world=a.world,
                policy=Path("exports/standalone-middle-576") / a.arm,
                stage=STAGE,
                tokens=512,
                cache_mode="resident",
                followups=False,
            ),
            manual=True,
        )
