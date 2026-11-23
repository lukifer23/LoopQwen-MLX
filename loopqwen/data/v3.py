"""Fresh evaluation registry. V2 remains the source of Phase 3 training examples."""

import json
import random
import tempfile
from pathlib import Path
from shutil import rmtree

from transformers import AutoTokenizer

from . import v2
from .graphs import graph
from .reasoning import FAMILIES, problem
from ..identity import fingerprint
from ..io import read, save, digest
from ..model_loader import local_snapshot

ROOT = Path("data/v3")


def build():
    if ROOT.exists():
        m = read(ROOT / "manifest.json")
        for name, meta in m["splits"].items():
            if digest(ROOT / (name + ".jsonl")) != meta["sha256"]:
                raise ValueError("V3 split changed " + name)
        return m
    tok = AutoTokenizer.from_pretrained(local_snapshot())
    seen = set()
    for split in ("train", "validation", "test", "ood"):
        with (v2.ROOT / (split + ".jsonl")).open() as f:
            seen.update(json.loads(line)["id"] for line in f)
    tmp = Path(tempfile.mkdtemp(dir=ROOT.parent, prefix=".v3-"))
    m = dict(
        dataset="v3",
        seed=3703,
        license="MIT",
        training_dataset="v2",
        splits={},
        generator={
            f: digest(Path(__file__).with_name(f))
            for f in ("v3.py", "graphs.py", "reasoning.py")
        },
        exclusions=digest(v2.ROOT / "manifest.json"),
    )
    try:
        for split, n in (
            ("development", 150),
            ("development_ood", 150),
            ("confirm", 300),
            ("confirm_ood", 300),
        ):
            rng = random.Random(f"3703/{split}")
            ids = []
            with (tmp / (split + ".jsonl")).open("w") as f:
                for i in range(n):
                    family = FAMILIES[i % 5]
                    level = (i // 5) % 3
                    ood = split.endswith("ood")
                    for _ in range(500):
                        if family == "dependencies":
                            shift = (
                                ("topology" if (i // 15) % 2 else "operand")
                                if ood
                                else "id"
                            )
                            r = graph(rng, level, shift)
                        else:
                            # Hold ID operands/operations fixed in the isolated wording stratum.
                            shift = (
                                ("wording" if (i // 15) % 2 else "combined-v2")
                                if ood
                                else "id"
                            )
                            r = problem(rng, family, level, shift == "combined-v2")
                            if shift == "wording":
                                r["question"] = (
                                    "Determine the requested value. " + r["question"]
                                )
                        if r["id"] not in seen:
                            break
                    else:
                        raise ValueError("Could not generate a distinct instance")
                    seen.add(r["id"])
                    ids.append(r["id"])
                    r["shift"] = shift
                    r["variants"] = {
                        p: v2.encode_protocol(r, tok, p == "trace")
                        for p in ("answer", "trace")
                    }
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
            m["splits"][split] = dict(
                count=n, ids=fingerprint(ids), sha256=digest(tmp / (split + ".jsonl"))
            )
        save(tmp / "manifest.json", m)
        tmp.rename(ROOT)
    finally:
        if tmp.exists():
            rmtree(tmp)
    return m


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
