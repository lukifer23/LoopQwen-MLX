"""Post-installation sensitivity of causal transplants to donor identity and state."""

import random
from pathlib import Path

from ..data.pointer5 import rows
from ..io import read, save, digest
from ..training.pointer5 import OUT
from .pointer5_diagnostics import load, transplant


def balanced(seed):
    cases = rows("cases")
    dev = rows("development")
    rng = random.Random(seed)
    rng.shuffle(dev)
    targets = set()
    selected = []
    for case in cases:
        r = case["original"]
        allowed = [node for node in range(8) if node != r["states"][0]]
        selected.append((r, allowed))
    # Bipartite assignment pairs 8 copies of each abstract state with eligible recipients.
    slots = [j for j in range(8) for _ in range(8)]
    rng.shuffle(slots)
    owners = {}

    def assign(i, seen):
        for j, node in enumerate(slots):
            if j in seen or node not in selected[i][1]:
                continue
            seen.add(j)
            if j not in owners or assign(owners[j], seen):
                owners[j] = i
                return True
        return False

    if not all(assign(i, set()) for i in range(len(cases))):
        raise ValueError("Cannot balance donor states")
    for j, i in sorted(owners.items()):
        r = cases[i]["original"]
        node = slots[j]
        donor = next(
            d
            for d in dev
            if d["map_id"] != r["map_id"]
            and d["map_id"] not in targets
            and d["states"][0] == node
            and d["states"][1] != r["map"][node]
        )
        targets.add(donor["map_id"])
        cases[i] = {**cases[i], "cross": donor}
    return cases


def diagnose(seed):
    cases = balanced(seed)
    for arm, step in (("recurrent", 480), ("outcome", 560)):
        model, tok, weights = load(seed, arm, step)
        path = OUT / str(seed) / f"balanced-donors-{arm}.json"
        ident = dict(
            weights=weights,
            implementation=digest(Path(__file__)),
            seed=seed,
            donors=[c["cross"]["example_id"] for c in cases],
            recipients=[c["original"]["example_id"] for c in cases],
        )
        if path.exists():
            if read(path)["identity"] != ident:
                raise ValueError("Incompatible donor sensitivity result")
            continue
        trial = transplant(model, tok, arm, cases, seed)
        save(
            path,
            dict(
                identity=ident,
                records=[r for r in trial["records"] if r["kind"] == "cross"],
                summary=trial["summary"]["cross/all"],
            ),
        )
