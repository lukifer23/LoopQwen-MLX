"""Report sampling-unit sensitivity and family counts without refitting or resampling models."""

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read


def audit(name):
    root = Path("research/results") / name
    spec = read(root / "registration.json")
    data = read(spec["material"])
    rows = data["rows"] if isinstance(data, dict) else data
    arms = [unpack(root / (side + ".json.gz")) for side in ("candidate", "control")]
    groups = defaultdict(list)
    families = defaultdict(lambda: [0, 0, 0])
    for w, a, b in zip(rows, *arms, strict=True):
        assert w["id"] == a["id"] == b["id"]
        av, bv = int(a["score"]["success"]), int(b["score"]["success"])
        group = w["domain"] if "domain" in w else w["function"][0]["name"]
        groups[group].append(av - bv)
        vals = families[w.get("kind", group)]
        vals[0] += av
        vals[1] += bv
        vals[2] += 1
    effects = {g: sum(d) for g, d in groups.items()}
    nonzero = [d for d in effects.values() if d]
    target = abs(sum(nonzero))
    if len(nonzero) > 20:
        raise ValueError("Exact enumeration intended for small registered group panels")
    count = sum(
        abs(sum(s * d for s, d in zip(signs, nonzero))) >= target
        for signs in itertools.product((-1, 1), repeat=len(nonzero))
    )
    return dict(
        families=dict(sorted(families.items())),
        group_differences=effects,
        group_signflip_p=count / 2 ** len(nonzero),
        net=sum(effects.values()),
        items=len(rows),
    )


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    print(json.dumps(audit(parser.parse_args().study), indent=2))
