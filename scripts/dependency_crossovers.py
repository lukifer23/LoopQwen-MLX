"""Keep the dependency and its exact released version fixed while changing the original wheel."""

from itertools import combinations
from packaging.specifiers import SpecifierSet
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from scripts.dependency_sources import DEST, versions


def build():
    original = read(DEST / "originals.json")
    result = {}
    for split, cohort in original["cohorts"].items():
        pool = []
        for a, b in combinations(cohort, 2):
            dep = a["dependency"]
            if dep != b["dependency"]:
                continue
            left, right = map(SpecifierSet, (a["requirement"], b["requirement"]))
            different = [
                v
                for v in versions(dep, False)
                if left.contains(v[0]) != right.contains(v[0])
            ]
            if different:
                release = min(
                    different,
                    key=lambda v: fingerprint([771927, a["file"], b["file"], v[0]]),
                )
                pool.append(
                    {
                        "wheels": [a["file"], b["file"]],
                        "dependency": dep,
                        "release": release,
                        "decisions": [
                            left.contains(release[0]),
                            right.contains(release[0]),
                        ],
                    }
                )
        pool.sort(key=lambda w: fingerprint([771928, w["wheels"]]))
        seen = {}
        rows = []
        for w in pool:
            if any(seen.get(name, 0) >= 2 for name in w["wheels"]):
                continue
            for name in w["wheels"]:
                seen[name] = seen.get(name, 0) + 1
            rows.append(w)
            if len(rows) == 4:
                break
        assert len(rows) >= (4 if split == "held" else 2) and all(
            sum(r["decisions"]) == 1 for r in rows
        )
        result[split] = rows
    path = DEST / "same-version.json"
    if path.exists():
        assert read(path) == result
    else:
        save(path, result)
    print("Same-version constraints", {k: len(r) for k, r in result.items()})
    return result


if __name__ == "__main__":
    build()
