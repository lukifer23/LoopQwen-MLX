"""Independently read each observed peer color and both native matched-depth answers."""

import argparse
import csv
import gzip
import io
import json
import re
from math import comb
from pathlib import Path

STUDIES = (
    "native-induction-anchored-matched-01",
    "native-induction-anchored-replication-01",
)


def read(path):
    return json.loads(
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )


def observations(row):
    colors = {}
    species = {}
    for fact in row["facts"]:
        name, attribute = fact.removesuffix(".").split(" is ")
        if attribute.startswith(("a ", "an ")):
            kind = attribute.split(" ", 1)[1]
            assert species.setdefault(name, kind) == kind
        else:
            assert (
                attribute in {"gray", "green", "white", "yellow"}
                and colors.setdefault(name, attribute) == attribute
            )
    subject = re.fullmatch(r"What color is (\w+)\?", row["question"])[1]
    assert subject not in colors and subject in species
    peers = {
        name: color
        for name, color in colors.items()
        if species.get(name) == species[subject]
    }
    assert row["answer"] in peers.values()
    return peers


def audit(root):
    rows = []
    for study in STUDIES:
        folder = root / "research/results" / study
        spec = read(folder / "registration.json")
        items = read(root / spec["material"])[spec["panel"]]
        arms = [
            read(folder / (s + ".json.gz"))["records"] for s in ("recurrent", "dense")
        ]
        for item, a, b in zip(items, *arms, strict=True):
            assert (
                item["id"] == a["id"] == b["id"]
                and item["prompt_ids"] == a["prompt_ids"] == b["prompt_ids"]
            )
            peers = observations(item)
            options = set(peers.values())
            gold = item["answer"]
            result = {
                "split": spec["panel"],
                "id": item["id"],
                "question": item["question"],
                "facts": json.dumps(item["facts"]),
                "peer_observations": json.dumps(peers, sort_keys=True),
                "distinct_peer_colors": len(options),
                "publisher_answer": gold,
            }
            for side, model in (("loop", a), ("dense", b)):
                assert model["complete"] and model["stop"] == "chat_end"
                reply = re.fullmatch(
                    r"The answer is (\w+)\.", model["visible"].strip()
                )[1]
                assert reply == model["answer"] and (reply == gold) == model["correct"]
                result.update(
                    {
                        side + "_answer": reply,
                        side + "_exact": reply == gold,
                        side + "_peer_consistent": reply in options,
                    }
                )
            rows.append(result)
    assert len({r["id"] for r in rows}) == len(rows) == 128
    groups = []
    for split in ("primary", "replication", "both"):
        for subset in ("all", "one peer color", "multiple peer colors"):
            items = [
                r
                for r in rows
                if split in ("both", r["split"])
                and (
                    subset == "all"
                    or (subset == "one peer color") == (r["distinct_peer_colors"] == 1)
                )
            ]
            for metric in ("exact", "peer_consistent"):
                a = [r["loop_" + metric] for r in items]
                b = [r["dense_" + metric] for r in items]
                wins = sum(x and not y for x, y in zip(a, b, strict=True))
                losses = sum(y and not x for x, y in zip(a, b, strict=True))
                n = wins + losses
                p = min(
                    1.0,
                    2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n,
                )
                groups.append(
                    {
                        "split": split,
                        "observations": subset,
                        "score": metric,
                        "items": len(items),
                        "loop": sum(a),
                        "dense": sum(b),
                        "loop_gains": wins,
                        "loop_losses": losses,
                        "paired_p": p,
                    }
                )
    return rows, groups


def render(root, verify=False):
    rows, groups = audit(root)
    for name, data in (
        ("induction-peer-requests", rows),
        ("induction-peer-sensitivity", groups),
    ):
        out = io.StringIO(newline="")
        writer = csv.DictWriter(out, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
        target = root / "paper" / (name + ".csv")
        if verify:
            assert target.read_bytes() == out.getvalue().encode()
        else:
            target.write_bytes(out.getvalue().encode())
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    summaries = render(Path(__file__).resolve().parents[1], parser.parse_args().verify)
    print(
        [
            (s["observations"], s["score"], s["loop"], s["dense"], s["items"])
            for s in summaries
            if s["split"] == "both"
        ]
    )
