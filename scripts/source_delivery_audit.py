"""Count joint per-request delivery improvements, including individual regressions."""

from collections import Counter, defaultdict
from pathlib import Path

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, digest
from loopqwen.research.conditional_primary import costs
from scripts.assistance_deliberation import write_csv
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

STUDIES = (
    ("staged-query", "staged-query-transfer-01"),
    ("learner-prefix", "learner-prefix-transfer-01"),
    ("bounded-read", "bounded-read-transfer-01"),
)
DIRECTIONS = {
    "wall_s": -1,
    "first_s": -1,
    "peak": -1,
    "opening_tps": 1,
    "decode_tps": 1,
}
DOC = Path("docs/measurements")


def run():
    outputs = []
    sources = {}
    for cohort, study in STUDIES:
        review = DOC / (cohort + "-reading.json")
        if not review.exists():
            continue
        root = Path("research/results") / study
        for p in (review, root / "checksums.json"):
            committed(p)
        weights = read(root / "weights.json")
        sources[study] = dict(
            archive=digest(root / "checksums.json"),
            reading=digest(review),
            policy=weights["step"],
        )
        arms = [unpack(root / (arm + ".json.gz")) for arm in ("candidate", "control")]
        for a, b in zip(*arms, strict=True):
            assert a["id"] == b["id"] and a["family"] == b["family"]
            x, y = a["requests"][0], b["requests"][0]
            xc, yc = costs([x]), costs([y])
            r = dict(
                study=study, id=a["id"], family=a["family"], weights=weights["step"]
            )
            for k, sign in DIRECTIONS.items():
                aa, bb = xc[k], yc[k]
                r[k + "_loop"] = aa
                r[k + "_base"] = bb
                r[k + "_change"] = ((aa > bb) - (aa < bb)) * sign
            for k in ("success", "completed"):
                r[k + "_loop"] = int(x["score"][k])
                r[k + "_base"] = int(y["score"][k])
                r[k + "_change"] = r[k + "_loop"] - r[k + "_base"]
            r["faster_after_cold_start"] = int(r["wall_s_change"] > 0)
            r["wins_every_cost"] = int(all(r[k + "_change"] > 0 for k in DIRECTIONS))
            r["all_costs_and_delivered"] = int(
                r["wins_every_cost"] and r["success_loop"] and r["completed_loop"]
            )
            r["no_item_regression"] = int(
                all(
                    r[k + "_change"] >= 0 for k in (*DIRECTIONS, "success", "completed")
                )
            )
            outputs.append(r)
    summary = defaultdict(Counter)
    for r in outputs:
        for family in (r["family"], "all"):
            c = summary[(r["study"], family)]
            c["n"] += 1
            for k in (
                "wins_every_cost",
                "all_costs_and_delivered",
                "no_item_regression",
            ):
                c[k] += r[k]
            for k in (*DIRECTIONS, "success", "completed"):
                c[k + "_gains"] += r[k + "_change"] > 0
                c[k + "_losses"] += r[k + "_change"] < 0
    write_csv(Path("paper/source-delivery-requests.csv"), outputs)
    rows = [dict(study=k[0], family=k[1], **v) for k, v in summary.items()]
    write_csv(Path("paper/source-delivery-summary.csv"), rows)
    return sources, rows


if __name__ == "__main__":
    sources, rows = run()
    for study, obj in sources.items():
        persist(
            DOC / (study + "-joint-delivery.json"),
            dict(
                source=obj,
                groups=[r for r in rows if r["study"] == study],
                method="Evaluate both arms on all same-source requests using the original separately measured first-output wait, wall, per-request peak MLX bytes, newly processed opening tokens per second and post-first output tokens per second. A strict cost win improves all five for that one request. Delivered requires original objective success and completion. No-item-regression uses weak inequalities on the five costs, strict objective and completion. These descriptive individual observations depend on request order and latency noise. They do not replace the registered metrics, blind judgments or source-cluster significance.",
            ),
        )
    print([r for r in rows if r["family"] == "all"])
