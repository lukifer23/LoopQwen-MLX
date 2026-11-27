"""Separate matched initial-context work from opening the models' actual tool returns."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest
from scripts.assistance_deliberation import write_csv
from scripts.register_source_copy import PAIR, RESULT
from scripts.table_policy_material import persist


def run():
    root = RESULT / PAIR
    arms = ("candidate", "control")
    records = [unpack(root / (a + ".json.gz")) for a in arms]
    groups = defaultdict(list)
    clusters = []
    for left, right in zip(*records, strict=True):
        assert left["id"] == right["id"]
        cluster = [0.0, 0.0, 0.0]
        for lreq, rreq in zip(left["requests"], right["requests"], strict=True):
            stage = lreq["history"]
            assert stage == rreq["history"]
            if stage in ("first", "matched"):
                lt, rt = (q["trajectory"]["turns"][0] for q in (lreq, rreq))
                assert (
                    lt["prompt_ids"] == rt["prompt_ids"]
                    and lt["reused_prompt_tokens"] == rt["reused_prompt_tokens"] == 0
                )
                cluster[0] += len(lt["prompt_ids"])
                cluster[1] += lt["ttft_s"]
                cluster[2] += rt["ttft_s"]
            for arm, req in zip(arms, (lreq, rreq), strict=True):
                for j, t in enumerate(req["trajectory"]["turns"]):
                    groups[arm, stage if j == 0 else "callback"].append(t)
        clusters.append(cluster)
    rows = []
    for (arm, stage), turns in groups.items():
        total = sum(len(t["prompt_ids"]) for t in turns)
        reused = sum(t["reused_prompt_tokens"] for t in turns)
        seconds = sum(t["ttft_s"] for t in turns)
        rows.append(
            dict(
                arm=arm,
                stage=stage,
                opens=len(turns),
                tokens=total,
                reused=reused,
                uncached=total - reused,
                seconds=seconds,
                tokens_per_second=(total - reused) / seconds,
            )
        )
    write_csv(Path("paper/source-copy-opening-cost.csv"), rows)
    seed = 82403
    x = np.array(clusters)

    def diff(data):
        return data[..., 0] / data[..., 1] - data[..., 0] / data[..., 2]

    sums = x.sum(axis=0)
    draws = np.random.default_rng(seed).integers(0, len(x), size=(10000, len(x)))
    low, high = np.quantile(diff(x[draws].sum(axis=1)), [0.025, 0.975])
    out = dict(
        primary=digest(root / "checksums.json"),
        clusters=len(x),
        matched_presentations=2 * len(x),
        matched_tokens=int(sums[0]),
        loop_tps=sums[0] / sums[1],
        base_tps=sums[0] / sums[2],
        difference=diff(sums),
        interval=[low, high],
        seed=seed,
        method="Retrospective decomposition. Verify complete identical initial input IDs and zero prefix reuse on each first and common-history follow-up. Resample 48 whole sources 10000 times, retaining both systems and both presentations. Wall time and the registered overall cost comparisons retain extra tool turns. The interval measures source variation within one fixed machine run.",
    )
    persist(Path("docs/measurements/source-copy-opening-cost.json"), out)
    print(out)


if __name__ == "__main__":
    run()
