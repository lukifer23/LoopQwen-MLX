"""Recount full-history register conversions and complete isolated deployment streams."""

from pathlib import Path

from ..evaluation.mechanism import unpack
from ..io import digest, read


def material(spec):
    mat = read(spec["material"])
    if "register_material" in spec:
        source = Path(spec["register_material"])
        if digest(source) != spec["register_material_sha256"]:
            raise ValueError("Long retention input differs from the frozen document")
        mat["register"] = read(source)[0]
    return mat


def qualify(decision, data, mat, spec):
    reference = unpack(Path(spec["historical_register"]) / "direct_pages.json.gz")
    rows_ok = True
    streams = {}
    for side, records in data.items():
        streams[side] = []
        for i, r in enumerate(records):
            z = r["register"]
            n = len(z["token_ids"])
            rows_ok &= (
                n > 1
                and z["prompt_tokens"] >= 131072
                and z["token_ids"] == reference[i]["token_ids"]
            )
            rows_ok &= all(
                v.get("fused_queries", 0) == (n - 1 if side == "candidate" else 0)
                for v in z["page_sites"]
            )
            if side == "candidate":
                q = z["weight_precision"]
                ids = mat["register"]["panel"][i]["prompt_ids"]
                rows_ok &= (
                    q["bits"] == 8
                    and len(q["projections"]) == 72
                    and len(q["shared_adapter_projections"]) == 18
                )
                rows_ok &= q["actual_projections"] == n and q[
                    "prompt_embedding_rows"
                ] == len(set(ids))
                rows_ok &= q["final_embedding_rows"] == len(
                    set(ids + z["token_ids"][:-1])
                )
                rows_ok &= all("weight_precision" not in short for short in r["short"])
            else:
                rows_ok &= "weight_precision" not in z
            streams[side].append(
                {
                    "seconds": r["stream_seconds"],
                    "rss": r["high_water_rss_bytes"],
                    "decode_tps": z["decode_tps"],
                    "register_peak": z["peak_bytes"],
                    "register_seconds": z["elapsed_s"],
                    "register_first": z["ttft_s"],
                }
            )
    a, b = streams["candidate"], streams["control"]
    portable = (
        all(
            x["decode_tps"] >= (1 + spec["rate_margin"]) * y["decode_tps"]
            for x, y in zip(a, b, strict=True)
        )
        and all(
            x["register_peak"] < y["register_peak"] for x, y in zip(a, b, strict=True)
        )
        and max(r["rss"] for r in a) <= max(r["rss"] for r in b)
        and sum(r["seconds"] for r in a) <= sum(r["seconds"] for r in b)
    )
    decision.update(
        full_rows_and_outputs=bool(rows_ok),
        streams=streams,
        strict_resources=portable,
        passed=bool(decision["passed"] and rows_ok and portable),
    )
    return decision
