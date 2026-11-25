"""Replay one archived request per execution path with only the export's bundled runtime."""

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("export", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    local = args.export.resolve()
    sys.path.insert(0, str(local))
    from runtime import generate, load_export

    archived = args.study.resolve()
    spec = read(archived / "registration.json")
    panel = read(archived.parents[1] / "data" / (spec["id"] + ".json"))
    samples = read(archived / "candidate.json.gz")["records"]
    if len(samples) != len(panel) or not read(archived / "decision.json")["passed"]:
        raise ValueError("A fully qualified integrated model is required")
    if (
        digest(local / "context-policy.json")
        != read(archived / "decision.json")["weights"]
    ):
        raise ValueError("Export belongs to a different measured model")
    model, tokenizer = load_export(local)
    if set(model.context_policy.views()) - {"ordinary"} != set(spec["experts"]):
        raise ValueError("Freshly loaded specialist inventory differs")
    selected = {}
    for item, row in zip(panel, samples, strict=True):
        if item["id"] != row["id"]:
            raise ValueError("Archive order differs")
        key = (row["context_route"], row.get("recurrent_from") is not None)
        if key not in selected or len(item["prompt_ids"]) < len(
            selected[key][0]["prompt_ids"]
        ):
            selected[key] = item, row
    probes = []
    for (route, switched), (row, expected) in sorted(selected.items()):
        text = tokenizer.decode(row["prompt_ids"])
        if tokenizer.encode(text, add_special_tokens=False) != row["prompt_ids"]:
            raise ValueError("The bundled tokenizer changed the request")
        val = generate(model, tokenizer, text, row["cap"], stop_ids=(248046,))
        if any(
            val.get(k) != expected.get(k)
            for k in (
                "token_ids",
                "context_route",
                "recurrent_from",
                "attention_memory",
            )
        ):
            raise ValueError("The export changed an archived deployment response")
        probes.append(
            dict(
                id=row["id"],
                route=route,
                switched=switched,
                input_tokens=len(row["prompt_ids"]),
                output_tokens=len(val["token_ids"]),
                elapsed_s=val["elapsed_s"],
            )
        )
    result = dict(
        archive=digest(archived / "checksums.json"),
        verification=read(local / "verification.json"),
        probes=probes,
    )
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
