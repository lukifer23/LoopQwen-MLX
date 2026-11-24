"""Verify published snapshots against their original AdamW generations on the training host."""

import argparse
import json
import struct
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopqwen.io import digest, read


def header(stream):
    size = struct.unpack("<Q", stream.read(8))[0]
    fields = json.loads(stream.read(size))
    fields.pop("__metadata__", None)
    return fields


def verify(folder):
    pointer = read(folder / "recovery.json")
    generation = Path(pointer["generation"])
    state = read(folder / "state.json")
    metadata_digest = digest(folder / "state.json")
    if (
        digest(folder / "adapters.safetensors")
        != digest(generation / "adapters.safetensors")
        or digest(folder / "adapters.safetensors") != pointer["weights"]
        or metadata_digest != digest(generation / "state.json")
        or metadata_digest != pointer.get("metadata_sha256", metadata_digest)
        or digest(generation / "optimizer.safetensors") != pointer["optimizer_sha256"]
    ):
        raise ValueError(f"Training recovery digest mismatch for {folder}")

    with (folder / "adapters.safetensors").open("rb") as stream:
        weights = header(stream)
    with (generation / "optimizer.safetensors").open("rb") as stream:
        opt = header(stream)
        offset = stream.tell()
        expected = {
            f"{key}.{moment}": v["shape"]
            for key, v in weights.items()
            for moment in ("m", "v")
        }
        if set(opt) != set(expected) | {"step", "learning_rate"} or any(
            opt[key]["shape"] != shape for key, shape in expected.items()
        ):
            raise ValueError(f"AdamW schema mismatch for {folder}")
        cursor = opt["step"]
        if cursor["dtype"] != "U64" or cursor["shape"] != []:
            raise ValueError(f"Invalid AdamW step encoding for {folder}")
        stream.seek(offset + cursor["data_offsets"][0])
        step = struct.unpack("<Q", stream.read(8))[0]
    if step != state["step"]:
        raise ValueError(f"AdamW step differs from metadata for {folder}")
    print(f"{folder}: verified update {step}, {len(weights)} trainable tensors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="+")
    for path in parser.parse_args().folder:
        verify(path)
