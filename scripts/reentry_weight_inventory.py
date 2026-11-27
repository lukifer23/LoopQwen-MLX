"""Account for every Base parameter in the MLX four-bit pack and compare unchanged tensors bytewise."""

import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np

from loopqwen.io import digest, read, save
from scripts.assistance_deliberation import write_csv

SPEC = Path("research/experiments/table-reentry-fit-01.json")
DTYPES = {"BF16": "<u2", "F32": "<f4", "U32": "<u4"}


class MappedTensors:
    def __init__(self, path):
        with path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            self.entries = json.loads(f.read(size))
        self.entries.pop("__metadata__", None)
        self.data = np.memmap(path, mode="r", dtype="u1", offset=8 + size)

    def array(self, name):
        entry = self.entries[name]
        start, end = entry["data_offsets"]
        return self.data[start:end].view(DTYPES[entry["dtype"]]).reshape(entry["shape"])


def widened(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def audit():
    s = read(SPEC)
    original = Path(s["checkpoint"]) / "model.safetensors"
    packed = Path("exports/reentry-backbone-01/qwen-base-prefix20-q4.safetensors")
    receipt = Path("outputs/research/table-reentry-fit-01/packed-base.json")
    if not receipt.exists():
        receipt = Path("research/results/table-reentry-fit-01/packed-base.json")
    model = read(receipt)
    assert digest(original) == model["base"] == s["base_text_sha256"]
    assert digest(packed) == model["sha256"]
    a, b = MappedTensors(original), MappedTensors(packed)
    accounted = set()
    rows = []
    totals = Counter()
    squared_error = squared_reference = 0.0
    for name, e in sorted(a.entries.items()):
        source = a.array(name)
        target = b.array(name)
        accounted.add(name)
        where = (
            "embedding"
            if name.startswith("model.embed_tokens.")
            else name.split(".")[2]
            if name.startswith("model.layers.")
            else "output"
        )
        original_n = source.size
        err = power = largest = 0.0
        if b.entries[name]["dtype"] == "U32":
            assert (
                (where == "embedding" or int(where) < 20)
                and name.endswith(".weight")
                and e["dtype"] == "BF16"
            )
            prefix = name.removesuffix("weight")
            scales, biases = (b.array(prefix + k) for k in ("scales", "biases"))
            accounted.update(prefix + k for k in ("scales", "biases"))
            assert source.shape == (target.shape[0], target.shape[1] * 8)
            assert (
                scales.shape == biases.shape == (source.shape[0], source.shape[1] // 64)
            )
            shifts = np.arange(8, dtype=np.uint32) * 4
            for i in range(0, source.shape[0], 256):
                sel = slice(i, i + 256)
                original_f = widened(source[sel])
                codes = (
                    ((target[sel, :, None] >> shifts) & 15)
                    .astype(np.float32)
                    .reshape(len(original_f), -1, 64)
                )
                restored = (
                    codes * widened(scales[sel])[:, :, None]
                    + widened(biases[sel])[:, :, None]
                ).reshape(original_f.shape)
                diff = restored - original_f
                err += np.sum(np.square(diff), dtype=np.float64)
                power += np.sum(np.square(original_f), dtype=np.float64)
                largest = max(largest, float(np.max(np.abs(diff))))
            squared_error += err
            squared_reference += power
            method = "native Q4, group 64"
            totals["quantized_matrices"] += 1
            totals["quantized_parameters"] += original_n
        else:
            assert (
                e["dtype"] == b.entries[name]["dtype"] and source.shape == target.shape
            )
            assert np.array_equal(source.view("u1"), target.view("u1"))
            method = "byte-identical"
            totals["unchanged_tensors"] += 1
            totals["unchanged_parameters"] += original_n
            if where.isdigit() and int(where) >= 20:
                totals["unchanged_suffix_tensors"] += 1
                totals["unchanged_suffix_parameters"] += original_n
        rows.append(
            {
                "tensor": name,
                "layer": where,
                "parameters": original_n,
                "method": method,
                "relative_rms_error": float(np.sqrt(err / power)) if power else 0.0,
                "max_abs_error": largest,
            }
        )
    assert accounted == set(b.entries)
    table = Path("paper/reentry-backbone-tensors.csv")
    write_csv(table, rows)
    report = {
        "base_sha256": model["base"],
        "native_pack_sha256": model["sha256"],
        "base_bytes": original.stat().st_size,
        "native_pack_bytes": packed.stat().st_size,
        "original_tensors": len(a.entries),
        "packed_tensors": len(b.entries),
        **totals,
        "quantized_coordinates_relative_rms_error": float(
            np.sqrt(squared_error / squared_reference)
        ),
        "table_sha256": digest(table),
        "method": "Read physical native safetensors directly. For packed weights unpack low-nibble-first words and apply each 64-value affine scale and bias in FP32. Relative RMS is the square root of total squared reconstruction error divided by total squared original weight. All remaining tensor bytes must agree exactly.",
    }
    dest = Path("docs/measurements/table-reentry-backbone.json")
    if dest.exists():
        assert read(dest) == report
    else:
        save(dest, report)
    print(report)


if __name__ == "__main__":
    audit()
