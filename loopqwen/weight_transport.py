"""Losslessly pack a qualified Base/instruction pair and restore its original file bytes."""

import argparse
import json
import math
import shutil
import struct
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .io import digest, read

BASE = "model.safetensors"
TOOL = "tool/quantized-backbone.safetensors"
INVENTORY = "tool-export-files.json"
BLOCK = 1 << 20


def relative(name):
    p = Path(name)
    if p.is_absolute() or not p.parts or ".." in p.parts or str(p) != name:
        raise ValueError("Invalid bundle member path")
    return p


class Tensors:
    def __init__(self, file, header=None):
        self.file = file
        if header is None:
            count = file.read(8)
            header = count + file.read(struct.unpack("<Q", count)[0])
        else:
            if len(header) < 8 or struct.unpack("<Q", header[:8])[0] != len(header) - 8:
                raise ValueError("Invalid safetensors header")
            file.write(header)
        self.header = header
        self.tensors = json.loads(header[8:])
        self.tensors.pop("__metadata__", None)
        end = 0
        widths = {
            "BF16": 2,
            "F16": 2,
            "F32": 4,
            "U32": 4,
            "I32": 4,
            "U8": 1,
            "I64": 8,
            "BOOL": 1,
            "I16": 2,
            "U16": 2,
        }
        for v in sorted(self.tensors.values(), key=lambda t: t["data_offsets"]):
            first, last = v["data_offsets"]
            if (
                first != end
                or (last - first) != math.prod(v["shape"]) * widths[v["dtype"]]
            ):
                raise ValueError("Unexpected tensor layout")
            end = last
        self.size = len(header) + end

    def visit(self, k, at, sz, data=None):
        off, end = self.tensors[k]["data_offsets"]
        if at < 0 or at + sz > end - off:
            raise ValueError("Tensor block outside declared shape")
        self.file.seek(len(self.header) + off + at)
        if data is not None:
            if len(data) != sz:
                raise ValueError("Wrong tensor block length")
            self.file.write(data)
            return
        buf = self.file.read(sz)
        if len(buf) != sz:
            raise ValueError("Incomplete tensor block")
        return buf


def shuffled(buf, reverse):
    x = np.frombuffer(buf, np.uint8)
    return (x.reshape(2, -1).T if reverse else x.reshape(-1, 2).T).tobytes()


def predictor(base, tool, key, start, count):
    # Predict on the target's own scale grid. Unsigned byte residuals correct every bit.
    # The float64 operations and round-to-even are part of the portable file format.
    prefix = key.removesuffix(".weight")
    pos = start * 2

    def floats(blob):
        return (
            (np.frombuffer(blob, "<u2").astype("<u4") << 16)
            .view("<f4")
            .astype(np.float64)
        )

    original = floats(base.visit(key, pos, count * 2)).reshape(-1, 64)
    scale = floats(tool.visit(prefix + ".scales", start // 32, count // 32))[:, None]
    bias = floats(tool.visit(prefix + ".biases", start // 32, count // 32))[:, None]
    q = np.divide(original - bias, scale, out=np.zeros_like(original), where=scale != 0)
    return np.clip(np.rint(q), 0, 255).astype(np.uint8).reshape(-1)


def transform(archive, side, source, base, decode):
    reverse = decode
    # Restore quantization scales before reading their associated packed weights.
    for k, v in sorted(
        source.tensors.items(), key=lambda kv: (kv[0].endswith(".weight"), kv[0])
    ):
        m = base.tensors.get(k) if side == "tool" else None
        delta = bool(
            m and m["dtype"] == v["dtype"] == "BF16" and m["shape"] == v["shape"]
        )
        prefix = k.removesuffix(".weight")
        quant = (
            m
            and m["dtype"] == "BF16"
            and v["dtype"] == "U32"
            and k.endswith(".weight")
            and prefix + ".scales" in source.tensors
            and prefix + ".biases" in source.tensors
            and math.prod(m["shape"]) == math.prod(v["shape"]) * 4
        )
        first, end = v["data_offsets"]
        length = end - first
        with archive.open(side + "/" + k, "r" if decode else "w") as stream:
            for start in range(0, length, BLOCK):
                n = min(BLOCK, length - start)
                x = stream.read(n) if decode else source.visit(k, start, n)
                if len(x) != n:
                    raise ValueError("Incomplete encoded block")
                if quant:
                    expected = predictor(base, source, k, start, n)
                    residual = np.frombuffer(x, np.uint8)
                    y = (
                        residual + expected if reverse else residual - expected
                    ).tobytes()
                elif delta:
                    other = np.frombuffer(base.visit(k, start, n), "<u2")
                    a = np.frombuffer(shuffled(x, True) if reverse else x, "<u2")
                    y = (
                        (a + other).astype("<u2").tobytes()
                        if reverse
                        else shuffled((a - other).astype("<u2").tobytes(), False)
                    )
                elif v["dtype"] == "BF16":
                    y = shuffled(x, reverse)
                else:
                    y = x
                if decode:
                    source.visit(k, start, n, y)
                else:
                    stream.write(y)
            if decode and stream.read(1):
                raise ValueError("Unexpected trailing tensor data")


def pack(bundle, dest):
    if dest.exists():
        raise FileExistsError(dest)
    listed = read(bundle / INVENTORY)
    listed[INVENTORY] = digest(bundle / INVENTORY)
    if any(digest(bundle / relative(k)) != h for k, h in listed.items()):
        raise ValueError("Source differs from export inventory")
    if not {BASE, TOOL} <= listed.keys():
        raise ValueError("Expected a two-backbone export")
    dest.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        packed = Path(tmp) / "artifacts.zip"
        with (
            zipfile.ZipFile(
                packed, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as z,
            ExitStack() as files,
        ):
            z.writestr("manifest.json", json.dumps({"format": 1, "files": listed}))
            base = Tensors(files.enter_context((bundle / BASE).open("rb")))
            tool = Tensors(files.enter_context((bundle / TOOL).open("rb")))
            for side, file in [("base", base), ("tool", tool)]:
                z.writestr(side + "/header", file.header)
                transform(z, side, file, base, False)
            for k in sorted(listed.keys() - {BASE, TOOL}):
                z.write(bundle / k, "files/" + k)
        packed.rename(dest)


def unpack(archive, dest):
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(exist_ok=True, parents=True)
    with (
        tempfile.TemporaryDirectory(dir=dest.parent) as tmp,
        zipfile.ZipFile(archive) as z,
    ):
        m = json.loads(z.read("manifest.json"))
        files = m["files"]
        root = Path(tmp) / "bundle"
        if m["format"] != 1 or not {BASE, TOOL, INVENTORY} <= files.keys():
            raise ValueError("Unsupported transport")
        for name in files:
            (root / relative(name)).parent.mkdir(exist_ok=True, parents=True)
        for k in sorted(files.keys() - {BASE, TOOL}):
            with z.open("files/" + k) as s, (root / k).open("xb") as f:
                shutil.copyfileobj(s, f)
        with ExitStack() as stack:
            base = Tensors(
                stack.enter_context((root / BASE).open("x+b")), z.read("base/header")
            )
            tool = Tensors(
                stack.enter_context((root / TOOL).open("x+b")), z.read("tool/header")
            )
            for side, tensors in [("base", base), ("tool", tool)]:
                transform(z, side, tensors, base, True)
        for k, h in files.items():
            if digest(root / k) != h:
                raise ValueError("Restored file checksum differs: " + k)
        root.rename(dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["pack", "unpack"])
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    (pack if a.action == "pack" else unpack)(a.source, a.destination)
