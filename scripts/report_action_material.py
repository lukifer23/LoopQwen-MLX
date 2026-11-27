"""Freeze question-dependent action pairs from intact financial reports and a real package lock."""

import re
import subprocess
import tomllib
from fractions import Fraction
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

from mlx_lm.utils import load_tokenizer
from program_reasoning_material import sha
from question_contrast_material import TOOLS, action, answer, call_and_return, example

from loopqwen import qwen_tools
from loopqwen.calculator_tool import tool
from loopqwen.evaluation.answers import numeric
from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.research.native_function_data import catalog

REV = "870accc41953dcde885aabeb963d94aabdc0fbc3"
SOURCES = {
    "train": "2df6e722cdbaaa37efcbfb280f5c9a15be29a6ec18f618ef936fe63cc6d07c69",
    "dev": "8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5",
}
LOCK = "7848bcc"
DATA = Path("research/data")
CALC = tool()
INSTRUCTION = "\nGive the numeric answer in the units used in the source. Use the calculator if arithmetic is needed."
DEC = re.compile(r"\d+(?:\.\d+)?")
GROUP = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")


def source(split):
    p = Path("data/research_external/tatqa") / f"tatqa_dataset_{split}.json"
    if not p.exists():
        p.parent.mkdir(exist_ok=True, parents=True)
        with urlopen(
            f"https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{REV}/dataset_raw/{p.name}",
            timeout=60,
        ) as response:
            raw = response.read()
        assert sha(raw.decode()) == SOURCES[split]
        p.write_bytes(raw)
    assert digest(p) == SOURCES[split]
    return read(p)


def quantity(value):
    if isinstance(value, list):
        if len(value) != 1:
            return None
        value = value[0]
    text = str(value).strip().strip("$£€").strip().removesuffix("%").strip()
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    return numeric(text)


def eligible(split):
    pairs = []
    for d in source(split):
        context = (
            "\n".join(" | ".join(row) for row in d["table"]["table"])
            + "\n\n"
            + "\n\n".join(p["text"] for p in d["paragraphs"])
        )
        if len(context.encode()) > 4200:
            continue
        operands = {Fraction(m) for m in DEC.findall(GROUP.sub("", context))}
        qs = {"span": [], "arithmetic": []}
        for q in d["questions"]:
            kind = q["answer_type"]
            if kind not in qs:
                continue
            gold = quantity(q["answer"])
            if gold is None:
                continue
            if kind == "span":
                if abs(Fraction(gold)) not in operands:
                    continue
                expr = None
            else:
                expr = (
                    GROUP.sub("", q["derivation"].strip())
                    .replace("−", "-")
                    .replace("×", "*")
                    .replace("÷", "/")
                )
                if not re.fullmatch(r"[\d. ()+*/-]{3,128}", expr):
                    continue
                nums = [Fraction(v) for v in DEC.findall(expr)]
                if len(set(nums)) < 2 or not set(nums) <= operands:
                    continue
                try:
                    actual = CALC.callback([expr])["value"]
                except (ValueError, ZeroDivisionError, SyntaxError, KeyError):
                    continue
                if Fraction(actual) != Fraction(gold):
                    continue
            qs[kind].append((q, gold, expr))
        options = [
            (a, b)
            for a in qs["span"]
            for b in qs["arithmetic"]
            if a[1] != b[1]
            and a[0]["scale"] == b[0]["scale"]
            and a[0]["question"] != b[0]["question"]
        ]
        if options:
            a, b = min(
                options,
                key=lambda pair: sha(
                    "report-pair-808" + pair[0][0]["uid"] + pair[1][0]["uid"]
                ),
            )
            pairs.append((sha(context), context, [a, b]))
    return sorted(pairs, key=lambda t: sha("report-source-310" + t[0]))


def fixed(p, data):
    if p.exists():
        assert read(p) == data
    else:
        save(p, data)


def material():
    dev = eligible("dev")
    train = eligible("train")
    print("Eligible source reports", len(train), len(dev), flush=True)
    assert len(dev) >= 24
    dev = dev[:24]
    dev_q = {x[0]["question"].casefold().strip() for _, _, qs in dev for x in qs}

    def shingles(s):
        words = re.findall(r"[a-z]+|\d+", s.casefold())
        return {tuple(words[i : i + 16]) for i in range(len(words) - 15)}

    forbidden = set().union(*(shingles(s) for _, s, _ in dev))
    train = [
        v
        for v in train
        if not shingles(v[1]) & forbidden
        and all(q[0]["question"].strip().casefold() not in dev_q for q in v[2])
    ][:144]
    assert len(train) == 144
    fit = []
    manifest = []
    for group, s, qs in train:
        m = []
        for i, (q, g, expr) in enumerate(qs):
            ident = q["uid"]
            user = s + "\n\n" + q["question"] + INSTRUCTION
            # Preserve canonical decimals rather than teaching a rational serialization to users.
            visible = str(float(Fraction(g))) if Fraction(g).denominator != 1 else g
            target = action([expr]) if expr else answer(visible)
            fit.append(
                dict(
                    example(ident, user, target), pair=group, pair_order=i, thinking=""
                )
            )
            if expr:
                fit.append(
                    example(
                        ident + "-after", user, answer(visible), call_and_return([expr])
                    )
                )
            m.append({"uid": ident, "expression": expr, "gold": g})
        manifest.append({"id": group, "questions": m})
    previous = read(DATA / "native-function-training.json")["examples"]
    fit.extend(
        sorted(previous, key=lambda r: sha("report-rehearsal-363" + r["id"]))[:320]
    )
    rows = []

    def add(group, context, ident, q, g, expr, family):
        rows.append(
            {
                "id": ident,
                "cluster": group,
                "questions": [context + "\n\n" + q + INSTRUCTION],
                "gold": g,
                "expression": expr,
                "family": family,
            }
        )

    for group, s, qs in dev:
        for q, g, expr in qs:
            add(group, s, q["uid"], q["question"], g, expr, "report")
    lock = subprocess.check_output(["git", "show", LOCK + ":uv.lock"])
    pkgs = [
        p
        for p in tomllib.loads(lock.decode())["package"]
        if 3 <= len(p.get("wheels", [])) <= 16
        and len({w["size"] for w in p["wheels"]}) == len(p["wheels"])
    ]
    for p in sorted(pkgs, key=lambda p: sha("package-227" + p["name"]))[:8]:
        group = p["name"]
        ws = sorted(
            [
                (unquote(urlsplit(w["url"]).path.rsplit("/", 1)[1]), w["size"])
                for w in p["wheels"]
            ]
        )
        a, b = sorted(
            sorted(ws, key=lambda w: sha("package-target-810" + w[0]))[:2],
            key=lambda w: w[1],
        )
        s = (
            f"Distribution files for {group} {p['version']}\nFilename | Bytes\n"
            + "\n".join(f"{name} | {size}" for name, size in ws)
        )
        add(
            group,
            s,
            group + "-size",
            f"How many bytes is {a[0]}?",
            str(a[1]),
            None,
            "distribution",
        )
        add(
            group,
            s,
            group + "-change",
            f"How many more bytes does {b[0]} occupy than {a[0]}?",
            str(b[1] - a[1]),
            f"{b[1]} - {a[1]}",
            "distribution",
        )
    native = read(DATA / "question-contrast-panel.json")
    rows.extend(
        r for r in native["rows"] if r["family"] in {"public", "files", "packages"}
    )
    tok = load_tokenizer(local_snapshot())
    for r in rows:
        decl = (
            TOOLS
            if r["family"] in {"report", "distribution"}
            else qwen_tools.declarations(catalog(r))
        )
        ids = tok.encode(
            qwen_tools.render(
                tok, [{"role": "user", "content": r["questions"][0]}], decl
            ),
            add_special_tokens=False,
        )
        assert ids == r.get("initial_ids", ids)
        r["initial_ids"] = ids
    fixed(
        DATA / "report-action-training.json",
        {"examples": fit, "pairs": manifest, "sources": SOURCES, "revision": REV},
    )
    fixed(
        DATA / "report-action-panel.json",
        {
            "rows": rows,
            "files": native["files"],
            "environment": native["environment"],
            "lock_sha256": sha(lock.decode()),
            "lock_revision": LOCK,
        },
    )
    print(len(fit), "training turns,", len(rows), "dialogues")


if __name__ == "__main__":
    material()
