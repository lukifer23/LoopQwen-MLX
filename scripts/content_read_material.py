"""Reserve real human source comparisons and same-report arithmetic for a recurrent content read."""

import ast
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import pyarrow.parquet as pq

from loopqwen import qwen_tools
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from scripts.learner_prefix_material import DATA
from scripts.learner_prefix_panel import PANEL as PRIOR
from scripts.program_language_material import answer, call, observed, user
from scripts.shared_query_material import CALCULATOR, ROOT as OLD, TOOLS
from scripts.table_policy_material import persist

DIR = Path("research/sources/content-read")
CACHE = Path("data/research_external")
SEED = 6943301
HOT = {
    "train-00000-of-00002": "76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68",
    "train-00001-of-00002": "713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f",
    "validation-00000-of-00001": "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
}
FIN = {
    "train": "2df6e722cdbaaa37efcbfb280f5c9a15be29a6ec18f618ef936fe63cc6d07c69",
    "dev": "8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5",
}
NUM = re.compile(r"(?<![\w.])[+−-]?\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])")


def windows(text):
    words = re.findall(r"[a-z]+|\d+", text.lower())
    return {" ".join(words[i : i + 16]) for i in range(len(words) - 15)}


def prior_text():
    text = set()
    for r in unpack(DATA):
        for m in r["messages"][:1]:
            text.update(windows(m["content"]))
    for p in (OLD / "held.json.gz", PRIOR):
        obj = unpack(p)
        for r in obj if isinstance(obj, list) else obj["rows"]:
            if "request" in r:
                text.update(windows(r["request"]))
    for file in ("report-action-panel", "report-action-training"):
        obj = read(Path("research/data") / (file + ".json"))
        for r in obj.get("rows", []):
            if r["family"] == "report":
                text.update(windows(r["questions"][0]))
        for r in obj.get("examples", []):
            for m in r["messages"]:
                if m["role"] == "user":
                    text.update(windows(m["content"]))
    return text


def science(split):
    pool = []
    for key, sha in HOT.items():
        if not key.startswith(split):
            continue
        path = CACHE / "hotpotqa" / (key + ".parquet")
        assert digest(path) == sha
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for w in batch.to_pylist():
                if w["answer"] not in ("yes", "no") or w["type"] != "comparison":
                    continue
                ctx = w["context"]
                sf = w["supporting_facts"]
                sentences = dict(zip(ctx["title"], ctx["sentences"], strict=True))
                facts = [
                    sentences[t][i].strip()
                    for t, i in zip(sf["title"], sf["sent_id"], strict=True)
                    if t in sentences and 0 <= i < len(sentences[t])
                ]
                if (
                    len(set(sf["title"])) != 2
                    or len(facts) != len(sf["title"])
                    or not 2 <= len(facts) <= 4
                    or not all(facts)
                ):
                    continue
                body = "\n\n".join(
                    t + "\n" + " ".join(s.strip() for s in ss)
                    for t, ss in sentences.items()
                )
                if (
                    len(body) > 6500
                    or sum(map(len, facts)) > 670
                    or len(sentences) != len(ctx["title"])
                ):
                    continue
                pool.append(
                    dict(
                        id="hotpotqa/" + w["id"],
                        family="strategy",
                        source="HotpotQA distractor " + split,
                        cluster=w["id"],
                        titles=ctx["title"],
                        facts=facts,
                        support_titles=sf["title"],
                        support_ids=sf["sent_id"],
                        label=w["answer"] == "yes",
                        question=w["question"],
                        request=body + "\n\n" + w["question"],
                        reference=w["answer"].capitalize() + ". " + " ".join(facts),
                        decomposition=[],
                    )
                )
    return sorted(pool, key=lambda w: fingerprint([SEED, w["id"]]))


def finance(split):
    path = CACHE / "tatqa" / ("tatqa_dataset_" + split + ".json")
    assert digest(path) == FIN[split]
    pool = []
    for doc in read(path):
        table = "\n".join(" | ".join(r) for r in doc["table"]["table"])
        body = table + "\n\n" + "\n\n".join(p["text"] for p in doc["paragraphs"])
        if len(body) > 5600:
            continue
        by_value = defaultdict(list)
        for m in NUM.finditer(table):
            val = Fraction(m[0].replace(",", "").replace("−", "-"))
            by_value[abs(val)].append(list(m.span()))
        cases = []
        for q in doc["questions"]:
            if q["answer_type"] != "arithmetic":
                continue
            expr = re.sub(
                r"(?<=\d),(?=\d{3}(?:\D|$))",
                "",
                q["derivation"].replace("−", "-").replace("×", "*").replace("÷", "/"),
            ).strip()
            if not re.fullmatch(r"[\d. ()+*/-]{3,125}", expr):
                continue
            try:
                tree = ast.parse(expr, mode="eval").body
                if not isinstance(tree, ast.BinOp):
                    continue
                ref = Fraction(str(q["answer"]))
                steps = [expr]
                obs = CALCULATOR.callback(steps=steps)
                if q["scale"] == "percent" and abs(
                    Fraction(obs["value"]) - ref
                ) > Fraction(1, 200):
                    steps.append("r1*100")
                    obs = CALCULATOR.callback(steps=steps)
                if abs(Fraction(obs["value"]) - ref) > Fraction(1, 200):
                    continue
                source_tree = tree
                if (
                    "average" in q["question"].lower()
                    and isinstance(tree.op, ast.Div)
                    and isinstance(tree.right, ast.Constant)
                ):
                    source_tree = tree.left
                literals = {
                    abs(Fraction(ast.get_source_segment(expr, n)))
                    for n in ast.walk(source_tree)
                    if isinstance(n, ast.Constant)
                }
                if not literals or any(len(by_value[v]) != 1 for v in literals):
                    continue
            except (SyntaxError, ValueError, ArithmeticError):
                continue
            spans = [by_value[v][0] for v in sorted(literals)]
            cases.append(
                dict(
                    id="tatqa/" + q["uid"],
                    family="report",
                    source="TAT-QA " + split,
                    cluster=doc["table"]["uid"],
                    question=q["question"],
                    request=body + "\n\n" + q["question"],
                    reference=(str(q["answer"]) + " " + q["scale"]).strip(),
                    scale=q["scale"],
                    program=q["derivation"],
                    steps=steps,
                    observed=obs,
                    gold=str(ref),
                    tolerance="1/200",
                    operand_spans=spans,
                    evidence=[table],
                )
            )
        if cases:
            pool.append(
                (
                    doc["table"]["uid"],
                    body,
                    sorted(cases, key=lambda w: fingerprint([SEED, w["id"]])),
                )
            )
    return sorted(pool, key=lambda w: fingerprint([SEED, w[0]]))


def persist_rows(path, obj):
    if path.exists():
        assert unpack(path) == obj
    else:
        packed(path, obj)


def build():
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    prior = prior_text()
    held = []
    train = []
    blocked = set()
    titles = set()

    def enroll(w):
        defs = TOOLS if w["family"] == "report" else []
        ids = tok.encode(
            qwen_tools.render(tok, [user(w["request"])], defs), add_special_tokens=False
        )
        if len(ids) > 1800:
            return False
        held.append(
            dict(
                **w,
                defs=defs,
                questions=[w["request"]],
                initial_ids=ids,
                initial_sha256=fingerprint(ids),
            )
        )
        return True

    balance = Counter()
    for w in science("validation"):
        if balance[w["label"]] >= 12 or titles.intersection(w["titles"]):
            continue
        tokens = windows(w["request"])
        if tokens & prior or tokens & blocked:
            continue
        if enroll(w):
            balance[w["label"]] += 1
            titles.update(w["titles"])
            blocked.update(tokens)
        if len(held) == 24:
            break
    assert dict(balance) == {False: 12, True: 12}, balance
    for _, body, cases in finance("dev"):
        tokens = windows(body)
        if tokens & prior or tokens & blocked:
            continue
        if enroll(cases[0]):
            blocked.update(tokens)
        if len(held) == 48:
            break
    assert len(held) == 48

    def add(w, suffix, group, msgs, target, anchors):
        defs = TOOLS if w["family"] == "report" else []
        row = dict(
            id=w["id"] + "/" + suffix,
            group=group,
            cluster=w["cluster"],
            tools=defs,
            messages=msgs,
            target=target,
            anchors=anchors,
        )
        p, y = token_pair(tok, row)
        if len(p) + len(y) > 2100 or len(y) > 230:
            return False
        train.append(row)
        return True

    balance.clear()
    for w in science("train"):
        if (
            balance[w["label"]] >= 160
            or titles.intersection(w["titles"])
            or blocked.intersection(windows(w["request"]))
        ):
            continue
        msg = user(w["request"])
        # Archive exact character spans in the original request. Each cited sentence has its own target.
        if any(w["request"].count(f) != 1 for f in w["facts"]):
            continue
        anchors = [
            [w["request"].index(f), w["request"].index(f) + len(f)] for f in w["facts"]
        ]
        if add(
            w,
            "answer",
            "reasoning",
            [msg],
            answer(
                w["reference"],
                "Compare " + " and ".join(dict.fromkeys(w["support_titles"])) + ".",
            ),
            anchors,
        ):
            balance[w["label"]] += 1
        if sum(balance.values()) == 320:
            break
    pairs = 0
    from itertools import combinations

    for _, body, cases in finance("train"):
        if blocked.intersection(windows(body)):
            continue
        choice = next(
            (
                (a, b)
                for a, b in combinations(cases, 2)
                if a["gold"] != b["gold"]
                and a["operand_spans"] != b["operand_spans"]
                and a["question"] != b["question"]
            ),
            None,
        )
        if choice is None:
            continue
        oldlen = len(train)
        for w in choice:
            action = call(CALCULATOR, steps=w["steps"])
            msg = user(w["request"])
            add(w, "action", "calculate", [msg], action, w["operand_spans"])
            add(
                w,
                "reply",
                "quantity",
                [msg, action, observed(w["observed"])],
                answer(w["reference"]),
                [],
            )
        if len(train) - oldlen != 4:
            del train[oldlen:]
            continue
        pairs += 1
        if pairs == 160:
            break
    assert min(balance.values()) >= 120 and pairs >= 64, (balance, pairs)
    held.sort(key=lambda w: fingerprint([SEED, "held", w["id"]]))
    train.sort(key=lambda w: fingerprint([SEED, "training", w["id"]]))
    assert (
        len({r["id"] for r in train}) == len(train)
        and len({r["id"] for r in held}) == 48
    )
    DIR.mkdir(parents=True, exist_ok=True)
    persist_rows(DIR / "held.json.gz", {"rows": held})
    persist_rows(DIR / "training.json.gz", train)
    inventory = dict(
        hotpot_files=HOT,
        hotpot_revision="1908d6afbbead072334abe2965f91bd2709910ab",
        tatqa_files=FIN,
        seed=SEED,
        held=digest(DIR / "held.json.gz"),
        training=digest(DIR / "training.json.gz"),
        pairs=pairs,
        training_groups=dict(Counter(w["group"] for w in train)),
        held_groups=dict(Counter(w["family"] for w in held)),
        question="Can an additional learned content read at each recurrent visit distinguish the quantities and facts relevant to the current request?",
    )
    persist(DIR / "inventory.json", inventory)
    print(inventory, flush=True)


if __name__ == "__main__":
    build()
