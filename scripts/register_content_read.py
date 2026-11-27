"""Bind every source span to actual model tokens and freeze the new architecture's direct test."""

import re
from fractions import Fraction
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen import qwen_tools
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from scripts.content_read_material import DIR, SEED, persist_rows
from scripts.dialogue_database_sources import DEST
from scripts.learner_prefix_material import DATA
from scripts.native_history_sources import OUT as SQL
from scripts.program_language_material import answer, call, observed, user
from scripts.provider_recovery_material import mount_databases
from scripts.register_learner_prefix import FIT as PARENT, RESULT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

FIT = "content-read-fit-01"
PRIMARY = "content-read-transfer-01"
EX = Path("research/experiments")
PANEL = DIR / "held.json.gz"
TARGETS = DIR / "tokens.json.gz"
PLAN = Path("research/diagnostics/content-read-direct.json")


def extra():
    cases = []
    with TemporaryDirectory() as tmp:
        services = mount_databases(read(DEST / "originals.json"), Path(tmp))
        for pair in read(SQL)["pairs"]:
            db = pair["database"]
            schema, reader = services[db]
            args = {"database_name": db + ".sqlite"}
            tools = qwen_tools.declarations([schema, reader])
            for a, b in (pair["requests"], pair["requests"][::-1]):
                for prev, group in ((a, "reuse"), (b, "redirect")):
                    obs = reader.callback(**args, statement=prev["sql"])
                    assert obs == prev["returned"]
                    hist = [
                        user(a["question"]),
                        call(schema, **args),
                        observed(schema.callback(**args)),
                        call(reader, **args, statement=prev["sql"]),
                        observed(obs),
                    ]
                    target = (
                        answer(a["human"])
                        if group == "reuse"
                        else call(reader, **args, statement=a["sql"])
                    )
                    cases.append(
                        dict(
                            id="cosql/" + a["dialogue"] + "/" + group,
                            group=group,
                            cluster=db,
                            messages=hist,
                            tools=tools,
                            target=target,
                            anchors=[],
                        )
                    )
    previous = unpack(DATA)
    for group in (
        "instruction",
        "explanation",
        "arithmetic",
        "legacy",
        "qualitative",
        "program",
        "full_rows",
        "service_action",
    ):
        rows = sorted(
            (r for r in previous if r["group"] == group),
            key=lambda r: fingerprint([SEED, "retain", r["id"]]),
        )[:24]
        cases.extend(dict(**r, anchors=[]) for r in rows)
    return cases


def bind(tok, r):
    p, y = token_pair(tok, r)
    if not r["anchors"]:
        return p, y, []
    template = qwen_tools.render(tok, r["messages"], r["tools"])
    request = r["messages"][0]["content"]
    source_start = template.index(request.strip()) - (
        len(request) - len(request.lstrip())
    )
    suffix = tok.decode(y)

    def offsets(text, ids):
        enc = tok._tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        assert enc["input_ids"] == ids
        return enc["offset_mapping"]

    inp = offsets(template, p)
    out = offsets(suffix, y)
    annotations = []
    for a, b in r["anchors"]:
        needle = request[a:b]
        if r["group"] == "calculate":
            amount = abs(Fraction(needle.replace(",", "").replace("−", "-")))
            start = next(
                m.start()
                for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", suffix)
                if Fraction(m[0]) == amount
            )
        else:
            start = suffix.index(needle)
        j = next(i for i, (left, right) in enumerate(out) if left <= start < right)
        spans = [
            i
            for i, (l, h) in enumerate(inp)
            if l < source_start + b and h > source_start + a
        ]
        assert spans and all(i < len(p) for i in spans)
        # One decoding decision at the exact first token of each supported span.
        annotations.append(dict(query=j, source=[[spans[0], spans[-1] + 1]]))
    return p, y, annotations


def run():
    sources = [
        DIR / "inventory.json",
        DIR / "training.json.gz",
        PANEL,
        SQL,
        DATA,
        RESULT / PARENT / "checksums.json",
        Path("docs/measurements/learner-prefix-calculator-audit.json"),
    ]
    for p in sources:
        committed(p)
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    material = unpack(DIR / "training.json.gz") + extra()
    tokens = []
    groups = defaultdict(list)
    for r in sorted(material, key=lambda w: fingerprint([SEED, "all", w["id"]])):
        p, y, marks = bind(tok, r)
        # SQL targets and rehearsal retain the original public message, with no enforced maximum at inference.
        tokens.append(
            dict(
                id=r["id"],
                group=r["group"],
                region="source",
                p=p,
                y=y,
                annotations=marks,
            )
        )
        groups[r["group"]].append(r["id"])
    assert len(tokens) == len({w["id"] for w in tokens})
    persist_rows(TARGETS, tokens)
    base = read(RESULT / PARENT / "registration.json")
    inputs = {str(p): digest(p) for p in [*sources, TARGETS]}
    proto = "One direct comparison on all 24 original HotpotQA yes/no comparison questions, using all ten published distractor-context passages, and 24 previously unused TAT-QA development tables. Both get the identical opening token IDs, official reasoning-enabled template, ordered requests, greedy unforced generation, 384 tokens per turn and three turns, and the same actual calculator when applicable. Preserve every emitted token, observation, completion and source answer. Grade masked original responses first. No annotated facts, evidence indices, source parser, model selection or externally supplied thought in deployment."
    gate = "Within each family require six net new strict successes and six net new fully usable grounded answers, with zero lost prior successes and two-sided exact paired p<=0.05 on both endpoints. Require no increase in unsupported replies or missed completions per family. Require lower peak MLX, total wall and summed first-token seconds and higher opening and decode tokens/s; a miss closes downstream confirmation. At a pass separately verify the same digest on long full sources, broad retention, real SQL error histories, same-donor equally trained ordinary depth and another seed."
    persist(
        PLAN,
        dict(
            question="Can the tied content reader generalize to new source-grounded explanations and correct executed arithmetic?",
            material=digest(PANEL),
            design=proto,
            gate=gate,
            ordered=[w["id"] for w in unpack(PANEL)["rows"]],
            limits=dict(
                seconds_per_arm=1500, total=3000, peak_gib=3, available_gib=1.5
            ),
        ),
    )
    s = {
        k: base[k]
        for k in (
            "checkpoint",
            "base_text_sha256",
            "config_sha256",
            "donor",
            "passes",
            "shared_query",
            "revisits",
            "rank",
            "gate_rank",
            "compute_precision",
        )
    }
    s.update(
        id=FIT,
        content_read=True,
        parent=PARENT,
        fit_seed=SEED,
        training=str(TARGETS),
        sources={**inputs, str(PLAN): digest(PLAN)},
        probes=[
            k
            for group, keys in sorted(groups.items())
            for k in sorted(keys, key=lambda i: fingerprint([SEED, "probe", i]))[:6]
        ],
        epochs=3,
        lr=0.00055,
        teacher_kl=0.12,
        evidence_weight=8.0,
        micro_positions=20,
        update_positions=240,
        expected_positions=3 * sum(len(w["y"]) for w in tokens),
        expected_citations=3 * sum(len(w["annotations"]) for w in tokens),
        limits=dict(wall_seconds=3300, peak_gib=3, available_gib=1.5),
        question="Does learning a distance-independent content address over the unshortened original values improve source-relevant decisions at each recurrent visit?",
        protocol="Initialize exactly from the frozen learner-prefix loop. Leave its entire recurrent step and donor unchanged and learn only the shared address and write projections. Train on the full ordered supervised corpus for three seeded epochs. Include two different human arithmetic questions on each of 160 reports, source-exact supporting sentences in natural HotpotQA comparisons, 324 different human database requests under valid and unrelated outcomes on original databases, and eight protected earlier task groups. Maximize the exact likelihood of the designated current source span at its output decision. Add full-vocabulary distillation against the frozen parent. Require matching zero-initialization on the probe set, a gain >0.02 nats in reasoning, calculator actions and delivered quantities, mean protected loss increase <=0.05, mean positive SQL improvement, improved evidence NLL >=0.10, fitted content-read advantage >0.01 and positive second physical read advantage >0.002. Require finite gradients, exact exposure, replay and unchanged parent tensors. Stop at failed prerequisite or resource cap.",
    )
    persist(EX / (FIT + ".json"), s)
    direct = {
        k: s[k]
        for k in (
            "checkpoint",
            "base_text_sha256",
            "config_sha256",
            "donor",
            "passes",
            "revisits",
        )
    }
    direct.update(
        id=PRIMARY,
        fit_stage=FIT,
        material=str(PANEL),
        sources={
            **s["sources"],
            str(EX / (FIT + ".json")): digest(EX / (FIT + ".json")),
        },
        limits=dict(wall_seconds=3000, peak_gib=3, available_gib=1.5),
        seconds_per_arm=1500,
        max_tokens=384,
        max_turns=3,
        context_turns=1,
        question=read(PLAN)["question"],
        protocol=proto + " " + gate,
        content_read=True,
    )
    persist(EX / (PRIMARY + ".json"), direct)
    print(len(tokens), s["expected_positions"], s["expected_citations"])


if __name__ == "__main__":
    run()
