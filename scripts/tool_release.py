"""Verify one offline tool-capable build across field calls, language replay and 128K."""

import argparse
import gc
import json
import re
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.local_packages import tools as package_tools
from loopqwen.data import retention3
from loopqwen.evaluation.generation_eval import generate
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research import bootstrap, ledger_curriculum
from loopqwen.research.core import Budget
from loopqwen.research.registered import registered
from loopqwen.tool_model import Assistant
from loopqwen.worker import child_lease
from scripts.export_tool_assistant import export

SUITES = [
    (
        "native_induction",
        "research/data/native-induction-anchored-01.json",
        "replication",
    ),
    ("deduction", "research/data/deduction-stories-02.json", "replication"),
    ("mixed", "research/data/conditional-language-02.json", None),
]


class QualificationError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise QualificationError(message)


def tokens(traj):
    return [
        (r["prompt_ids"], r["token_ids"], r.get("call"), r.get("observation"))
        for r in traj["turns"]
    ]


def field_question(name):
    return f"What version of the Python distribution {name} is installed in this environment? Use the available tool."


def field_score(result, name, value):
    calls = [t for t in result["turns"] if "observation" in t]
    return (
        result["status"] == "answered"
        and len(calls) == 1
        and calls[0]["call"]["name"] == "package_version"
        and str(calls[0]["call"]["arguments"].get("distribution", "")).casefold()
        == name.casefold()
        and calls[0]["observation"] == {"version": value}
        and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", result["answer"] or "")
        is not None
    )


def portable(bundle, name):
    # Import only the bundled runtime and environment dependencies in an isolated interpreter.
    code = """import json,sys
from pathlib import Path
from importlib.metadata import version,metadata
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
from runtime.tool_model import Assistant
from runtime.tool_session import Tool
assert not any(k=='loopqwen' or k.startswith('loopqwen.') for k in sys.modules)
schema=json.loads(sys.argv[3]); callbacks=[lambda distribution:{'version':version(distribution)},lambda distribution:{'summary':metadata(distribution).get('Summary','')}]
tools=[Tool(f['name'],f['description'],f['parameters'],c) for f,c in zip(schema,callbacks,strict=True)]
out=Assistant(root).respond(sys.argv[2],tools,max_tokens=256)
print(json.dumps(out))
"""
    args = [
        sys.executable,
        "-I",
        "-c",
        code,
        str(bundle.resolve()),
        field_question(name),
        json.dumps([t.definition() for t in package_tools()]),
    ]
    kw = child_lease()
    kw["env"].update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with tempfile.TemporaryDirectory() as temp:
        return json.loads(subprocess.check_output(args, cwd=temp, text=True, **kw))


def check_release(spec, out, budget, decision):
    trained = Path(spec["tools"])
    prior = Path(spec["language_record"])
    bundle = Path(spec["destination"])
    for p, h in [(trained, spec["tool_sha256"]), (prior, spec["language_sha256"])]:
        require(
            digest(p / "checksums.json") == h and read(p / "decision.json")["passed"],
            "Unqualified release parent",
        )
    prefix = None
    if spec.get("prefix_study"):
        study = Path(spec["prefix_study"])
        reg = read("research/registry.json")["experiments"][study.name]
        require(
            reg["registration"] == spec["prefix_registration"]
            and digest(study / "checksums.json") == reg["archive"]
            and digest(study / "registration.json") == reg["registration"]
            and all(
                digest(study / k) == v
                for k, v in read(study / "checksums.json").items()
            ),
            "Unverified prefix crossover",
        )
        if read(study / "decision.json")["passed"]:
            prefix = study
    for name, expected in spec.get("additional_qualifications", {}).items():
        directory = Path(name)
        require(
            digest(directory / "checksums.json") == expected
            and read(directory / "decision.json")["passed"],
            "Missing qualified tool retention control",
        )
    if not bundle.exists():
        export(
            Path(spec["language"]),
            prior,
            trained,
            trained,
            bundle,
            prefix,
            Path(spec["backbone_release"]) if "backbone_release" in spec else None,
        )
    meta = read(bundle / "tool/tool-model.json")
    require(
        meta["qualification_archive"] == spec["tool_sha256"]
        and meta["weights"]
        == read(trained / "fit.json")["weights"]
        == digest(bundle / "tool/tool-adapters.safetensors"),
        "Different tool export",
    )
    if spec.get("field_cli"):
        require(
            meta.get("prefix_reuse", {}).get("study")
            == (prefix.name if prefix else None),
            "Different prefix policy",
        )
    inventory = read(bundle / "tool-export-files.json")
    require(
        all(digest(bundle / k) == v for k, v in inventory.items()),
        "Export file identity",
    )
    save(out / "export-files.json", inventory)
    save(out / "tool-model.json", meta)
    assistant = Assistant(bundle)
    source = unpack(trained / "candidate.json.gz")
    from scripts.tool_study import workload

    training_spec = read(trained / "registration.json")
    inputs, catalog, _ = workload(training_spec)
    checks = []
    for w, expected in zip(inputs, source, strict=True):
        budget.check()
        tr = assistant.respond(
            w["user_request"],
            catalog(w),
            max_tokens=w.get("generation_cap", training_spec["max_tokens"]),
            max_turns=w.get("turn_cap", training_spec["max_turns"]),
            progress=lambda _: budget.check(),
        )
        checks.append(dict(id=w["id"], trajectory=tr))
        require(
            tokens(tr) == tokens(expected["trajectory"]),
            "Public tool replay differs: " + w["id"],
        )
    packed(out / "tool-replay.json.gz", checks)
    decision["tool_replayed"] = len(checks)
    field = []
    for name, value in spec["packages"].items():
        require(version(name) == value, "Environment changed for " + name)
        tr = assistant.respond(
            field_question(name),
            package_tools(),
            max_tokens=256,
            progress=lambda _: budget.check(),
        )
        field.append(
            dict(
                name=name,
                value=value,
                trajectory=tr,
                success=field_score(tr, name, value),
            )
        )
    packed(out / "field.json.gz", field)
    decision["field"] = sum(r["success"] for r in field)
    # Validate offline portability even when the optional field score is low.
    del assistant
    gc.collect()
    mx.clear_cache()
    replay = portable(bundle, field[0]["name"])
    packed(out / "portable.json.gz", replay)
    require(
        tokens(replay) == tokens(field[0]["trajectory"]),
        "Bundled runtime changed tool behavior",
    )
    decision["portable"] = True
    if spec.get("field_cli"):
        from scripts.field_cli_replay import measure

        records = measure(bundle, budget)
        packed(out / "field-cli.json.gz", records)
        decision["cli_correct"] = sum(r["correct"] for r in records)
        decision["cli_exact"] = sum(r["identical"] for r in records)
        decision["prefix_default"] = prefix is not None
        require(decision["cli_exact"] == 4, "Installed/isolated command disagreement")
    assistant = Assistant(bundle)
    # Trigger the real mode switch in the same instance before checking the language model.
    tr = assistant.respond(
        field_question(field[0]["name"]),
        package_tools(),
        max_tokens=256,
        progress=lambda _: budget.check(),
    )
    require(tokens(tr) == tokens(replay), "Mode-switch setup differs")
    assistant._load_language()
    model, tok = assistant.language, assistant.tokenizer
    for suite, file, key in SUITES:
        material = read(file)
        rows = material[key] if key else material
        reference = unpack(prior / (suite + ".json.gz"))["records"]
        records = []
        for row, expected in zip(rows, reference, strict=True):
            ids = row["prompt_ids"]
            text = tok.decode(ids)
            require(
                tok.encode(text, add_special_tokens=False) == ids,
                "Language input roundtrip",
            )
            val = generate(
                model,
                tok,
                text,
                row.get("cap", 768),
                stop_ids=[tok.convert_tokens_to_ids("<|im_end|>")],
                progress=lambda _: budget.check(),
            )
            r = dict(
                id=row["id"],
                input_sha256=fingerprint(ids),
                route=val.get("context_route"),
                token_ids=val["token_ids"],
                elapsed_s=val["elapsed_s"],
            )
            require(
                all(
                    r[k] == expected[k]
                    for k in ("id", "input_sha256", "route", "token_ids")
                ),
                "Language replay differs: " + row["id"],
            )
            records.append(r)
        packed(out / (suite + ".json.gz"), records)
        decision[suite] = len(records)
        print(suite, len(records), flush=True)
    retention3.build()
    raw = [
        json.loads(line)
        for line in (retention3.ROOT / "validation.jsonl").read_text().splitlines()
    ]
    for r in raw:
        view, route = model.context_policy.choose(model, r["ids"], 96)
        require(
            view is model.context_policy.ordinary and route == "ordinary",
            "Raw language route",
        )
    decision["retention"] = bootstrap.retention(
        model.context_policy.ordinary, 1, budget
    )
    require(
        decision["retention"] == read(prior / "decision.json")["retention"],
        "Language NLL changed",
    )
    decision["raw_requests"] = len(raw)
    context = []
    refs = unpack(prior / "context.json.gz")["records"]
    for row, expected in zip(
        read("research/data/conditional-language-128k.json")["panel"], refs, strict=True
    ):
        mx.reset_peak_memory()
        r = ledger_curriculum.measure(model, tok, row, 96, budget)
        require(
            r["correct"]
            and all(
                r[k] == expected[k]
                for k in (
                    "input_sha256",
                    "prompt_tokens",
                    "token_ids",
                    "context_route",
                    "attention_memory",
                )
            ),
            "Long-context response differs",
        )
        context.append(r)
        packed(out / "context.json.gz", context)
        print(
            "full history", len(context), r["prompt_tokens"], r["elapsed_s"], flush=True
        )
    decision["context"] = dict(
        correct=sum(r["correct"] for r in context),
        seconds=sum(r["elapsed_s"] for r in context),
        peak=max(r["peak_bytes"] for r in context),
    )
    require(decision["field"] >= spec["field_floor"], "Field call threshold")
    require(
        decision["context"]["seconds"] <= spec["long_time_limit"],
        "Long-context latency threshold",
    )


def verify(root, d):
    spec = read(root / "registration.json")
    if not d["passed"]:
        require(bool(d["reason"]), "No recorded failed check")
        return
    tools = Path(spec["tools"])
    prior = Path(spec["language_record"])
    if spec.get("field_cli"):
        from scripts.field_cli_replay import correct, identical

        field_records = unpack(root / "field-cli.json.gz")
        require(
            [r["kind"] for r in field_records]
            == ["packages", "catalog", "calculator", "sqlite"]
            and all(
                r["identical"]
                and identical(r["installed"], r["bundled"])
                and r["correct"] == correct(r)
                for r in field_records
            )
            and d["cli_correct"] == sum(r["correct"] for r in field_records)
            and d["cli_exact"] == 4,
            "Field CLI replay",
        )
        prefix_passed = bool(
            spec.get("prefix_study")
            and read(Path(spec["prefix_study"]) / "decision.json")["passed"]
        )
        require(
            d["prefix_default"]
            == prefix_passed
            == bool(read(root / "tool-model.json").get("prefix_reuse")),
            "Recorded prefix policy",
        )
    fields = unpack(root / "field.json.gz")
    require(
        [r["success"] for r in fields]
        == [field_score(r["trajectory"], r["name"], r["value"]) for r in fields],
        "Field scores differ",
    )
    require(
        d["field"] == sum(r["success"] for r in fields) >= spec["field_floor"],
        "Field summary",
    )
    require(
        tokens(unpack(root / "portable.json.gz")) == tokens(fields[0]["trajectory"]),
        "Offline model differs",
    )
    cases = unpack(root / "tool-replay.json.gz")
    require(
        d["tool_replayed"] == len(cases) == spec.get("expected_tool_rows", 22)
        and all(
            tokens(a["trajectory"]) == tokens(b["trajectory"]) and a["id"] == b["id"]
            for a, b in zip(cases, unpack(tools / "candidate.json.gz"), strict=True)
        ),
        "Tool replay",
    )
    for suite, _, _ in SUITES:
        out = unpack(root / (suite + ".json.gz"))
        ref = unpack(prior / (suite + ".json.gz"))["records"]
        require(
            d[suite] == len(out)
            and all(
                all(a[k] == b[k] for k in ("id", "input_sha256", "route", "token_ids"))
                for a, b in zip(out, ref, strict=True)
            ),
            "Language stream",
        )
    require(
        d["retention"] == read(prior / "decision.json")["retention"]
        and d["raw_requests"] == 120,
        "Raw prose retention",
    )
    long = unpack(root / "context.json.gz")
    old = unpack(prior / "context.json.gz")["records"]
    require(
        all(
            a["correct"]
            and all(
                a[k] == b[k]
                for k in (
                    "input_sha256",
                    "prompt_tokens",
                    "token_ids",
                    "attention_memory",
                    "context_route",
                )
            )
            for a, b in zip(long, old, strict=True)
        ),
        "Long replay",
    )
    require(
        d["context"]
        == dict(
            correct=2,
            seconds=sum(r["elapsed_s"] for r in long),
            peak=max(r["peak_bytes"] for r in long),
        )
        and d["context"]["seconds"] <= spec["long_time_limit"],
        "Resource summary",
    )


def main(name):
    spec = registered(name)
    out = Path("outputs/research") / name
    if (out / "decision.json").exists():
        raise ValueError("Existing run")
    out.mkdir(exist_ok=True, parents=True)
    budget = Budget(name, spec["limits"])
    decision = dict(passed=False)
    try:
        check_release(spec, out, budget, decision)
        decision["passed"] = True
    except QualificationError as exc:
        decision["reason"] = str(exc)
    finally:
        budget.close()
    save(out / "decision.json", decision)
    print(decision, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    main(p.parse_args().name)
