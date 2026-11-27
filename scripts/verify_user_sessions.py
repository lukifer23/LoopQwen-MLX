"""Reconstruct the installed user-facing examples cited in the release 05 report."""

from pathlib import Path
from mlx_lm.utils import load_tokenizer

from loopqwen import REVISION, file_tools, package_tools
from loopqwen.calculator_tool import tool
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.model_loader import local_snapshot
from loopqwen.research.native_function_study import replay as qwen_replay
from loopqwen.research.mixed_field import replay_exchange
from loopqwen.tool_session import system_message


def check(result, tok, tools, *, after_callback=False):
    assert result["base_revision"] == REVISION and result["initial_prompt_identical"]
    sides = result["runs"]
    a, b = sides["recurrent"], sides["base"]
    assert [
        x["initial_sha256"] == y["initial_sha256"] for x, y in zip(a, b, strict=True)
    ] == result["conversation_prompt_matches"]
    assert result["conversation_prompt_matches"][0]
    spec = dict(max_turns=result["max_turns"], max_tokens=result["max_tokens_per_turn"])
    native = result.get("tool_format") == "qwen"
    for side, rows in sides.items():
        history = [] if native else [system_message(tools)]
        for r in rows:
            history.append(dict(role="user", content=r["question"]))
            ids = r["turns"][0]["prompt_ids"]
            assert (
                fingerprint(ids) == r["initial_sha256"]
                and len(ids) == r["initial_tokens"]
            )
            assert r["output_tokens"] == sum(len(t["token_ids"]) for t in r["turns"])
            if native:
                q = dict(
                    trajectory=r,
                    wall_s=r["request_seconds"],
                    first_s=r["first_token_seconds"],
                    peak=r["peak_mlx_bytes"],
                )
                qwen_replay(
                    tok,
                    None,
                    tools,
                    history,
                    q,
                    spec,
                    "candidate" if side == "recurrent" else "control",
                    after_callback=after_callback,
                )
            else:
                replay_exchange(tok, tools, history, r, spec)


def main():
    docs = Path("docs/measurements")
    results = Path("research/results")
    tok = load_tokenizer(local_snapshot())
    environment = read("research/data/native-function-panel.json")["environment"]
    native = read(docs / "native-function-installed-cli.json")
    native_root = results / "native-function-dialogue-01"
    assert native["checkpoint"] == digest(native_root / "checksums.json")
    file = Path("research/experiments/native-function-dialogue-01.json")
    assert digest(file) == native["file_sha256"]
    assert native["file_reference"] == [
        file.stat().st_size,
        len(file.read_bytes().splitlines()),
    ]
    assert native["field_reference"] == [
        environment[x]["version"] for x in ("numpy", "mlx")
    ]
    for obs, tools in zip(
        native["observations"],
        [package_tools.tools(environment), file_tools.tools([str(file)])],
        strict=True,
    ):
        check(obs["result"], tok, tools)
    wheel = read(docs / "native-function-wheel.json")
    assert (
        wheel["outside_checkout"]
        and wheel["offline"]
        and wheel["weights"] == digest(native_root / "tool-adapters.safetensors")
    )
    assert wheel["bundle"] == digest(
        results / "packed-release-retention-04/export-files.json"
    )
    for r, obs in zip(wheel["sessions"], native["observations"], strict=True):
        original = obs["result"]["runs"]["recurrent"]
        assert r["complete_trace_matches"] == [True] * len(original)
        assert r["questions"] == [x["question"] for x in original] and r["answers"] == [
            x["answer"] for x in original
        ]
    path = Path("exports/wheels") / wheel["wheel"]
    if path.is_file():
        assert (
            digest(path) == wheel["wheel_sha256"]
            and path.stat().st_size == wheel["wheel_bytes"]
        )
    recovery = read(docs / "callback-recovery-installed-cli.json")
    assert recovery["checkpoint"] == digest(
        results / "callback-recovery-01/checksums.json"
    )
    for r in recovery["requests"]:
        check(r["result"], tok, [tool()])
    from verify_runtime_wheel import trace

    review = read(docs / "observation-review-installed-cli.json")
    assert review["policy"] == digest(results / "observation-review-01/checksums.json")
    identity = read(docs / "observation-review-base-identity.json")
    protocol = read(results / "observation-review-01/registration.json")
    assert identity["export_sha256"] == protocol["base_text_sha256"]
    assert identity["revision"] == REVISION and identity[
        "review_adapter_sha256"
    ] == digest(results / "observation-review-01/tool-adapters.safetensors")
    assert identity["review_sites"] == identity["opening_sites"] + [[23, 1]]
    assert (
        identity["exact_text_tensors"] > 0
        and identity["shared_quantized_and_unquantized_tensors"]
        >= identity["exact_text_tensors"]
    )
    assert all(
        rows[0]["question"] == review["question"]
        for rows in review["result"]["runs"].values()
    )
    assert (
        review["outside_checkout"] and review["offline"] and review["reference"] == 704
    )
    assert (
        trace(review["result"]["runs"]["recurrent"][0])
        == review["independent_candidate_trace"]
    )
    check(review["result"], tok, [tool()], after_callback=True)
    dist = Path("exports/observation-review-runtime") / review["wheel"]
    if dist.is_file():
        assert (
            digest(dist) == review["wheel_sha256"]
            and dist.stat().st_size == review["wheel_bytes"]
        )
    from document_frequency_material import documents
    from loopqwen.generate_cli import render
    from loopqwen.evaluation.mechanism import unpack

    portable = read(docs / "portable-base-wheel.json")
    document = read(docs / "document-reader-installed.json")
    root = results / "document-frequency-reader-01"
    assert document["policy"] == digest(root / "checksums.json")
    document_row = unpack(root / "candidate.json.gz")[0]
    _, _, original, question, expected, _ = next(documents())
    rendered, _ = render(tok, original + "\n\n" + question, True)
    assert document["rendered_prompt_sha256"] == fingerprint(rendered)
    assert (
        document["reference"] == expected and document["grade"] == document_row["grade"]
    )
    assert all(
        document[k] == document_row[k]
        for k in ("token_ids", "context_route", "attention_memory", "prompt_tokens")
    )
    contrast = read(docs / "contrast-reader-installed.json")
    assert contrast["policy"] == digest(results / "question-contrast-01/checksums.json")
    assert contrast["references"] == [
        identity["shared_quantized_and_unquantized_tensors"]
        - identity["exact_text_tensors"],
        identity["exact_text_tensors"],
    ]
    assert contrast["result"]["base_source"] == "certified_release_text"
    check(contrast["result"], tok, [tool()])
    assert all(
        obs["offline"]
        and obs["outside_checkout"]
        and obs["network_connections_blocked"]
        and obs["empty_model_cache"]
        and obs["wheel_sha256"] == portable["sha256"]
        for obs in (document, contrast)
    )
    checked = read(docs / "checked-calculator-installed-cli.json")
    assert checked["registration"] == digest(
        results / "checked-equations-field-01/registration.json"
    )
    check(checked["result"], tok, [tool(check_equalities=True)])
    unicode = read(docs / "unicode-package-installed-cli.json")
    assert unicode["checkpoint"] == digest(
        results / "packed-release-retention-04/checksums.json"
    )
    check(unicode["result"], tok, package_tools.tools(environment))
    for x, y in zip(
        unicode["result"]["runs"]["recurrent"],
        unicode["offline"]["exchanges"],
        strict=True,
    ):
        keys = ("prompt_ids", "token_ids", "call", "observation")
        assert [[t.get(k) for k in keys] for t in x["turns"]] == [
            [t.get(k) for k in keys] for t in y["turns"]
        ]
    raw = read(docs / "source-return-installed-cli.json")
    assert raw["checkpoint"] == digest(
        results / "selective-raw-return-01/checksums.json"
    )
    for q in raw["requests"]:
        r = q["result"]
        assert r["base_revision"] == REVISION and r["initial_prompt_identical"]
        ids = tok.encode(r["prompt"], add_special_tokens=False)
        for v in r["runs"].values():
            assert v["initial_sha256"] == fingerprint(ids) and v[
                "prompt_tokens"
            ] == len(ids)
            assert (
                v["text"] == tok.decode(v["token_ids"])
                and 0 < v["cold_first_seconds"] <= v["request_seconds"]
            )
    assert (
        portable["empty_model_cache"]
        and portable["network_connections_blocked"]
        and portable["outside_checkout"]
    )
    assert portable["source_inventory"] == digest(
        results / "packed-release-retention-04/export-files.json"
    )
    for session in portable["sessions"]:
        source = read(docs / session["source"])
        index = session["index"]
        if session["kind"] == "functions":
            rows = source["observations"][index]["result"]["runs"]
            assert session["identical_trajectories"] == {
                k: len(v) for k, v in rows.items()
            }
            assert session["base_trace_sha256"] == fingerprint(
                [trace(r) for r in rows["base"]]
            )
        else:
            row = source["requests"][index]["result"]["runs"]["base"]
            assert (
                session["initial_sha256"],
                session["output_sha256"],
                session["tokens"],
            ) == (
                row["initial_sha256"],
                fingerprint(row["token_ids"]),
                len(row["token_ids"]),
            )
    path = Path("exports/portable-base-runtime") / portable["wheel"]
    if path.is_file():
        assert (
            digest(path) == portable["sha256"]
            and path.stat().st_size == portable["bytes"]
        )
    print(
        "Verified installed field examples, free generations and the portable Base control."
    )


if __name__ == "__main__":
    main()
