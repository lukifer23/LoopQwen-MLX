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


def check(result, tok, tools):
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
    print("Reconstructed all release 05 field examples and both raw-text comparisons.")


if __name__ == "__main__":
    main()
