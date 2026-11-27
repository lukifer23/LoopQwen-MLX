"""Replay a real uninterrupted Qwen callback through the public external-history continuation API."""

import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx

from loopqwen.conditional_reentry import ConditionalAssistant
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.relational_tools import tools
from loopqwen.research.native_function_study import executions
from loopqwen.research.table_worlds import SOURCE, locations, prepare_tables, prompt
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

FIT = Path("research/results/conditional-program-fit-01")
PLAN = Path("research/diagnostics/external-tool-native.json")
RESULT = Path("docs/measurements/external-tool-native.json")


def study(measure):
    source = unpack(FIT / "free-sql.json.gz")
    ids = sorted((r["id"] for r in source), key=lambda k: fingerprint([12069, k]))[:3]
    plan = {
        "fit": digest(FIT / "checksums.json"),
        "training_source": digest(SOURCE),
        "ordered": ids,
        "protocol": "Use up to three already enrolled training questions to obtain one uninterrupted actual callback and a following native turn from the archived conditional Qwen. Disable prefix reuse in both paths and use the same full official prompts, greedy 128-token budget and model. Resume the frozen external history without an extra user turn. Require byte-identical opening IDs, emitted IDs, resulting callback arguments and returns; require the input to remain unchanged. Each attempt is recorded. Stop at first available follow-up, 120 seconds, 2 GiB native peak, or a system warning. This engineering measurement has no held task scores.",
    }
    if not measure:
        persist(PLAN, plan)
        return
    import subprocess
    import psutil

    committed(PLAN)
    assert read(PLAN) == plan
    timers = perf_counter()

    def guard(*_):
        if perf_counter() - timers > 120 or mx.get_peak_memory() > 2 * 2**30:
            raise InterruptedError("Registered resource limit")
        if (
            psutil.virtual_memory().available < 1.5 * 2**30
            or subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            != "1"
        ):
            raise InterruptedError("Host pressure")

    mx.reset_peak_memory()
    cases = []
    passed = False
    status = "no follow-up produced"
    try:
        guard()
        spec = read(FIT / "registration.json")
        actor = ConditionalAssistant(spec["checkpoint"], FIT)
        rows = {r["id"]: r for r in read(SOURCE)["cohorts"]["train"]}
        with TemporaryDirectory() as tmp:
            dest = Path(tmp)
            prepare_tables(dest, [rows[k] for k in ids])
            for k in ids:
                providers = tools([locations(dest, rows[k])[0]])
                chat = actor.conversation(providers)
                opts = {"max_tokens": 128, "reuse_prefix": False, "progress": guard}
                tr = chat.respond(prompt(rows[k]), max_turns=2, **opts)
                cur = {
                    "id": k,
                    "original_status": tr["status"],
                    "turns": len(tr["turns"]),
                }
                cases.append(cur)
                if len(tr["turns"]) != 2:
                    continue
                stream = list(chat.messages)
                end = 2
                while end < len(stream) and stream[end]["role"] == "tool":
                    end += 1
                history = stream[:end]
                original = deepcopy(history)
                assert len(history) >= 3 and history[-1]["role"] == "tool"
                replay = actor.continue_after_tool(
                    history, providers, max_turns=1, **opts
                )
                expected = tr["turns"][1]
                got = replay["turns"][0]
                cur.update(
                    prompt_ids=expected["prompt_ids"],
                    emitted_ids=expected["token_ids"],
                    replay_prompt=got["prompt_ids"],
                    replay_emitted=got["token_ids"],
                    uninterrupted_callbacks=executions({"turns": [expected]}),
                    resumed_callbacks=executions(replay),
                    input_preserved=history == original,
                )
                passed = all(
                    (
                        cur["prompt_ids"] == cur["replay_prompt"],
                        cur["emitted_ids"] == cur["replay_emitted"],
                        cur["uninterrupted_callbacks"] == cur["resumed_callbacks"],
                        cur["input_preserved"],
                    )
                )
                status = "measured"
                break
    except (InterruptedError, MemoryError) as e:
        status = str(e)
    persist(
        RESULT,
        {
            "plan": digest(PLAN),
            "passed": passed,
            "status": status,
            "cases": cases,
            "native_seconds": perf_counter() - timers,
            "peak": mx.get_peak_memory(),
        },
    )
    print(status, passed, len(cases))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("freeze", "measure"))
    study(p.parse_args().mode == "measure")
