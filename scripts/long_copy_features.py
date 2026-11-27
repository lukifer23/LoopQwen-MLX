"""Extract frozen recurrent features for new version changes on complete real dependency files."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
import subprocess
import psutil
import mlx.core as mx

from loopqwen.identity import fingerprint
from loopqwen.io import read, digest
from loopqwen.long_training_memory import DocumentMemory
from loopqwen.research.conditional_features import CACHE
from loopqwen.research.native_function_data import token_pair
from loopqwen.shared_query import SharedQuery
from loopqwen.shared_query_assistant import SharedQueryAssistant
from scripts.changed_crate_sources import OUT as SOURCES
from scripts.long_branch_sources import SPECS, CACHE as RAW
from scripts.program_language_material import user, answer
from scripts.register_source_copy import FIT, RESULT
from scripts.table_policy_material import persist
from scripts.table_policy_reading import committed

PLAN = Path("research/diagnostics/long-copy-features.json")
OUT = Path("docs/measurements/long-copy-features.json")
ROOT = CACHE / "long-copy-features"


def run(freeze):
    fit = RESULT / FIT
    cold = Path("docs/measurements/source-copy-fit-01-cold.json")
    for p in (SOURCES, cold, fit / "checksums.json"):
        committed(p)
    assert read(cold)["passed"]
    s = dict(
        source=digest(SOURCES),
        model=digest(fit / "checksums.json"),
        implementation=digest(Path(__file__)),
        protocol="Feasibility and independent feature extraction for a long-training intervention. Use unchanged verified ordinary trunk and the complete frozen two-read source-copy parent. Stage both full release documents. Store one complete contextual-value prefix per version, the actual independently forked suffix states, every true visible version token and turn terminator, and the two vocabulary alternatives at the first divergent position of the other release and a nearby real crate. Keep the old Deno cohort out. No accuracy or weight updating in this step. Withhold all 24 new reserved names from all optimization and calibration. Require all 208 whole-file opening and target histories to match source IDs, physical histories retained at all six attention layers and chosen branches to reproduce after intervening questions. Limit 1350 native seconds, 3 GiB allocated and 1.5 GiB available. A failed extraction closes fitting.",
    )
    if freeze:
        persist(PLAN, s)
        return
    committed(PLAN)
    assert read(PLAN) == s
    bank = read(SOURCES)
    dest = ROOT / digest(PLAN)
    dest.mkdir(exist_ok=True, parents=True)
    start = monotonic()
    next_check = 0.0
    mx.reset_peak_memory()

    def guard(_=None):
        nonlocal next_check
        now = monotonic()
        if now < next_check:
            return
        next_check = now + 0.65
        if (
            now - start > 1350
            or mx.get_peak_memory() > 3 * 2**30
            or psutil.virtual_memory().available < 1.5 * 2**30
            or subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
            != "1"
        ):
            raise InterruptedError("full-context extraction limit")

    cfg = read(fit / "registration.json")
    actor = SharedQueryAssistant(cfg["checkpoint"], fit)
    model = actor._tool_model
    tok = actor.tokenizer
    trunk = SharedQuery(
        model.text, revisits=2, rank=cfg["rank"], gate_rank=cfg["gate_rank"]
    )
    trunk.eval()
    manifests = []
    failure = None
    try:
        for v, doc in enumerate(bank["documents"]):
            path = RAW / SPECS["deno"][3][v]
            assert digest(path) == doc["source"]
            docbody = (
                "Here is the complete Cargo.lock file.\n\n"
                + path.read_text()
                + "\nEnd of file.\n\n"
            )
            first = doc["rows"][0]
            full, official_target = token_pair(
                tok,
                dict(
                    messages=[user(docbody + first["question"])],
                    tools=[],
                    target=answer(first["value"]),
                ),
            )
            prefix = full[: doc["prefix_tokens"]]
            assert fingerprint(prefix) == doc["prefix"]
            entries = []

            def suffix(val):
                # The official assistant suffix is independent of the user message.
                return token_pair(
                    tok,
                    dict(
                        messages=[user("What is the exact version?")],
                        tools=[],
                        target=answer(val),
                    ),
                )[1]

            assert suffix(first["value"]) == official_target
            with TemporaryDirectory(dir=dest) as tmp:
                shared = DocumentMemory(trunk, prefix, tmp, guard)
                n = len(prefix)
                histories = [
                    slot.physical_length()
                    for slot in shared.disk.cache.slots.values()
                    if hasattr(slot, "physical_length")
                ]
                assert histories == [n] * 6
                kv = None
                for i in [*range(len(doc["rows"])), 0]:
                    guard()
                    r = doc["rows"][i]
                    p = prefix + r["tail"]
                    assert fingerprint(p) == r["input_sha256"]
                    y = suffix(r["value"])
                    obj = shared.branch(p, y)
                    if kv is None:
                        kv = mx.load(str(obj["prefix_file"]))
                        mx.eval(kv)
                        vp = dest / f"values-{v}.safetensors"
                        mx.save_safetensors(
                            str(vp), {"values": kv["v"], "ids": mx.array([prefix])}
                        )
                    pre = obj["pre"]
                    h = obj["ordinary"]
                    pos = obj["offset"]
                    k = mx.concatenate([kv["k"], obj["tail_keys"]], axis=2)
                    vals = mx.concatenate([kv["v"], obj["tail_values"]], axis=2)
                    text = tok.decode(y)
                    index = text.index(r["value"], text.index("</think>") + 8)
                    encoded = tok._tokenizer(
                        text, add_special_tokens=False, return_offsets_mapping=True
                    )
                    assert encoded["input_ids"] == y
                    lead = next(
                        j
                        for j, (a, b) in enumerate(encoded["offset_mapping"])
                        if a <= index < b
                    )

                    def branch_ids(alt):
                        other = suffix(alt)
                        j = next(j for j, (a, b) in enumerate(zip(y, other)) if a != b)
                        assert j >= lead
                        return [j - lead, other[j]]

                    contrasts = [
                        branch_ids(bank["documents"][1 - v]["rows"][i]["value"]),
                        branch_ids(r["foil"]["value"]),
                    ]
                    args = (pre[:, lead:], h[:, lead:], k, vals, pos + lead, "causal")
                    data = dict(
                        two=model.text.model.norm(model.refine(*args, visits=2)),
                        one=model.text.model.norm(model.refine(*args, visits=1)),
                        tail_values=obj["tail_values"],
                        tail_ids=mx.array([r["tail"] + y[:-1]]),
                        labels=mx.array([y[lead:]]),
                    )
                    file = dest / f"v{v}-{i}.safetensors"
                    digest_before = digest(file) if i == 0 and entries else None
                    mx.save_safetensors(str(file), data)
                    assert (
                        vals.shape[2] == pos + len(y)
                        and data["two"].shape[1] == len(y) - lead
                    )
                    if digest_before:
                        assert digest_before == digest(file) == entries[0]["sha256"]
                    else:
                        entries.append(
                            dict(
                                id=r["name"],
                                version=v,
                                source=r["tokens"],
                                foil=r["foil"]["tokens"],
                                offset=pos + lead,
                                positions=len(y) - lead,
                                alternatives=contrasts,
                                file=file.name,
                                sha256=digest(file),
                            )
                        )
                manifests.append(
                    dict(
                        prefix=vp.name,
                        sha256=digest(vp),
                        rows=entries,
                        histories=histories,
                    )
                )
            del kv, k, vals, shared, obj
            mx.clear_cache()
            print("original release features", v, len(entries), flush=True)
        guard()
    except (InterruptedError, RuntimeError, AssertionError) as e:
        import traceback

        failure = dict(error=repr(e), trace=traceback.format_exc())
    persist(
        OUT,
        dict(
            passed=failure is None and len(manifests) == 2,
            plan=digest(PLAN),
            versions=manifests,
            stop=failure,
            seconds=monotonic() - start,
            peak_bytes=mx.get_peak_memory(),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "measure"))
    run(p.parse_args().action == "freeze")
