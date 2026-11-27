"""Check exact staged model outputs on a complete trained prefix and a real callback."""

from pathlib import Path
from time import monotonic

import mlx.core as mx

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.looped_model import LoopedModel
from loopqwen.middle_depth import MiddleAssistant, donor
from loopqwen.recurrence import Span
from loopqwen.research.resident_document import StagedVisitDocument
from scripts.middle_depth_bounded_fit import use_backed_head
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.middle_depth_documents import FROZEN, worlds
from scripts.middle_depth_resident import callbacks
from scripts.middle_depth_retention import recipe, saved

PLAN = Path("research/diagnostics/middle-depth/staged-visit-recovery.md")
PROOF = ROOT / "staged-visit-qualification.json"
POLICY = Path("checkpoints/middle-depth/4619/loop/step-0144")


def implementation():
    return {
        str(path): digest(path)
        for path in (
            PLAN,
            Path("loopqwen/offloaded_prefill.py"),
            Path("loopqwen/research/resident_document.py"),
            Path("scripts/middle_depth_head_storage.py"),
            Path("scripts/middle_depth_staged_verify.py"),
        )
    }


def alternate(model, prefix):
    if use_backed_head(model) is None:
        raise ValueError("Missing exact frozen vocabulary qualification")
    return StagedVisitDocument(prefix)


def measure(guard):
    runner = MiddleAssistant(BASE, POLICY)
    model = runner._tool_model
    _, _, books, ids = worlds("anne", runner.tokenizer)
    cfg = read(FROZEN)["anne"]
    weights = read(POLICY / "registration.json")["weights"]
    ident = fingerprint([weights, model.sites, cfg["common_sha"], "saved-prefix-v1"])
    directory = Path("outputs/middle-depth/documents") / ident
    assert (directory / "prefix.json").exists() and fingerprint(ids) == cfg[
        "common_sha"
    ]
    command = read(recipe(directory))["command"]
    prefix = saved(model, ids, directory, ident, guard, command)
    opening = books["original"][1][: len(ids) + 128]
    common = {
        "input": fingerprint(opening),
        "prefix": ident,
        "physical": len(opening),
        "prefix_tokens": len(ids),
        "weights": weights,
        "implementation": implementation(),
    }
    folder = Path("outputs/middle-depth/staged-verification")
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = folder / "reference.safetensors"
    receipt = ROOT / "staged-visit-reference.json"
    rows = []
    for mode in ("staged", "materialized"):
        if mode == "staged" and receipt.exists():
            prior = read(receipt)
            if prior["identity"] != common or digest(snapshot) != prior["values"]:
                raise ValueError("Staged reference changed")
            rows.append(prior["row"])
            continue
        reader = prefix if mode == "staged" else alternate(model, prefix)
        mx.clear_cache()
        mx.reset_peak_memory()
        tick = monotonic()
        disk = reader.branch(model, opening)
        tokens, values, times = [], {}, []
        for i in range(3):
            guard(force=True)
            begin = monotonic()
            if i == 0:
                scores = reader.complete_opening(disk, opening, guard)[:, -1]
            else:
                scores = disk.decode(mx.array([[tokens[-1]]]))[:, -1]
            mx.eval(scores)
            values[str(i)] = scores
            token = mx.argmax(scores, axis=-1).item()
            tokens.append(token)
            times.append(monotonic() - begin)
        guard(force=True)
        row = {
            "mode": mode,
            "seconds": monotonic() - tick,
            "tokens": tokens,
            "times": times,
            "peak": mx.get_peak_memory(),
            "scratch": prefix.disk.scratch_bytes,
        }
        rows.append(row)
        del disk
        if mode == "staged":
            mx.save_safetensors(str(snapshot), values)
            save(receipt, {"identity": common, "row": row, "values": digest(snapshot)})
        else:
            refs = mx.load(str(snapshot))
            errors = [
                mx.max(mx.abs(refs[k].astype(mx.float32) - v.astype(mx.float32))).item()
                for k, v in values.items()
            ]
        del values, scores
    return common, rows, errors


def qualify():
    with worker("long", "staged-visit-parity") as guard:
        common, rows, errors = measure(guard)
        mx.clear_cache()
        text, tok = donor(BASE, "base")
        model = LoopedModel(text, Span(12, 4, 1))
        model.eval()
        path = ROOT / "staged-visit-callbacks.json.gz"
        parity = callbacks(model, tok, guard, lambda doc: alternate(model, doc), path)
        passed = (
            rows[0]["tokens"] == rows[1]["tokens"]
            and max(errors) == 0
            and rows[1]["peak"] < rows[0]["peak"]
            and parity
        )
        save(
            PROOF,
            {
                "passed": passed,
                "identity": common,
                "rows": rows,
                "errors": errors,
                "callbacks": parity,
                "callback_trace": digest(path),
            },
        )
        if not passed:
            raise ValueError("Staged visit output, memory or callback gate missed")
        print(rows, errors, flush=True)


if __name__ == "__main__":
    qualify()
