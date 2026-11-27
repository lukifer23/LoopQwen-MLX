"""Compare physical long continuation logits and greedy tokens before enabling resident branches."""

from pathlib import Path
from time import monotonic
from tempfile import TemporaryDirectory
from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.research.saved_document import SavedDocument
from loopqwen.tool_session import run
from scripts.middle_depth_screen import SOURCE, tools
from scripts.verify_runtime_wheel import trace

import mlx.core as mx

from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from loopqwen.looped_model import LoopedModel
from loopqwen.middle_depth import donor
from loopqwen.recurrence import Span
from loopqwen.research.resident_document import ResidentDocument
from scripts.middle_depth_budget import BASE, ROOT, worker
from scripts.middle_depth_documents import FROZEN, worlds
from scripts.middle_depth_retention import saved


def callbacks(model, tok, guard):
    row = next(r for r in read(SOURCE) if r["source"] == "tool")
    first = unpack(ROOT / "screen" / "base-4.json.gz")["trace"]["turns"][0][
        "prompt_ids"
    ]
    traces = []
    with TemporaryDirectory() as tmp:
        doc = SavedDocument(model, first[:256], Path(tmp), "screen-tool-replay", guard)
        for reader in (doc, ResidentDocument(doc, 256, guard)):
            result = run(
                model,
                tok,
                row["prompt"],
                tools(row),
                tool_format="qwen",
                max_tokens=256,
                max_turns=3,
                document_prefix=reader,
                record_kv_history=True,
                progress=guard,
            )
            assert result["turns"][0]["prompt_ids"] == first
            traces.append(result)
    calls = sum(
        int("observation" in t) + len(t.get("observations", []))
        for t in traces[0]["turns"]
    )
    same = trace(traces[0]) == trace(traces[1])
    packed(
        ROOT / "resident-callbacks.json.gz",
        dict(traces=traces, calls=calls, identical=same),
    )
    return same and calls > 0


def profile():
    with worker("long", "full-prefix resident-vs-staged output parity") as guard:
        text, tok = donor(BASE, "base")
        model = LoopedModel(text, Span(12, 4, 1))
        model.eval()
        cfg = read(FROZEN)["book"]
        _, _, w, ids = worlds("book", tok)
        ident = fingerprint(
            ["base-dc7cdfe", model.sites, cfg["common_sha"], "saved-prefix-v1"]
        )
        path = Path("outputs/middle-depth/documents") / ident
        prefix = saved(
            model,
            ids,
            path,
            ident,
            guard,
            [".venv/bin/python", "-m", "scripts.middle_depth_resident"],
        )
        # The prescribed next positions are the real original book continuation.
        # At each position record the complete vocabulary before choosing its top token.
        full = w["original"][1]
        x = full[len(ids) : len(ids) + 128]
        assert len(x) == 128
        rows = []
        refs = []
        for mode, reader in (
            ("staged", prefix),
            ("resident", ResidentDocument(prefix, 16, guard)),
        ):
            mx.clear_cache()
            mx.reset_peak_memory()
            tick = monotonic()
            disk = reader.branch(model, ids + x)
            opened = monotonic()
            logits = disk.decode(mx.array([x]))[:, -1]
            mx.eval(logits)
            first = monotonic()
            errors = []
            tokens = []
            for i in range(16):
                guard()
                if mode == "staged":
                    refs.append(logits)
                else:
                    errors.append(mx.max(mx.abs(refs[i] - logits)).item())
                token = mx.argmax(logits[0]).item()
                tokens.append(token)
                if i < 15:
                    logits = disk.decode(mx.array([[token]]))[:, -1]
                    mx.eval(logits)
            rows.append(
                dict(
                    mode=mode,
                    actual_prefix=len(ids),
                    continuation=len(x),
                    tokens=tokens,
                    maximum_logit_errors=errors,
                    branch_seconds=opened - tick,
                    opening_seconds=first - opened,
                    decode_seconds=monotonic() - first,
                    peak=mx.get_peak_memory(),
                )
            )
            del disk, logits
        passed = (
            rows[0]["tokens"] == rows[1]["tokens"]
            and max(rows[1]["maximum_logit_errors"]) == 0
            and rows[1]["decode_seconds"] < rows[0]["decode_seconds"]
        )
        callback_parity = callbacks(model, tok, guard)
        passed = passed and callback_parity
        save(
            ROOT / "resident-qualification.json",
            dict(passed=passed, rows=rows, callback_parity=callback_parity),
        )
        if not passed:
            raise ValueError(
                "Resident branch did not clear exact output and latency gates"
            )


if __name__ == "__main__":
    profile()
