"""Reconstruct the failed Unicode pretokenization and the corrected input identities."""

from pathlib import Path

from mlx_lm.tokenizer_utils import load

from ..export_runtime import load_export_tokenizer
from ..identity import fingerprint
from ..io import read
from ..tool_session import initial_messages, prompt
from .folded_system import tools_for


def audit(spec, field, panel):
    directory = Path(spec["checkpoint"])
    eos = [read(directory / "config.json")["text_config"]["eos_token_id"]]
    old = load(directory, eos_token_ids=eos)
    new = load_export_tokenizer(directory, eos_token_ids=eos)
    changed = []
    for i, row in enumerate(panel, 1):
        tools = tools_for(row, field, True)
        q = row.get("user_request", row["question"])
        messages = (
            initial_messages(q, tools) if tools else [{"role": "user", "content": q}]
        )
        former = old.encode(prompt(old, messages), add_special_tokens=False)
        fixed = new.encode(prompt(new, messages), add_special_tokens=False)
        assert fixed == row["initial_ids"]
        if former != fixed:
            changed.append(
                {
                    "position": i,
                    "id": row["id"],
                    "original_ids": former,
                    "pinned_ids": fixed,
                    "original_sha256": fingerprint(former),
                    "pinned_sha256": fingerprint(fixed),
                }
            )
    return {
        "original_class": type(old._tokenizer).__name__,
        "fixed_class": type(new._tokenizer).__name__,
        "checked": len(panel),
        "mismatches": changed,
    }


def stopped(panel, records, report):
    [bad] = report["mismatches"]
    assert len(records) < bad["position"] <= len(records) + 12
    assert [r["id"] for r in records] == [w["id"] for w in panel[: len(records)]]
    return {
        "passed": False,
        "status": "unequal_initial_ids",
        "first_invalid_position": bad["position"],
        "stored_valid_candidate": len(records),
        "stored_control": 0,
        "enrolled": len(panel),
    }
