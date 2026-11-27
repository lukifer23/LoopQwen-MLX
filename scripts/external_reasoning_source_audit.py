"""Measure the completeness and output-budget fit of four independently published reasoning corpora."""

import heapq
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import median

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from scripts.assistance_deliberation import write_csv

REVISION = "fc6cc2103c066455aade5d7fbb346039ae36ca5e"
HASHES = {
    "multi_turn_reasoning_if_think": "3319a3987192cd92eb72a3b1b44b61cb653c9e2f19335baf7dd377c6f574ada4",
    "smolagents_toolcalling_traces_think": "7785c904fdaae971881bde732bd168db6d034ac60752f9396d51554a7267efe4",
    "smoltalk_everyday_convs_reasoning_Qwen3_32B_think": "baa621d326f4a884627be17aae8fab29418897bc4af110760d690fa0cd85c5e1",
    "table_gpt_Qwen3_32B_think": "b5298a70e2ac9486ba8451c2be1a4dfd966817b0da0281b46c3ba14d703dc7ab",
}
CAP = 384
SAMPLE = 256


def normalized(t):
    return " ".join(unicodedata.normalize("NFKC", t).casefold().split())


def build():
    spec = read("research/experiments/table-policy-gradient-01.json")
    tok = tokenizer(spec)
    local = read(spec["training"])["examples"]
    training = {
        normalized(m["content"])
        for r in local
        for m in r["messages"]
        if m["role"] == "user"
    }
    held = {
        normalized(t) for r in read(spec["material"])["rows"] for t in r["questions"]
    }
    studies, rows = [], []
    for subset, sha in HASHES.items():
        path = Path("data/smoltalk2-fc6cc210", subset + ".parquet")
        if not path.is_file():
            path = Path(
                hf_hub_download(
                    "HuggingFaceTB/smoltalk2",
                    f"SFT/{subset}-00000-of-00001.parquet",
                    repo_type="dataset",
                    revision=REVISION,
                )
            )
        assert digest(path) == sha
        data = pq.ParquetFile(path)
        population = data.metadata.num_rows
        chosen = set(
            heapq.nsmallest(
                SAMPLE,
                range(population),
                key=lambda i: fingerprint([813477, subset, i]),
            )
        )
        total = 0
        overlap = Counter()
        for batch in data.iter_batches(batch_size=64, columns=["messages"]):
            for messages in batch.column(0).to_pylist():
                index = total
                total += 1
                if index not in chosen:
                    continue
                prompts = {
                    normalized(m["content"]) for m in messages if m["role"] == "user"
                }
                overlap["training"] += bool(prompts & training)
                overlap["evaluation"] += bool(prompts & held)
                for turn, m in enumerate(messages):
                    if m["role"] != "assistant":
                        continue
                    text = m["content"]
                    thought, closed, visible = text.partition("</think>")
                    formed = bool(
                        closed
                        and thought.lstrip().startswith("<think>")
                        and text.count("</think>") == 1
                        and text.count("<think>") == 1
                    )
                    # Measure complete reasoning-enabled Qwen response strings using their native tokenization.
                    # Historical tools have a different schema. Their source strings remain opaque here.
                    if formed:
                        reasoning = thought.lstrip()[len("<think>") :].strip()
                        visible = visible.strip()
                        response = reasoning + "\n</think>\n\n" + visible + "<|im_end|>"
                        ids = tok.encode(response, add_special_tokens=False)
                        length = len(ids)
                    else:
                        reasoning = visible = ""
                        length = None
                    preceding = messages[:turn]
                    tool_obs = [
                        len(tok.encode(x["content"], add_special_tokens=False))
                        for x in preceding
                        if x["role"] == "tool"
                    ]
                    rows.append(
                        {
                            "subset": subset,
                            "row": index,
                            "turn": turn,
                            "assistant_text_sha256": fingerprint(text),
                            "has_single_thinking_channel": formed,
                            "thought_tokens": len(
                                tok.encode(reasoning, add_special_tokens=False)
                            )
                            if formed
                            else None,
                            "visible_tokens": len(
                                tok.encode(visible, add_special_tokens=False)
                            )
                            if formed
                            else None,
                            "full_output_tokens": length,
                            "visible_output": bool(visible),
                            "complete_in_128": bool(visible and length <= 128),
                            "complete_in_384": bool(visible and length <= CAP),
                            "has_source_tool_call": bool(
                                re.search(
                                    r"<tool_call>.*?</tool_call>", visible, re.DOTALL
                                )
                            ),
                            "previous_tool_observations": len(tool_obs),
                            "largest_prior_observation_tokens": max(
                                tool_obs, default=0
                            ),
                        }
                    )
        assert total == population
        subset_rows = [r for r in rows if r["subset"] == subset]
        lengths = [
            r["full_output_tokens"]
            for r in subset_rows
            if r["full_output_tokens"] is not None
        ]
        studies.append(
            {
                "subset": subset,
                "upstream_sha256": sha,
                "source_rows": population,
                "sampled_documents": len(chosen),
                "assistant_turns": len(subset_rows),
                "valid_thinking_channel": len(lengths),
                "nonempty": sum(r["visible_output"] for r in subset_rows),
                "median_output_tokens": median(lengths) if lengths else None,
                "fits_128": sum(r["complete_in_128"] for r in subset_rows),
                "fits_384": sum(r["complete_in_384"] for r in subset_rows),
                "source_action_markup": sum(
                    r["has_source_tool_call"] for r in subset_rows
                ),
                "turns_after_tool": sum(
                    r["previous_tool_observations"] > 0 for r in subset_rows
                ),
                "prior_observation_over_1536": sum(
                    r["largest_prior_observation_tokens"] > 1536 for r in subset_rows
                ),
                "document_exact_user_overlap_training": overlap["training"],
                "document_exact_user_overlap_evaluation": overlap["evaluation"],
            }
        )
    rows_path = Path("paper/external-reasoning-source-turns.csv")
    write_csv(rows_path, rows)
    doc = {
        "source": "https://huggingface.co/datasets/HuggingFaceTB/smoltalk2",
        "revision": REVISION,
        "new_subsets_license": "Apache-2.0 per publisher",
        "sample_seed": 813477,
        "source_rows": sum(s["source_rows"] for s in studies),
        "sampled_documents": len(HASHES) * SAMPLE,
        "turns": len(rows),
        "source_csv": digest(rows_path),
        "checkpoint_tokenizer": spec["checkpoint"],
        "method": "Deterministic hash of source name and original row index. All assistant turns in each selected document. Assistant content is split only when both source thought markers occur once and in order. The normalized full target starts immediately after the already-open Qwen thinking prefix and includes a natural end marker. Tool texts are kept in their source format. Output length is a source feasibility measure, not a trained accuracy, verified correctness, data split guarantee or physical generation measurement.",
        "subsets": studies,
    }
    save("docs/measurements/external-reasoning-source-audit.json", doc)
    print(json.dumps(studies, indent=2), flush=True)


if __name__ == "__main__":
    build()
