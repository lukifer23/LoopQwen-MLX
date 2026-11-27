"""Run the installed command on sealed real cases and count tokenizer passes over its actual inputs."""

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read, save
from loopqwen.worker import child_lease
from scripts.verify_runtime_wheel import CODE, ROOT, trace

SPEC = Path("research/diagnostics/tool-input-parity.json")
OUT = Path("docs/measurements/tool-input-parity.json")
PROBE = """
from loopqwen import tool_session
import json
receipt=Path(sys.argv.pop(1))
encodings=[]
class Watched:
    def __init__(self, original):
        self.original=original
    def __getattr__(self, k):
        return getattr(self.original,k)
    def encode(self, text, *args, **kwargs):
        ids=self.original.encode(text,*args,**kwargs)
        encodings.append(len(ids))
        return ids
run=tool_session.run
def watch(model,tok,*args,**kwargs):
    return run(model,Watched(tok),*args,**kwargs)
tool_session.run=watch
def offline(name,args):
    if name in ('socket.connect','socket.getaddrinfo'):
        raise RuntimeError('External access from offline replay')
sys.addaudithook(offline)
"""


def run():
    spec = read(SPEC)
    panel = read("research/data/valid-return-panel.json")
    corpus = ROOT / "research/results/valid-return-01"
    archived = {k: unpack(corpus / (k + ".json.gz")) for k in ("candidate", "control")}
    src = ROOT / spec["wheel"]
    assert digest(src) == spec["sha256"]
    results = []
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        site = work / "site"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--no-deps",
                "--offline",
                "--target",
                str(site),
                str(src),
            ],
            check=True,
        )
        for chosen in spec["requests"]:
            i = next(i for i, w in enumerate(panel["rows"]) if w["id"] == chosen)
            w = panel["rows"][i]
            recorded = work / "calls.json"
            program = CODE.replace(
                "main()\n",
                PROBE + "\nmain()\nreceipt.write_text(json.dumps(encodings))\n",
            )
            args = [
                sys.executable,
                "-I",
                "-c",
                program,
                str(site),
                str(recorded),
                "ask",
                "--checkpoint",
                str(ROOT / "exports/tool-language-packed-05"),
                "--session-policy",
                str(corpus),
                "--tool-format",
                "qwen",
                "--trace",
                "--compare-base",
                "--max-turns",
                "3",
                "--max-tokens",
                "512",
                "--prompt",
                w["questions"][0],
            ]
            if w.get("document"):
                (work / w["document"]).write_text(panel["documents"][w["document"]])
                args.extend(["--keyed-csv", w["document"]])
            lease = child_lease()
            lease["env"].update(
                HF_HOME=str(work / "empty"),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
            )
            native = json.loads(
                subprocess.check_output(
                    args, cwd=work, **lease, timeout=spec["per_pair_seconds"]
                )
            )
            lengths = []
            for side, arm in (("recurrent", "candidate"), ("base", "control")):
                (r,) = native["runs"][side]
                target = archived[arm][i]["requests"][0]["trajectory"]
                assert (
                    trace(r) == trace(target)
                    and r["peak_mlx_bytes"] < spec["peak_gib"] * 2**30
                )
                turns = r["turns"]
                lengths.extend(len(t["prompt_ids"]) for t in turns)
                results.append(
                    {
                        "id": chosen,
                        "arm": side,
                        "turns": len(turns),
                        "tokens": r["output_tokens"],
                        "status": r["status"],
                    }
                )
            assert read(recorded) == lengths
    receipt = {
        "source": digest(SPEC),
        "runs": results,
        "exact_original_traces": True,
        "native_tokenizer_calls": sum(r["turns"] for r in results),
    }
    if OUT.exists():
        assert read(OUT) == receipt
    else:
        save(OUT, receipt)
    print(receipt)


if __name__ == "__main__":
    run()
