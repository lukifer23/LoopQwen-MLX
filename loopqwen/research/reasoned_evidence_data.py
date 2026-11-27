"""Prepare native conversation views from source text and actual CSV callbacks."""

import csv
import io

from .. import qwen_tools
from ..csv_tools import tools
from ..identity import fingerprint
from ..tool_session import compact


def table(headers, rows):
    out = io.StringIO(newline="")
    w = csv.writer(out)
    w.writerow(headers)
    w.writerows(rows)
    return out.getvalue()


def choose(rows, seed, key):
    return sorted(rows, key=lambda r: fingerprint([seed, key(r)]))


def bind_file(tmp, filename, raw, key, value, content):
    path = tmp / filename
    path.write_bytes(raw.encode())
    provider = tools([path], keyed=True)
    headers = next(csv.reader(io.StringIO(raw)))
    args = {
        "document": filename,
        "key_column": headers[0],
        "key": key,
        "value_column": headers[1],
    }
    call = {"name": provider[0].name, "arguments": args}
    obs = provider[0].callback(**args)
    assert obs == {"value": value}
    defs = qwen_tools.declarations(provider)
    history = [{"role": "user", "content": content}]
    action = qwen_tools.assistant_message([call])
    return (
        defs,
        call,
        obs,
        [*history, action, {"role": "tool", "content": compact(obs)}],
    )


def text(source, claim):
    return (
        source
        + "\n\nClaim: "
        + claim
        + "\nDoes the evidence establish the claim, rule it out, or leave it uncertain? Check your reasoning and give one final verdict: true, false, or uncertain."
    )
