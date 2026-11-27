"""Replay every archived failed call and compare its CSV arguments with the enrolled actions."""

import sqlite3
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import read, save
from loopqwen.research.conditional_primary import configure, providers
from loopqwen.research.native_function_study import executions
from scripts.assistance_deliberation import write_csv
from scripts.conditional_tool_recovery import ARMS, ROOT, P


def build():
    spec = read(ROOT / "registration.json")
    material = read(spec["material"])
    items = {w["id"]: w for w in material["rows"]}
    rows = []
    sums = {}
    with TemporaryDirectory() as t:
        folder = Path(t)
        configure(material, folder)
        for arm in ARMS:
            classes = Counter()
            role = Counter()
            exact_role = Counter()
            for r in unpack(ROOT / (arm + ".json.gz")):
                w = items[r["id"]]
                (q,) = r["requests"]
                ex = executions(q["trajectory"])
                if not any("error" in e["observation"] for e in ex):
                    continue
                live = {p.name: p for p in providers(w, folder)}
                for i, e in enumerate(ex, 1):
                    if "error" not in e["observation"]:
                        continue
                    name = e["call"]["name"]
                    args = e["call"]["arguments"]
                    try:
                        live[name].callback(**args)
                    except (ValueError, sqlite3.Error) as err:
                        assert {"error": type(err).__name__} == e["observation"]
                        message = str(err)
                    else:
                        raise AssertionError(
                            "An archived provider failure now succeeds"
                        )
                    classes[name + " / " + message.split(":")[0]] += 1
                    detail = {
                        "id": r["id"],
                        "arm": arm,
                        "family": w["family"],
                        "ordinal": i,
                        "tool": name,
                        "error": e["observation"]["error"],
                        "diagnostic": message,
                        "arguments": args,
                        "reference_csv": None,
                        "differing_arguments": None,
                        "key_differs_only_by_case": None,
                    }
                    if name == "csv_lookup":
                        options = [
                            c["arguments"] for c in w["calls"] if c["name"] == name
                        ]
                        assert len(options) == 1
                        expected = options[0]
                        keys = sorted(k for k in expected if args[k] != expected[k])
                        case_only = (
                            args["key"] != expected["key"]
                            and args["key"].casefold() == expected["key"].casefold()
                        )
                        detail.update(
                            reference_csv=expected,
                            differing_arguments=keys,
                            key_differs_only_by_case=case_only,
                        )
                        exact_role[" + ".join(keys)] += 1
                        for k in keys:
                            role[k] += 1
                        role["key_changed_only_in_case"] += case_only
                        role["calls"] += 1
                    rows.append(detail)
            sums[arm] = {
                "provider_diagnostics": dict(classes),
                "different_csv_fields": dict(role),
                "joint_csv_fields": dict(exact_role),
            }
    assert len(rows) == 127
    write_csv(P / "conditional-primary-provider-errors.csv", rows)
    save(P / "conditional-primary-provider-errors.json", sums)
    return sums


if __name__ == "__main__":
    print(build())
