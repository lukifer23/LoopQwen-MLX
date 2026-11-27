"""Check original simulator transcripts, then pair incompatible human replies from the same service."""

from collections import Counter, defaultdict

from loopqwen import qwen_tools
from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.dialogue_state_material import ROOT
from scripts.grounded_service_turns import OUT
from scripts.program_language_material import norm


def run():
    path = OUT / "training.json.gz"
    audit = read(OUT / "qualification.json")
    assert digest(path) == audit["training"]
    rows = unpack(path)
    byfile = defaultdict(list)
    for r in rows:
        byfile[r["file"]].append(r)
    schema = {s["service_name"]: s for s in read(ROOT / "train/schema.json")}
    for file, records in sorted(byfile.items()):
        assert digest(ROOT / file) == audit["train_files"][file]
        dialogues = {d["dialogue_id"]: d["turns"] for d in read(ROOT / file)}
        for r in records:
            j = int(r["id"].rsplit("/", 1)[1])
            turns = dialogues[r["dialogue"]]
            assert [(m["role"], m["content"]) for m in r["messages"]] == [
                ("user" if t["speaker"] == "USER" else "assistant", t["utterance"])
                for t in turns[:j]
            ]
            assert all(
                not f.get("service_call") for t in turns[:j] for f in t["frames"]
            )
            (frame,) = [f for f in turns[j]["frames"] if "service_call" in f]
            call = frame["service_call"]
            assert (
                r["reply"] == turns[j]["utterance"]
                and r["observation"] == frame["service_results"]
            )
            assert r["target"] == qwen_tools.assistant_message(
                [{"name": call["method"], "arguments": call["parameters"]}]
            )
            readonly = {
                i["name"]
                for i in schema[r["service"]]["intents"]
                if not i["is_transactional"]
            }
            assert {t["function"]["name"] for t in r["tools"]} == readonly and r[
                "method"
            ] in readonly
            before = " " + " ".join(norm(m["content"]) for m in r["messages"]) + " "
            for support in r["supports"]:
                values = support["fields"]
                actual = r["observation"][support["result_index"]]
                assert len({norm(v) for v in values.values()}) >= 2
                assert all(
                    norm(actual[k]) == norm(v)
                    and " " + norm(v) + " " in " " + norm(r["reply"]) + " "
                    for k, v in values.items()
                )
                assert any(" " + norm(v) + " " not in before for v in values.values())
    available = defaultdict(list)
    for r in rows:
        support = r["supports"][0]["fields"]
        # Each match uses exactly the same method and supported field names on the two records.
        k = (r["service"], r["method"], tuple(sorted(support)))
        values = {norm(val) for val in support.values()}
        observed = {norm(str(val)) for row in r["observation"] for val in row.values()}
        available[k].append((r["id"], values, observed))
    pairs = []
    for (service, method, fields), records in sorted(available.items()):
        records.sort(key=lambda r: fingerprint([44219, r[0]]))
        while len(records) > 1:
            left, lv, lo = records.pop()
            for i, (right, rv, ro) in enumerate(records):
                if len(lv - ro) >= 2 and len(rv - lo) >= 2:
                    pairs.append(
                        {
                            "service": service,
                            "method": method,
                            "fields": list(fields),
                            "ids": [left, right],
                            "absent_from_other_results": [
                                sorted(lv - ro),
                                sorted(rv - lo),
                            ],
                        }
                    )
                    records.pop(i)
                    break
    assert len({i for p in pairs for i in p["ids"]}) == len(pairs) * 2
    save(OUT / "paired-replies.json", {"source_sha256": digest(path), "pairs": pairs})
    result = {
        "verified_dialogues": len(rows),
        "verified_original_calls": len(rows),
        "verified_complete_observations": len(rows),
        "distinct_source_fields_attested": sum(
            len(r["supports"][0]["fields"]) for r in rows
        ),
        "disjoint_incompatible_pairs": len(pairs),
        "paired_by_service": dict(Counter(p["service"] for p in pairs)),
        "paired_by_method": dict(Counter(p["method"] for p in pairs)),
        "pair_file": digest(OUT / "paired-replies.json"),
    }
    save(OUT / "independent-replay.json", result)
    print(result)


if __name__ == "__main__":
    run()
