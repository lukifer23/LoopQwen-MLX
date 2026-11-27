"""Freeze both sides of evidence-changing human examples and withhold complete source groups."""

from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from loopqwen.relational_tools import tools
from loopqwen.research.conditional_features import changed_token_spans
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.table_worlds import FILENAME, SOURCE, canonical, mount, state
from loopqwen.tool_session import compact
from scripts.dialogue_state_material import DEST as DIALOGUES
from scripts.grounded_service_turns import OUT as SERVICES
from scripts.multiworld_program_material import OUT as TABLES
from scripts.program_language_material import DEST as OLD
from scripts.program_language_material import answer, call, norm, observed, turn, user
from scripts.qualitative_training_pairs import OUT as REASON
from scripts.table_policy_material import persist

OUT = Path("research/sources/contrastive-worlds")
SEED = 792571
GROUP_LIMIT = 24


def ordered(rows, key):
    return sorted(rows, key=lambda r: fingerprint([SEED, key, r]))


def canonical_pair(group, source, contexts, targets):
    """Each human answer is positive under its own source and negative under the other."""
    assert len(contexts) == len(targets) == 2 and norm(targets[0]) != norm(targets[1])
    for i, (messages, definitions) in enumerate(contexts):
        yield turn(
            group + "/" + source + "/" + str(i),
            group,
            messages,
            definitions,
            answer(targets[i]),
            pair=source,
            member=i,
            rejected=[
                {"edit": "other source answer", "target": answer(targets[1 - i])}
            ],
        )


def table_rows(train, folder):
    for w in train:
        original, *alternates = w["versions"]
        assert original["world"] == 0
        alternatives = [
            v
            for v in alternates
            if canonical(v["answer"]) != canonical(original["answer"])
        ]
        other = min(
            alternatives, key=lambda v: fingerprint([SEED, w["id"], v["world"]])
        )
        selected = [original, other]
        paths = [mount(folder / w["id"] / str(v["world"]), v["rows"]) for v in selected]
        desc = {**w, "versions": selected}
        messages, defs = state(desc, folder)
        query = tools([paths[0]])[1]
        action = call(query, database_name=FILENAME, statement=w["statement"])
        yield turn(
            "operator/" + w["id"], "operator", messages, defs, action, pair=w["id"]
        )
        yield turn(
            "inspect/" + w["id"],
            "inspect",
            messages[:1],
            defs,
            messages[1],
            pair=w["id"],
        )
        targets = [v["answer"] for v in selected]
        for mode, statement in (
            ("resolved", w["statement"]),
            ("full_rows", "SELECT * FROM records ORDER BY rowid"),
        ):
            histories = []
            for path, v in zip(paths, selected, strict=True):
                provider = tools([path])[1]
                try:
                    obs = provider.callback(database_name=FILENAME, statement=statement)
                except ValueError:
                    if mode == "full_rows":
                        break
                    raise
                if mode == "full_rows" and obs["truncated"]:
                    break
                assert obs["referenced_tables"] == ["records"] and not obs["truncated"]
                if mode == "resolved":
                    assert obs["rows"] == [[v["observed_scalar"]]]
                else:
                    assert [obs["columns"], *obs["rows"]] == v["rows"]
                act = call(provider, database_name=FILENAME, statement=statement)
                histories.append(([*messages, act, observed(obs)], defs))
            if len(histories) == 2:
                yield from canonical_pair(mode, w["id"], histories, targets)


def reasoning_rows(train):
    for w in train:
        # The same independently hashed answer order in both questions prevents the position
        # of the first listed choice from revealing which version is the authored flip.
        choices = ordered(w["members"][0]["choices"], w["id"])
        contexts = []
        targets = []
        for m in w["members"]:
            prompt = (
                m["background"]
                + "\n\n"
                + m["question"]
                + "\n"
                + " or ".join(choices)
                + "?"
            )
            contexts.append(([user(prompt)], []))
            targets.append(m["correct"])
        yield from canonical_pair("qualitative", w["id"], contexts, targets)


def service_rows(rows, pairs):
    for pair in pairs:
        contexts = []
        targets = []
        for k in pair["ids"]:
            r = rows[k]
            yield turn(
                "service_action/" + k,
                "service_action",
                r["messages"],
                r["tools"],
                r["target"],
                pair=k,
            )
            messages = [
                *r["messages"],
                r["target"],
                {"role": "tool", "content": compact(r["observation"])},
            ]
            contexts.append((messages, r["tools"]))
            targets.append(r["reply"])
        yield from canonical_pair(
            "service_reply", fingerprint(pair["ids"]), contexts, targets
        )


def reserve(pool, key, blocked=()):
    groups = defaultdict(list)
    for x in pool:
        groups[x[key]].append(x)
    chosen = set(ordered(set(groups) - set(blocked), "reserve/" + key)[:GROUP_LIMIT])
    assert len(chosen) == GROUP_LIMIT
    held = [
        min(groups[k], key=lambda r: fingerprint([SEED, "member", r["id"]]))
        for k in sorted(chosen)
    ]
    return [r for r in pool if r[key] not in chosen], held


def store(name, obj):
    path = OUT / (name + ".json.gz")
    if path.exists():
        assert unpack(path) == obj
    else:
        packed(path, obj)
    return path


def build():
    OUT.mkdir(exist_ok=True, parents=True)
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    tables = unpack(TABLES / "training.json.gz")
    old_tables = read(SOURCE)["cohorts"]["train"]
    table_train, table_hold = reserve(
        tables, "context", [w["context"] for w in old_tables]
    )
    no_questions = {norm(w["question"]) for w in table_hold}
    no_bodies = {fingerprint(v["rows"]) for w in table_hold for v in w["versions"]}
    assert not no_questions.intersection(norm(w["question"]) for w in old_tables)
    assert not no_bodies.intersection(
        fingerprint(v["rows"]) for w in old_tables for v in w["versions"]
    )
    table_train = [
        w
        for w in table_train
        if norm(w["question"]) not in no_questions
        and not any(fingerprint(v["rows"]) in no_bodies for v in w["versions"])
    ]
    qualitative = unpack(REASON / "training.json.gz")
    reason_train, reason_hold = reserve(qualitative, "source_background")
    no_q = {norm(m["question"]) for w in reason_hold for m in w["members"]}
    reason_train = [
        w
        for w in reason_train
        if all(norm(m["question"]) not in no_q for m in w["members"])
    ]
    service = {r["id"]: r for r in unpack(SERVICES / "training.json.gz")}
    pairs = read(SERVICES / "paired-replies.json")["pairs"]
    prior = {r["dialogue"] for r in read(DIALOGUES)["examples"] if "dialogue" in r}
    untouched = [
        p for p in pairs if all(service[k]["dialogue"] not in prior for k in p["ids"])
    ]
    service_hold = sorted(
        untouched, key=lambda p: fingerprint([SEED, "service hold", p["ids"]])
    )[:GROUP_LIMIT]
    no_dialogues = {service[k]["dialogue"] for p in service_hold for k in p["ids"]}
    no_replies = {norm(service[k]["reply"]) for p in service_hold for k in p["ids"]}
    long_replies = [v for v in no_replies if len(v.split()) >= 8]

    def separate_reply(k):
        text = " " + norm(service[k]["reply"]) + " "
        return not any(" " + v + " " in text for v in long_replies)

    candidates = [
        p
        for p in pairs
        if all(
            service[k]["dialogue"] not in no_dialogues
            and norm(service[k]["reply"]) not in no_replies
            and separate_reply(k)
            for k in p["ids"]
        )
    ]
    strat = defaultdict(list)
    for p in sorted(
        candidates, key=lambda p: fingerprint([SEED, "service enrollment", p["ids"]])
    ):
        strat[p["service"], p["method"]].append(p)
    # Round-robin budget gives scarce services their natural examples before expanding common ones.
    service_train = [
        vals[i]
        for i in range(max(map(len, strat.values())))
        for _, vals in sorted(strat.items())
        if i < len(vals)
    ][:900]
    hold = store(
        "withheld",
        {
            "tables": table_hold,
            "qualitative": reason_hold,
            "services": [[service[k] for k in p["ids"]] for p in service_hold],
        },
    )
    with TemporaryDirectory() as tmp:
        generated = list(table_rows(table_train, Path(tmp)))
    generated.extend(reasoning_rows(reason_train))
    generated.extend(service_rows(service, service_train))
    units = defaultdict(list)
    for row in generated:
        units[row["group"], row["pair"]].append(row)
    totals = defaultdict(Counter)
    drops = Counter()
    accepted = []
    for (group, _), members in units.items():
        stats = Counter()
        fits = True
        for m in members:
            p, y = token_pair(tok, m)
            length = len(p) + len(y)
            stats["turns"] += 1
            stats["response_positions"] += len(y)
            stats["opening_positions"] += len(p)
            for bad in m.get("rejected", []):
                prefix, z = token_pair(tok, {**m, "target": bad["target"]})
                assert prefix == p
                x, zpos = changed_token_spans(y, z)
                length = max(length, len(p) + len(z))
                stats["rejected_positions"] += len(zpos)
                stats["preferred_positions"] += len(x)
                stats["preferences"] += 1
            fits &= length <= 2048 and len(y) <= 384
        if not fits:
            drops[group] += len(members)
            continue
        totals[group].update(stats)
        accepted.extend(members)
    assert len(accepted) == len({r["id"] for r in accepted})
    for group in ("resolved", "full_rows", "qualitative", "service_reply"):
        assert (
            totals[group]["preferences"] == totals[group]["turns"]
            and totals[group]["turns"] % 2 == 0
        )
    training = store("training", accepted)
    manifest = {
        "seed": SEED,
        "withheld": digest(hold),
        "training": digest(training),
        "limits": {"complete_with_target": 2048, "supervised_target": 384},
        "table_questions_enrolled": len(table_train),
        "qualitative_pairs_enrolled": len(reason_train),
        "service_pairs_enrolled": len(service_train),
        "withheld_groups_per_source": GROUP_LIMIT,
        "withheld_service_dialogues": len(no_dialogues),
        "service_methods": dict(Counter(p["method"] for p in service_train)),
        "complete_turn_length_exclusions": dict(drops),
        "groups": dict(totals),
        "sources": {
            str(path): digest(path)
            for path in (
                TABLES / "training.json.gz",
                SERVICES / "training.json.gz",
                SERVICES / "paired-replies.json",
                REASON / "training.json.gz",
                OLD / "training.json.gz",
                SOURCE,
                DIALOGUES,
            )
        },
    }
    persist(OUT / "qualification.json", manifest)
    print(len(accepted), dict(totals), flush=True)


if __name__ == "__main__":
    build()
