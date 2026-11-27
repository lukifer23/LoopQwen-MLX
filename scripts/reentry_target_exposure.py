"""Account for actual token-level supervision and repeated target strings in the warm-up."""

from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.integrated_assistance_fit import positions
from loopqwen.research.table_reentry_fit import NAME, decks, inputs
from scripts.assistance_deliberation import write_csv

P = Path("paper")
SPEC = Path("research/experiments") / (NAME + ".json")


def build():
    s = read(SPEC)
    tok = tokenizer(s)
    examples = {r["id"]: r for r in read(s["training"])["examples"]}
    source = inputs(tok, s)
    pools = decks(source, s["fit_seed"])
    closer = tok.convert_tokens_to_ids("</think>")
    terminal = tok.convert_tokens_to_ids("<|im_end|>")
    raw = defaultdict(Counter)
    seen = defaultdict(set)
    domains = defaultdict(set)
    targets = defaultdict(Counter)
    rationale = defaultdict(set)
    mass = defaultdict(Counter)
    labels_by_id = {}
    supervised = defaultdict(set)

    @cache
    def whitespace(token):
        return not tok.decode([token]).strip()

    for key, (_, _, labels, _) in source.items():
        at = labels.index(closer)
        assert labels.count(closer) == 1 and labels[-1] == terminal
        labels_by_id[key] = (fingerprint(labels), at)
        domains[examples[key]["group"]].add(key)
    for step in range(1, s["warm_updates"] + 1):
        for name in sorted(pools):
            key = next(pools[name])
            _, prefix, labels, visible = source[key]
            code, at = labels_by_id[key]
            family = examples[key]["group"]
            ix = positions(
                labels, visible, s["warm_targets"], f"{s['fit_seed']}/{step}/{key}"
            )
            total = raw[family]
            total["draws"] += 1
            total["prefix_and_history"] += len(prefix) + len(labels) - 1
            total["positions"] += len(ix)
            seen[family].add(key)
            if any(not whitespace(i) for i in labels[:at]):
                rationale[family].add(tuple(labels[:at]))
            targets[family][code] += 1
            for j in ix:
                token = labels[j]
                if token == closer or token == terminal:
                    category = "boundary"
                elif whitespace(token):
                    category = "whitespace"
                else:
                    category = "thought" if j < at else "visible"
                total[category] += 1
                supervised[family].add((key, j))
                mass[family][category] += 1 / len(ix)
    result = []
    for family, c in sorted(raw.items()):
        n = c["draws"]
        result.append(
            {
                "source_group": family,
                "eligible_turns": len(domains[family]),
                "draws": n,
                "distinct_turns_drawn": len(seen[family]),
                "distinct_target_sequences": len(targets[family]),
                "most_frequent_target_draws": max(targets[family].values()),
                "distinct_nonempty_thoughts": len(rationale[family]),
                "positions": c["positions"],
                "unique_turn_positions": len(supervised[family]),
                "input_history_tokens": c["prefix_and_history"],
                **{
                    kind + "_tokens": c[kind]
                    for kind in ("boundary", "whitespace", "thought", "visible")
                },
                **{
                    kind + "_objective_fraction": mass[family][kind] / n
                    for kind in ("boundary", "whitespace", "thought", "visible")
                },
            }
        )
    path = P / "reentry-target-exposure.csv"
    write_csv(path, result)
    categories = ("boundary", "whitespace", "thought", "visible")
    total_draws = sum(v["draws"] for v in raw.values())
    assert total_draws == len(pools) * s["warm_updates"] and sum(
        sum(v[k] for k in categories) for v in raw.values()
    ) == sum(v["positions"] for v in raw.values())
    report = {
        "training_material": digest(Path(s["training"])),
        "registration": digest(SPEC),
        "rows": digest(path),
        "draws": total_draws,
        "distinct_source_turns": sum(map(len, seen.values())),
        "selected_positions": sum(v["positions"] for v in raw.values()),
        "unique_turn_positions": sum(map(len, supervised.values())),
        "causal_input_tokens": sum(v["prefix_and_history"] for v in raw.values()),
        "objectives": {
            k: {
                "tokens": sum(v[k] for v in raw.values()),
                "objective_fraction": sum(v[k] for v in mass.values()) / total_draws,
            }
            for k in categories
        },
    }
    assert (
        abs(sum(v["objective_fraction"] for v in report["objectives"].values()) - 1)
        < 1e-8
    )
    save("docs/measurements/reentry-target-exposure.json", report)
    print(report)


if __name__ == "__main__":
    build()
