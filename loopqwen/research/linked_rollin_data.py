"""Train on source-backed decisions at the actual histories visited by a recurrent decoder."""

from .. import qwen_tools
from ..csv_tools import tools as csv_tools
from ..evaluation.extractive import score as qa
from ..evaluation.observed_chain import progress
from ..io import read
from ..tool_session import compact
from .native_function_data import token_pair
from .native_function_study import executions
from .valid_return import score as prior_score


def providers(w, folder):
    if "files" in w:
        return csv_tools([folder / path for path in w["files"]], keyed=True)
    from scripts.valid_return_material import tool_set

    return tool_set(w, folder)


def matched(w, index, e):
    expected = w["events"][index]
    if e == expected:
        return True
    # A different unique key in the correct physical row can surface the same link.
    args = e["call"].get("arguments", {})
    if (
        e["call"].get("name") != "csv_lookup"
        or args.get("document") != expected["call"]["arguments"]["document"]
    ):
        return False
    if index == len(w["events"]) - 1:
        return (
            e["observation"] == expected["observation"]
            and args.get("value_column") == "Article"
        )
    return w["page"] in str(e["observation"].get("value", "")).split(" | ")


def reached(w, event_turns):
    return progress(event_turns, len(w["events"]), lambda pos, e: matched(w, pos, e))


def score(tok, w, i, tr):
    if "files" not in w:
        return prior_score(tok, w, i, tr)
    assert i == 0
    done = tr["status"] == "answered"
    gold = w["gold"]
    answer = (tr["answer"] or "").strip()
    exact, f1 = qa(answer, [gold])
    observed = reached(w, [executions({"turns": [t]}) for t in tr["turns"]])
    grounded = observed == len(w["events"])
    return {
        "completed": done,
        "correct_answer": bool(done and exact),
        "f1": f1 if done else 0,
        "milestones": observed,
        "observed": grounded,
        "success": bool(done and exact and grounded),
    }


def expert(w, position):
    if position < len(w["events"]):
        return qwen_tools.assistant_message([w["events"][position]["call"]])
    return {
        "role": "assistant",
        "content": w["gold"],
        "reasoning_content": f"The table links {w['cell']} to {w['page']}. The article says: {w['quote']}",
    }


def canonical(w, defs):
    hist = [{"role": "user", "content": w["questions"][0]}]
    for j in range(len(w["events"]) + 1):
        target = expert(w, j)
        yield {
            "id": f"{w['id']}-expert-{j}",
            "kind": "action" if j < len(w["events"]) else "answer",
            "messages": list(hist),
            "tools": defs,
            "target": target,
        }
        if j < len(w["events"]):
            hist.extend(
                (
                    target,
                    {"role": "tool", "content": compact(w["events"][j]["observation"])},
                )
            )


def corrections(w, tr, tok, defs, phase):
    hist = [{"role": "user", "content": w["questions"][0]}]
    observed = []
    close = tok.convert_tokens_to_ids("</think>")
    for j, t in enumerate(tr["turns"]):
        pos = reached(w, observed)
        needed = expert(w, pos)
        prefix = tok.encode(
            qwen_tools.render(tok, hist, defs), add_special_tokens=False
        )
        assert t["prompt_ids"] == prefix
        acts = executions({"turns": [t]})
        solved = (
            any(matched(w, pos, e) for e in acts)
            if pos < len(w["events"])
            else bool(
                tr["status"] == "answered"
                and j == len(tr["turns"]) - 1
                and qa(tr["answer"] or "", [w["gold"]])[0]
            )
        )
        actual = t["token_ids"]
        if not solved:
            yield {
                "id": f"{w['id']}-round-{phase}-turn-{j}",
                "kind": "correction",
                "messages": list(hist),
                "tools": defs,
                "target": needed,
                "learner": actual,
                "completed_thinking": close in actual,
                "visited_milestones": pos,
                "actual_valid_return": any("value" in e["observation"] for e in acts),
            }
        observed.append(acts)
        if acts:
            raw = tok.decode(actual[:-1])
            reasoning = raw.partition("</think>")[0].strip()
            msg = qwen_tools.assistant_message(
                [e["call"] for e in acts],
                t["visible"].partition("<tool_call>")[0].strip(),
                reasoning,
            )
            hist.extend(
                [
                    msg,
                    *[
                        {"role": "tool", "content": compact(e["observation"])}
                        for e in acts
                    ],
                ]
            )


def encode(tok, view):
    prefix, ref = token_pair(tok, view)
    close = tok.convert_tokens_to_ids("</think>")
    at = ref.index(close) + 1
    active = list(range(at, len(ref)))
    wrong = None
    if view["kind"] == "correction" and view["completed_thinking"]:
        learner = view["learner"]
        where = learner.index(close) + 1
        # The student's complete reasoning is observed context; no gradient targets
        # are assigned to those generated tokens. Correct the subsequent action.
        prefix = prefix + learner[:where]
        ref = ref[at:]
        at = 0
        actual = learner[where:]
        wrong = next(
            ((j, b) for j, (a, b) in enumerate(zip(ref, actual)) if a != b), None
        )
        active = list(range(len(ref)))
    assert active and ref[-1] == tok.convert_tokens_to_ids("<|im_end|>")
    return prefix, ref, active, wrong


def protected_views():
    from .reasoned_evidence_data import choose
    from .valid_return_data import turns

    v = turns(read("research/experiments/valid-return-01.json"), None)
    for k in ("field", "operation", "verdict", "repair", "rehearsal"):
        subset = [w for w in v if w["kind"] == k and w["choice"] == "preferred"]
        for row in choose(subset, 670027, lambda q: q["id"])[:64]:
            yield {**row, "id": "retained-" + row["id"], "kind": "retained"}


def selected_cases(training, phase):
    from .reasoned_evidence_data import choose

    clusters = choose(sorted({w["cluster"] for w in training["rows"]}), 670028, str)
    ids = {
        cluster: ("linked_file", "linked_inline")[i % 2]
        for i, cluster in enumerate(clusters[64 * phase : 64 * (phase + 1)])
    }
    selected = [w for w in training["rows"] if ids.get(w["cluster"]) == w["family"]]
    assert len(selected) == 64
    return selected
