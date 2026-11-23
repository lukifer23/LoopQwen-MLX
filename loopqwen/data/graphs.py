"""Small scheduling DAGs with relabel-invariant task identities."""

import hashlib
import json
import string
from graphlib import TopologicalSorter


def solve(durations, parents, target):
    ready, depth, path = {}, {}, {}
    for v in TopologicalSorter(parents).static_order():
        start = max((ready[u] for u in parents[v]), default=0)
        ready[v] = start + durations[v]
        depth[v] = 1 + max((depth[u] for u in parents[v]), default=0)
        path[v] = max(
            (path[u] for u in parents[v]),
            key=lambda p: sum(durations[j] for j in p),
            default=(),
        ) + (v,)

    def paths(v):
        if not parents[v]:
            yield durations[v]
        for u in parents[v]:
            for total in paths(u):
                yield total + durations[v]

    if any(max(paths(v)) != ready[v] for v in ready):
        raise ArithmeticError("Path enumeration disagrees with topological solver")
    return ready, depth[target], len(path[target])


def graph(rng, level, shift="id"):
    # Two parallel chains meet at a join. A separate sink shares a root.
    parents = {0: [], 1: [], 2: [0], 3: [1], 4: [2, 3], 5: [rng.randrange(2)]}
    target = 4
    for i in range(level):
        target = len(parents)
        parents[target] = [4] if i == 0 else [target - 1]
    if shift == "topology":
        v = len(parents)
        parents[v] = []
        parents[v + 1] = [v, target]
        target = v + 1
    scale = 100 if shift == "operand" else 1
    names = rng.sample(string.ascii_uppercase, len(parents))
    duration = dict(
        zip(names, rng.sample(range(2 * scale, 2 * scale + 32), len(names)))
    )
    preds = {names[v]: tuple(names[u] for u in pp) for v, pp in parents.items()}
    target = names[target]
    ready, depth, critical = solve(duration, preds, target)
    # Durations are all distinct, hence give a canonical labeling independent of display order.
    key = [
        duration[target],
        sorted(
            (duration[v], sorted(duration[u] for u in pp)) for v, pp in preds.items()
        ),
    ]
    nodes = list(preds)
    rng.shuffle(nodes)
    spec = "; ".join(
        f"{v}: {duration[v]} after {','.join(preds[v]) or 'none'}" for v in nodes
    )
    question = f"Jobs run in parallel as soon as all prerequisites finish. Job: duration after prerequisites. {spec}. When does job {target} finish?"
    lines = [
        f"job {v} finishes at {ready[v] - duration[v]}+{duration[v]} = {ready[v]}"
        for v in ready
    ]
    outdegree = [sum(v in pp for pp in preds.values()) for v in preds]
    return dict(
        id=hashlib.sha256(json.dumps(["dag", key]).encode()).hexdigest(),
        question=question,
        answer=str(ready[target]),
        trace="\n".join(lines + [f"Final answer: {ready[target]}"]),
        category="dependencies",
        complexity=level,
        steps=len(lines),
        template="parallel-jobs",
        verification_method="topological DP and independent exhaustive path enumeration",
        graph=dict(durations=duration, predecessors=preds, target=target),
        difficulty=dict(
            nodes=len(nodes),
            edges=sum(map(len, preds.values())),
            dependency_depth=depth,
            critical_path_nodes=critical,
            critical_path_duration=ready[target],
            max_fanout=max(outdegree),
            joins=sum(len(p) > 1 for p in preds.values()),
            roots=sum(not p for p in preds.values()),
            sinks=outdegree.count(0),
            intermediate_states=len(nodes),
        ),
    )
