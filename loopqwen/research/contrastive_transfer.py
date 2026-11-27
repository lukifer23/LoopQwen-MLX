"""One physical product comparison on complete held sources, including all four unabridged long inputs."""

import gc
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mlx.core as mx

from scripts.register_contrastive_world import PRIMARY

from ..conditional_reentry import ConditionalAssistant
from ..evaluation.mechanism import packed, unpack
from ..identity import fingerprint
from ..io import digest, read, save
from ..qwen_tools import declarations
from ..relational_tools import tools
from ..tool_compare import DenseAssistant
from .conditional_primary import ARMS, costs
from .contrastive_features import CACHE, FIT, SOURCES
from .contrastive_grading import attested, selected_choice
from .core import Budget
from .cross_view_study import tokenizer
from .file_evidence_environment import capture
from .linked_rollin import exact_sign_flip
from .native_function_study import executions, replay
from .registered import registered
from .table_reentry_fit import root_for
from .table_worlds import FILENAME, mount, present


def database(w, folder):
    if w["family"] not in ("tables", "service"):
        return []
    base = folder / fingerprint(w["id"])
    db = base / FILENAME
    if not db.is_file():
        mount(base, w["rows"])
    providers = tools([db])
    assert declarations(providers) == w["defs"]
    return providers


def score(w, tr):
    completed = tr["status"] == "answered"
    text = tr["answer"] or ""
    f = w["family"]
    if f == "qualitative":
        ans = selected_choice(text, w["choices"])
        return {
            "completed": completed,
            "choice": ans,
            "success": completed and ans == w["gold"],
        }
    if f == "locale":
        (t,) = tr["turns"]
        n = len(w["initial_ids"])
        assert (
            n > 131072
            and t["prompt_ids"] == w["initial_ids"]
            and not t.get("reused_prompt_tokens", 0)
        )
        assert t["cache_state_blocks"] == (n - 1) // 128
        assert fingerprint(t["prompt_ids"]) == w["initial_sha256"]
        return {
            "completed": completed,
            "success": completed and text.strip() == w["gold"],
            "raw_context": n,
        }
    observed = [
        e["observation"]
        for e in executions(tr)
        if e["call"]["name"] == "database_query"
        and e["observation"].get("referenced_tables") == ["records"]
        and not e["observation"].get("truncated", True)
        and e["observation"].get("rows")
    ]
    if f == "tables":
        valid = completed and present(text, w["gold"])
        return {
            "completed": completed,
            "correct_answer": valid,
            "observed": bool(observed),
            "success": bool(observed) and valid,
        }
    assert f == "service"
    support = attested(
        text, w["rows"][1:], [r for obs in observed for r in obs["rows"]]
    )
    return {
        "completed": completed,
        "observed": bool(observed),
        "observed_two_fields": support,
        "success": completed and support,
    }


def keep(w, s, arm, actor, folder, guard):
    providers = database(w, folder)
    mx.synchronize()
    mx.reset_peak_memory()
    start = perf_counter()
    if actor is None:
        if arm == "candidate":
            actor = ConditionalAssistant(
                s["checkpoint"], Path("research/results") / FIT
            )
        else:
            actor = DenseAssistant(True, "qwen", s["checkpoint"])
            assert actor.language.base_source == "certified_release_text"
            actor.native_plain = True
    chat = actor.conversation(providers)
    stamps = {}

    def observe(count):
        if count not in stamps:
            stamps[count] = perf_counter() - start
        guard.check()

    tr = chat.respond(
        w["questions"][0],
        max_tokens=s["max_tokens"],
        max_turns=s["context_turns"] if w["family"] == "locale" else s["max_turns"],
        reuse_prefix=True,
        record_kv_history=True,
        prefill_cache_state=True,
        progress=observe,
    )
    mx.synchronize()
    elapsed = perf_counter() - start
    assert tr["turns"][0]["prompt_ids"] == w["initial_ids"]
    q = {
        "trajectory": tr,
        "wall_s": elapsed,
        "first_s": stamps[len(w["initial_ids"])],
        "peak": mx.get_peak_memory(),
        "score": score(w, tr),
    }
    return actor, {"id": w["id"], "family": w["family"], "requests": [q]}


def paired(values):
    counts = [
        sum(all(v[i] for v in group) for group in values.values()) for i in (0, 1)
    ]
    deltas = [
        int(all(v[0] for v in g)) - int(all(v[1] for v in g)) for g in values.values()
    ]
    return {
        "whole": counts,
        "p": exact_sign_flip(deltas),
        "losses": sum(not x and y for g in values.values() for x, y in g),
        "individual": [sum(v[i] for g in values.values() for v in g) for i in (0, 1)],
    }


def outcome(panel, runs):
    families = defaultdict(lambda: defaultdict(list))
    complete = {a: Counter() for a in ARMS}
    for w, *pair in zip(panel, *[runs[a] for a in ARMS], strict=True):
        assert all(r["id"] == w["id"] for r in pair)
        qq = [r["requests"][0]["score"] for r in pair]
        for a, v in zip(ARMS, qq, strict=True):
            complete[a][w["family"]] += v["completed"]
        families[w["family"]][w["cluster"]].append([q["success"] for q in qq])
    assert all(len(v) == 2 for group in families.values() for v in group.values())
    results = {f: paired(v) for f, v in families.items()}
    c = {a: costs([r["requests"][0] for r in runs[a]]) for a in ARMS}
    loop, base = (c[a] for a in ARMS)
    gates = {
        f: v["whole"][0] >= 6
        and v["whole"][0] - v["whole"][1] >= 5
        and v["p"] <= 0.05
        and not v["losses"]
        for f, v in results.items()
        if f in ("tables", "qualitative")
    }
    locale = results["locale"]
    gates.update(
        locale=locale["whole"][0] == 2
        and locale["individual"][0] > locale["individual"][1],
        completion=all(
            complete["candidate"][f] >= complete["control"][f] for f in families
        ),
    )
    gates.update({k: loop[k] < base[k] for k in ("wall_s", "first_s", "peak")})
    gates.update({k: loop[k] > base[k] for k in ("opening_tps", "decode_tps")})
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "counts": results,
        "completed": complete,
        "costs": c,
        "service_quality": "Independent committed masked judgments with real observed support are required before qualification.",
    }


def verify(root, d):
    s = registered(PRIMARY)
    assert s == read(root / "registration.json")
    parent = Path("research/results") / FIT
    assert read(root / "weights.json") == {
        "readout": digest(parent / "readout.safetensors"),
        "fit_archive": digest(parent / "checksums.json"),
        "backbone": read(parent / "fit.json")["backbone"]["sha256"],
        "base": s["base_text_sha256"],
    }
    if d.get("status") == "interrupted":
        assert not d["passed"] and d["reason"]
        for a, count in d["durable"].items():
            assert len(unpack(root / (a + "-partial.json.gz"))) == count
        return
    panel = unpack(Path(s["material"]))["rows"]
    streams = {}
    tok = tokenizer(s)
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for a in ARMS:
            streams[a] = unpack(root / (a + ".json.gz"))
            for w, r in zip(panel, streams[a], strict=True):
                (q,) = r["requests"]
                assert q["trajectory"]["turns"][0]["prompt_ids"] == w["initial_ids"]
                replay(
                    tok,
                    w,
                    database(w, folder),
                    [{"role": "user", "content": w["questions"][0]}],
                    q,
                    {
                        **s,
                        "max_turns": s["context_turns"]
                        if w["family"] == "locale"
                        else s["max_turns"],
                    },
                    a,
                )
                assert score(w, q["trajectory"]) == q["score"]
    assert outcome(panel, streams) == d


def run():
    s = registered(PRIMARY)
    parent = Path("research/results") / FIT
    if not read(parent / "decision.json")["passed"]:
        raise ValueError("The contrastive fit did not qualify for held comparison")
    root = root_for(s)
    capture(
        root,
        (
            *SOURCES,
            "loopqwen/research/contrastive_fit.py",
            "loopqwen/research/contrastive_transfer.py",
            "loopqwen/research/contrastive_grading.py",
            "loopqwen/evaluation/generation_eval.py",
            "loopqwen/research/native_function_study.py",
            "scripts/contrastive_transfer_panel.py",
        ),
    )
    physical = {
        "readout": digest(parent / "readout.safetensors"),
        "fit_archive": digest(parent / "checksums.json"),
        "backbone": read(parent / "fit.json")["backbone"]["sha256"],
        "base": s["base_text_sha256"],
    }
    save(root / "weights.json", physical)
    guard = Budget(PRIMARY, s["limits"])
    panel = unpack(Path(s["material"]))["rows"]
    streams = {}
    try:
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for a in ARMS:
                stream = streams[a] = []
                actor = None
                gc.collect()
                mx.clear_cache()
                partial = CACHE / PRIMARY / fingerprint([s, physical]) / a
                partial.mkdir(exist_ok=True, parents=True)
                for i, w in enumerate(panel):
                    path = partial / f"{i:04}.json.gz"
                    if path.is_file():
                        r = unpack(path)
                        assert (
                            r["id"] == w["id"]
                            and r["requests"][0]["trajectory"]["turns"][0]["prompt_ids"]
                            == w["initial_ids"]
                        )
                    else:
                        actor, r = keep(w, s, a, actor, folder, guard)
                        packed(path, r)
                    stream.append(r)
                    if (i + 1) % 8 == 0 or i + 1 == len(panel):
                        print(a, i + 1, flush=True)
                del actor
                packed(root / (a + ".json.gz"), stream)
        d = outcome(panel, streams)
    except (TimeoutError, MemoryError, InterruptedError) as e:
        d = {
            "passed": False,
            "status": "interrupted",
            "reason": str(e),
            "durable": {a: len(v) for a, v in streams.items()},
        }
        for a, rr in streams.items():
            packed(root / (a + "-partial.json.gz"), rr)
    finally:
        guard.close()
    save(root / "decision.json", d)


if __name__ == "__main__":
    run()
