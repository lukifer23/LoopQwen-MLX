"""Reconstruct prompts, observations and task outcomes from an archived token stream."""
from collections import Counter
import json
import re
from pathlib import Path

from mlx_lm.utils import load_tokenizer
from jsonschema import validate

from loopqwen.model_loader import local_snapshot
from loopqwen.research.tool_worlds import DOMAINS, tools_for
from loopqwen.tool_session import compact, initial_messages, prompt
from scripts.tool_study import summarize


def verify(root, decision, read, sha, require):
    spec = read(root / "registration.json")
    items = read(Path(spec["material"]))
    tokenizer = load_tokenizer(local_snapshot())
    stop_ids=set(tokenizer.eos_token_ids)
    if spec.get("stop_at_turn_end"):
        stop_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    arms = {}
    for side in ("candidate","control"):
        rows = read(root / (side+".json.gz"))
        require([r["id"] for r in rows] == [w["id"] for w in items], "Tool item identity")
        for row,w in zip(rows,items,strict=True):
            traj = row["trajectory"]
            tools = tools_for(w)
            catalog = {t.name:t for t in tools}
            messages = initial_messages(w["question"],tools)
            calls = []
            answer = None
            status = "turn_limit"
            require(1 <= len(traj["turns"]) <= spec["max_turns"], "Turn allocation")
            for i,t in enumerate(traj["turns"]):
                require(t["prompt_ids"] == tokenizer.encode(prompt(tokenizer,messages),add_special_tokens=False), "Tool transcript mismatch")
                ids = t["token_ids"]
                require(1 <= len(ids) <= spec["max_tokens"] and 0 < t["ttft_s"] <= t["seconds"] <= traj["seconds"], "Turn budget or timing")
                raw = tokenizer.decode(ids[:-1])
                if ids[-1] not in stop_ids:
                    require(len(ids) == spec["max_tokens"], "Incomplete capped output")
                    status = "token_limit"
                elif "</think>" not in raw:
                    status = "unfinished_thinking"
                else:
                    visible = raw.split("</think>",1)[1].strip()
                    require(visible == t["visible"], "Visible generation")
                    if "observation" in t:
                        call = json.loads(visible)
                        require(call == t["call"] and set(call) == {"name","arguments"}, "Call mismatch")
                        tool = catalog[call["name"]]
                        validate(call["arguments"],tool.parameters)
                        arg = next(iter(call["arguments"].values()))
                        # Reconstruct hidden tables directly from material, without callback execution.
                        names = w["names"] if tool.name == DOMAINS[w["domain"]][3] else w["places"]
                        vals = w["places"] if tool.name == DOMAINS[w["domain"]][3] else w["numbers"]
                        observation = {"value":vals[names.index(arg)]} if arg in names else {"error":"KeyError"}
                        require(observation == t["observation"], "Forged observation")
                        messages.extend((dict(role="assistant",content=visible),dict(role="tool",content=compact(observation))))
                        calls.append(call)
                        continue
                    if visible.startswith(("{","```")):
                        from loopqwen.tool_session import decode_action
                        from jsonschema import ValidationError
                        try:
                            obj=decode_action(visible)
                            validate(obj["arguments"],catalog[obj["name"]].parameters)
                        except (ValueError,KeyError,ValidationError):
                            status = "invalid_call"
                        else:
                            raise ValueError("Valid call was never executed")
                    else:
                        answer = visible
                        status = "answered" if answer else "empty_answer"
                require(i+1 == len(traj["turns"]), "Unexpected post-terminal turn")
            require(status == traj["status"] and answer == traj["answer"], "Terminal outcome")
            if w["kind"] == "compare":
                path = Counter(map(compact,calls)) == Counter(map(compact,w["expected"]))
            else:
                path = calls == w["expected"]
            if w["kind"] == "clarify":
                place = DOMAINS[w["domain"]][1]
                correct = bool("?" in (answer or "") and re.search(r"\b"+place+r"\b",answer or "",re.I))
            else:
                correct = (answer or "").strip(" .!\n").casefold() == w["gold"].casefold()
            require(row["score"] == dict(success=path and correct and status=="answered",path=path,correct_answer=correct,completed=status=="answered"),"Outcome mismatch")
        arms[side] = rows
    require(summarize(arms,spec) == decision,"Tool summary mismatch")
    if (root/"fit.json").exists():
        meta=read(root/"fit.json")
        require(meta["weights"] == sha(root/"tool-adapters.safetensors") and meta["training"] == sha(Path(spec["training"])) and meta["updates"]==spec["updates"],"Tool fit lineage")
