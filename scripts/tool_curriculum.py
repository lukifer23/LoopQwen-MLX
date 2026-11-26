"""Freeze grounded multi-turn reasoning plus independent, licensed tool-call demonstrations."""
from collections import Counter
from hashlib import sha256
import json
import random
import re
from pathlib import Path

from huggingface_hub import hf_hub_download
from jsonschema import Draft202012Validator
from loopqwen.io import save,digest
from loopqwen.research.tool_worlds import make,DOMAINS,tools_for
from loopqwen.tool_session import SYSTEM,compact,initial_messages

HERMES="NousResearch/hermes-function-calling-v1"
REV="dae3e1d28cfbcf4b915c04ea1e072030529b4bda"
TOPICS=re.compile(r"weather|forecast|temperature|distance|transit|train_schedule|bus_|flight|restaurant|book|recipe|ingredient|hotel|currency|exchange_rate|unit_conversion|timezone|time_zone|library|museum|event|movie|astronomy|planet|rainfall|soil|plant|garden|flower|playlist|album|music|song|language|translat|calorie|calendar|date_|time_difference")
READ_ONLY=re.compile(r"^(get|find|search|list|convert|calculate|lookup|estimate|forecast|translate|count|check|recommend|compare|retrieve|read|filter|sort|measure|summarize|compute|track|query|analyze|suggest|generate)_")


def public_examples():
    src=Path(hf_hub_download(HERMES,"glaive-function-calling-5k.json",repo_type="dataset",revision=REV))
    corpus=sorted(json.loads(src.read_bytes()),key=lambda r:sha256(r["id"].encode()).digest())
    counts=Counter()
    examples=[]
    for r in corpus:
        turns=r["conversations"]
        if len(turns)<5 or [t["from"] for t in turns[:5]]!=["system","human","gpt","tool","gpt"]:
            continue
        try:
            functions=[d["function"] for d in json.loads(r["tools"])]
            call=json.loads(re.fullmatch(r"<tool_call>\s*(.*?)\s*</tool_call>",turns[2]["value"].strip(),re.S)[1])
            name=call["name"]
            obs=json.loads(re.fullmatch(r"<tool_response>\s*(.*?)\s*</tool_response>",turns[3]["value"].strip(),re.S)[1])
            if not isinstance(obs,dict) or "content" not in obs:
                continue
            definitions={f["name"]:f for f in functions}
            if not (READ_ONLY.search(name) and TOPICS.search(name) and counts[name]<5):
                continue
            if not all(READ_ONLY.search(f["name"]) and TOPICS.search(f["name"]) for f in functions):
                continue
            if not Draft202012Validator(definitions[name]["parameters"]).is_valid(call["arguments"]):
                continue
            if len(compact(functions))>2250 or len(compact(obs))>600 or len(turns[4]["value"])>330:
                continue
        except (ValueError,KeyError,TypeError):
            continue
        call={"name":name,"arguments":call["arguments"]}
        msgs=[dict(role="system",content=SYSTEM+compact(functions)),dict(role="user",content=turns[1]["value"])]
        examples.append(dict(id=r["id"]+"-call",messages=msgs,visible=compact(call),thinking=""))
        msgs=msgs+[dict(role="assistant",content=compact(call)),dict(role="tool",content=compact(obs["content"]))]
        examples.append(dict(id=r["id"]+"-reply",messages=msgs,visible=turns[4]["value"],thinking=""))
        counts[name]+=1
        if sum(counts.values())==120:
            break
    return examples,dict(repository=HERMES,revision=REV,file=src.name,sha256=digest(src),calls=dict(counts))


def grounded_examples():
    rows=make(50528,["depot","museum","campus","library","garden","fleet"],64,"train","grounded")
    rng=random.Random(3147)
    examples=[]
    for row in rows:
        source,place,metric,link,measure=DOMAINS[row["domain"]]
        a,b=row["places"][:2];x,y=row["numbers"][:2]; s=row["names"][0]
        prompts={
            "lookup":[f"Report the {metric} at {a}, in digits only.",f"I need one figure: the {metric} for {place} {a}."],
            "chain":[f"For {s}, use its {place} assignment to discover the corresponding {metric}. Number only.",f"Trace {source} {s} to its {place}. Give that {place}'s {metric} as a number."],
            "compare":[f"Check the {metric} of both {place}s {a} and {b}. Give me the name scoring more.",f"Look up {a} and {b}. Which has more {metric}? Output one name."],
            "direct":[f"I measured {x} for {a}, and {y} for {b}. Name the larger.",f"Given {a} ({x}) and {b} ({y}), which name corresponds to more?"],
            "clarify":[f"Could you get the {metric}? I still need to give you the {place} name.",f"I want the {metric} for a {place}. Which detail do you need from me?"],
        }
        if rng.random()<.8:
            row["question"]=rng.choice(prompts[row["kind"]])
        old=row["args"]
        row["args"]=rng.choice([old,["identifier","location"],["name","name"],["item","site"],["code","code"]])
        for c in row["expected"]:
            orig=next(iter(c["arguments"].values()))
            c["arguments"]={row["args"][int(c["name"]==measure)]:orig}
        tools=tools_for(row)
        messages=initial_messages(row["question"],tools)
        catalog={t.name:t for t in tools}
        for i,c in enumerate(row["expected"]):
            thought=""
            if row["kind"]=="compare":
                thought=f"I need the value for {row['places'][i]}."
            if row["kind"]=="chain" and i==1:
                thought=f"The assigned {place} is {a}. I need its {metric}."
            examples.append(dict(id=row["id"]+f"-call-{i}",messages=messages,thinking=thought,visible=compact(c)))
            observation=catalog[c["name"]].callback(**c["arguments"])
            messages=messages+[dict(role="assistant",content=compact(c)),dict(role="tool",content=compact(observation))]
        thought=""
        if row["kind"] in ("direct","compare"):
            sign=">" if x>y else "<"
            thought=f"{a} has {x}. {b} has {y}. {x} {sign} {y}."
        examples.append(dict(id=row["id"]+"-reply",messages=messages,thinking=thought,visible=row["gold"]))
    return examples


if __name__=="__main__":
    public,provenance=public_examples()
    rows=grounded_examples()+public
    save("research/data/tool-dialogue-grounded-training.json",dict(public_source=provenance,examples=rows))
    print(len(rows),"training turns,",len(public)//2,"external conversations,",len(provenance["calls"]),"called function names")
