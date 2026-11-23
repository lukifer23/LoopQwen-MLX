"""Immutable, fingerprinted reasoning splits with separately encoded response protocols."""
from collections import Counter
from pathlib import Path
import json
import random
import re
import tempfile
import shutil
from urllib.request import urlopen

from transformers import AutoTokenizer
from .. import REVISION
from ..model_loader import local_snapshot
from ..io import read,save
from .build_data import digest
from .reasoning import FAMILIES,problem

ROOT=Path("data/v2")
BOOKS=(("Alice-s-Adventures-in-Wonderland_11","7f18748e114b994d5d9454d5793581bc0b9cf9d7","11"),
       ("The-Secret-Garden_113","0031c0badd0d3fc83860c941d2f39d4aa345c51b","113"))

def encode_protocol(row,tok,trace):
    prompt=row["question"]+("\nShow your steps. End with Final answer: <number>.\nSolution:" if trace else "\nAnswer:")
    prefix=tok.encode(prompt,add_special_tokens=False)
    response=tok.encode(" "+(row["trace"] if trace else row["answer"]),add_special_tokens=False)+[tok.eos_token_id]
    return dict(prompt=prompt,ids=prefix+response,prefix=len(prefix),input_length=len(prefix)+len(response)-1,output_length=len(response))

def build(count=24000,seed=2602,root=ROOT):
    if count<30:raise ValueError("Use at least 30 training instances")
    if root.exists():
        m=read(root/"manifest.json")
        if (m["train_requested"],m["seed"])!=(count,seed):raise ValueError("Choose a new dataset directory")
        for name,s in m["splits"].items():
            if digest(root/(name+".jsonl"))!=s["sha256"]:raise ValueError(name)
        return m
    root.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(dir=root.parent,prefix=".reasoning-"))
    tok=AutoTokenizer.from_pretrained(str(local_snapshot()))
    rng=random.Random(seed); seen=set()
    m=dict(schema=2,generator=digest(Path(__file__).with_name("reasoning.py")),builder=digest(Path(__file__)),
           tokenizer_revision=REVISION,seed=seed,train_requested=count,splits={},sources={"procedural":"Project MIT"},books=[])
    def write(name,rows):
        path=tmp/(name+".jsonl"); cats=Counter();levels=Counter();tokens=Counter()
        with path.open("w") as out:
            for row in rows:
                cats[row["category"]]+=1;levels[str(row["complexity"])]+=1
                for mode,v in row["variants"].items():tokens[mode]+=v["output_length"]
                out.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
        m["splits"][name]=dict(count=sum(cats.values()),sha256=digest(path),categories=dict(cats),levels=dict(levels),supervised=dict(tokens))
    def tasks(n,ood):
        for j in range(n):
            for _ in range(1000):
                r=problem(rng,FAMILIES[j%len(FAMILIES)],j//len(FAMILIES)%3,ood)
                if r["id"] not in seen:break
            else:raise ValueError("Exhausted unique instances")
            seen.add(r["id"])
            r["variants"]={k:encode_protocol(r,tok,k=="trace") for k in ("answer","trace")}
            yield r
    try:
        for split,n in (("train",count),("validation",450),("test",300),("ood",300)):
            write(split,tasks(n,split=="ood"))
        for idx,(repo,rev,num) in enumerate(BOOKS):
            url=f"https://raw.githubusercontent.com/GITenberg/{repo}/{rev}/"
            metadata=urlopen(url+"metadata.yaml",timeout=30).read().decode()
            if "rights: Public domain in the USA." not in metadata:raise ValueError("Unexpected source rights")
            content=urlopen(url+num+".txt",timeout=30).read().decode("utf-8-sig").replace("\r","")
            body=re.split(r"\*\*\* END OF",re.split(r"\*\*\* START OF[^\n]*\n",content,maxsplit=1,flags=re.I)[1],maxsplit=1,flags=re.I)[0]
            paragraphs=[]
            for para in body.split("\n\n"):
                para=" ".join(para.split())
                if not 180<len(para)<650:continue
                ids=tok.encode(para,add_special_tokens=False)
                if not 35<=len(ids)<=128:continue
                import hashlib
                key=hashlib.sha256(para.encode()).hexdigest()
                if key in seen:continue
                seen.add(key)
                paragraphs.append(dict(id=key,category="language",complexity=0,steps=0,answer="",template=repo,
                    variants={"language":dict(ids=ids,prefix=1,input_length=len(ids)-1,output_length=len(ids)-1,prompt="")}))
            rng.shuffle(paragraphs)
            if idx==0:write("language_train",paragraphs)
            else:
                write("language_validation",paragraphs[:64])
                write("language_test",paragraphs[64:192])
            m["books"].append(dict(repo=repo,url=url+num+".txt",rights="Public domain in the USA",revision=rev))
        save(tmp/"manifest.json",m);tmp.rename(root)
    finally:
        if tmp.exists():shutil.rmtree(tmp)
    return m

def rows(split,mode="answer",root=ROOT):
    path=root/(split+".jsonl")
    if digest(path)!=read(root/"manifest.json")["splits"][split]["sha256"]:raise ValueError(f"Changed split {split}")
    result=[]
    for line in path.read_text().splitlines():
        r=json.loads(line); variants=r.pop("variants")
        selected=mode
        if selected=="mixed":selected="trace" if int(r["id"][:8],16)%2 else "answer"
        result.append(r|variants[selected]|{"protocol":selected})
    return result
