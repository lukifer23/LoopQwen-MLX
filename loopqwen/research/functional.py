"""Measure zero-shot state advancement on general deterministic successor maps."""
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from . import core
from ..evaluation.mechanism import infer, packed
from ..training.mechanism import requests
from ..training.exposure import ARM
from ..io import save, read, digest


def orbit(mapping, start):
    path = []
    node = start
    while node not in path:
        path.append(node)
        node = mapping[node]
    first = path.index(node)
    return first, len(path) - first


def summarize(records):
    gold = np.array([r["states"] for r in records])
    pred = np.array([r["predicted"] for r in records])
    starts = np.array([r["start"] for r in records])[:,None]
    probs = np.array([r["probabilities"] for r in records])
    prev_gold = np.column_stack((starts,gold[:,:-1]))
    prev_pred = np.column_stack((starts,pred[:,:-1]))
    correct = pred == gold
    path = correct.cumprod(1)
    revisited = np.stack([(gold[:,j:j+1] == prev_gold[:, :j+1]).any(1) for j in range(gold.shape[1])],axis=1)
    stay = gold == prev_gold
    dist = np.array([r["map"] for r in records])
    follows = pred == dist[np.arange(len(records))[:,None],prev_pred]
    p = np.take_along_axis(probs,gold[:,:,None],axis=2)[:,:,0]
    def counts(events,mask):
        return {"numerator":(events & mask).sum(0).tolist(),"denominator":mask.sum(0).tolist()}
    boot = np.random.default_rng(95731).integers(0,len(records),(5000,len(records)))
    return dict(n=len(records),correct=correct.sum(0).tolist(),trajectory=path.sum(0).tolist(),
                terminal_ci=np.quantile(correct[boot,-1].mean(1),[.025,.975]).tolist(),
                nll=(-np.log(np.maximum(1e-30,p))).mean(0).tolist(),
                revisited=counts(correct,revisited),novel=counts(correct,~revisited),
                stationary=counts(correct,stay),incorrect_stall=counts((pred==prev_pred)&~correct,~stay),
                follows_prediction=follows.sum(0).tolist(),
                start_counts=dict(Counter(r["start"] for r in records)),
                orbit_shapes=dict(Counter(f"{r['transient']}/{r['cycle']}" for r in records)))


def run():
    s = read("research/experiments/functional-zero-01.json")
    name = s["id"]
    out = core.OUT/name
    if (out/"decision.json").exists():
        return
    b = core.Budget(name,s["limits"])
    try:
        w = Path(s["parent_path"])
        if digest(w/"adapters.safetensors") != s["sha256"]:
            raise ValueError("Unexpected parent")
        m,t=core.model(w,s["depth"])
        result={}
        for kind in ("permutation","functional"):
            rows=core.reserve(name+"-"+kind,s["seed"],{"dev":s["maps_each"]},functional=kind=="functional")["dev"]
            records=[]
            begin=perf_counter()
            for i in range(0,len(rows),s["batch"]):
                b.check()
                group=rows[i:i+s["batch"]]
                docs=requests(group,s["depth"],ARM,t,{})
                p,_=infer(m,ARM,[docs],s["depth"],False)
                for r,d,prob in zip(group,docs,np.array(p),strict=True):
                    transient,cycle=orbit(r["map"],r["start"])
                    records.append(dict(**r,ids=d["ids"],probabilities=prob.tolist(),predicted=prob.argmax(1).tolist(),transient=transient,cycle=cycle))
            seconds=perf_counter()-begin
            packed(out/(kind+".json.gz"),dict(weights=s["sha256"],records=records))
            result[kind]=dict(**summarize(records),seconds=seconds)
        f=result["functional"]
        result["passed"]=f["correct"][-1]/f["n"]>=.90 and f["trajectory"][-1]/f["n"]>=.85
        save(out/"decision.json",result)
    finally:
        b.close()

if __name__=="__main__":
    run()
