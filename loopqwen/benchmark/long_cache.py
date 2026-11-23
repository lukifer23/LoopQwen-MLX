"""Stream full-vocabulary comparisons in short slabs to bound unified memory."""
from dataclasses import replace
import mlx.core as mx
import numpy as np

from ..io import save
from ..looped_model import LoopedModel
from ..model_loader import load_base


def audit():
    base,tok,_=load_base()
    model=LoopedModel(base)
    model.eval()
    text="The rain passed over the mountains. A stream carried the fallen leaves downstream. "
    tokens=tok.encode(text*100)
    rows=[]
    for precision in ("bf16","fp32"):
        if precision=="fp32":
            base.set_dtype(mx.float32)
            mx.eval(base.parameters())
            mx.clear_cache()
        for k in (1,2,3):
            model.span=replace(model.span,passes=k)
            for length,window in ((17,None),(128,None),(512,None),(512,96),(1024,None)):
                if precision=="fp32" and (k!=3 or length!=512):
                    continue
                if length==1024 and k!=2:
                    continue
                x=mx.array([tokens[:length]])
                ref=model.hidden(x,window_size=window)
                mx.eval(ref)
                for chunk in (1,64):
                    cache=model.make_cache(window)
                    parts=[]
                    for i in range(0,length,chunk):
                        h=model.hidden(x[:,i:i+chunk],cache)
                        mx.eval(h)
                        parts.append(h)
                    actual=mx.concatenate(parts,1)
                    errors=[];maximum=0.; mismatches=[]; margins=[]
                    for i in range(0,length,16):
                        a,b=[model.project(h[:,i:i+16]).astype(mx.float32) for h in (ref,actual)]
                        diff=a-b
                        errors.extend(mx.sqrt(mx.mean(diff**2,-1)/mx.mean(a**2,-1))[0].tolist())
                        maximum=max(maximum,mx.max(mx.abs(diff)).item())
                        best=mx.topk(a,2,axis=-1)
                        margin=mx.max(best,-1)-mx.min(best,-1)
                        bad=(mx.argmax(a,-1)!=mx.argmax(b,-1))[0].tolist()
                        margins.extend(m for m,z in zip(margin[0].tolist(),bad) if z)
                        mismatches.extend(bad)
                    greedy=[];full=[]
                    for _ in range(6):
                        prefix=mx.array([tokens[:length]+full])
                        full.append(mx.argmax(model.project(model.hidden(prefix,window_size=window)[:,-1])).item())
                        greedy.append(mx.argmax(model.project(actual[:,-1])).item())
                        actual=model.hidden(mx.array([[greedy[-1]]]),cache)
                    row=dict(dtype=precision,k=k,length=length,window=window,chunk=chunk,
                             relative_rms_max=max(errors),relative_rms_median=float(np.median(errors)),
                             max_abs=maximum,top1=1-sum(mismatches)/length,changed_top1_margins=margins,
                             rollout_disagreement=sum(a!=b for a,b in zip(full,greedy))/len(full))
                    rows.append(row)
                    save("outputs/phase2/audit/long_cache.json",rows)
                    if not np.isfinite(max(errors)) or (precision=="fp32" and max(errors)>.001):
                        raise AssertionError(row)
    return rows

if __name__=="__main__":
    audit()
