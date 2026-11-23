"""Exact procedural tasks. Each renderer checks its trace against a second calculation."""
import hashlib
import json
import math
import operator
from fractions import Fraction

FAMILIES=("arithmetic","system","modular","dependencies","array")
OPS={"+":operator.add,"-":operator.sub,"*":operator.mul}

def problem(rng,family,level,ood=False):
    n=3+level+(2 if ood else 0)
    a=rng.randint(101 if ood else 2,260 if ood else 48)
    terms=[rng.randint(2,17) for _ in range(n)]
    lines=[]
    if family=="arithmetic":
        ops=rng.choices(list(OPS),k=n)
        x=a
        expression=str(a)
        for op,v in zip(ops,terms):
            new=OPS[op](x,v)
            lines.append(f"{x} {op} {v} = {new}")
            expression=f"({expression} {op} {v})"
            x=new
        # Only internally generated integers and the three literal operators enter eval.
        gold=eval(expression,{"__builtins__":{}},{})
        question=f"Evaluate {expression}."
        key=[a,ops,terms]; method="independent Python integer expression"
    elif family=="system":
        b,c,d=terms[:3]; u=a; v=rng.randint(-30,45)
        e=rng.randint(2,12)
        if b*e==c*d:e+=1
        f=b*u+c*v; g=d*u+e*v
        # Ask for a derived quantity to add real dependencies at higher levels.
        multiplier=terms[-1] if level or ood else 1
        offset=terms[-2] if level==2 or ood else 0
        question=f"{b}x + {c}y = {f}; {d}x + {e}y = {g}. Find {multiplier}x + y + {offset}."
        det=b*e-c*d
        xx=Fraction(f*e-c*g,det); yy=Fraction(b*g-d*f,det)
        assert (xx,yy)==(u,v) and b*xx+c*yy==f and d*xx+e*yy==g
        lines=[f"D = {b}*{e} - {c}*{d} = {det}",f"x = ({f}*{e} - {c}*{g})/{det} = {xx}",
               f"y = ({g} - {d}*{xx})/{e} = {yy}",f"{multiplier}*{xx} + {yy} + {offset} = {multiplier*xx+yy+offset}"]
        x=multiplier*xx+yy+offset; gold=multiplier*u+v+offset
        key=[sorted([(b,c,f),(d,e,g)]),multiplier,offset]; method="Cramer rule and substitution into both constraints"
    elif family=="modular":
        mod=rng.choice([11,13,17,19,23,29])
        x=a; scale=1; intercept=0
        for v in terms:
            old=x; x=(x*v+v-1)%mod
            scale*=v; intercept=intercept*v+v-1
            assert x==(a*scale+intercept)%mod
            lines.append(f"({old}*{v}+{v-1}) mod {mod} = {x}")
        question=f"Start at {a}. For each t in {terms}, replace x by (t*x+t-1) mod {mod}. Find the final x."
        gold=(a*math.prod(terms)+math.prod(terms)-1)%mod
        key=[a,terms,mod]; method="composed affine map and modular reduction"
    elif family=="dependencies":
        # A DAG with joins. Readiness requires every prerequisite to finish.
        times=[a]+terms
        edges=[[]]+[[rng.randrange(i)] for i in range(1,n+1)]
        for i in range(2,n+1):
            edges[i]=sorted(set(edges[i]+[i-1]))
        finishes=[]; specs=[]
        def paths(i):
            return [times[i]+t for j in edges[i] for t in paths(j)] if edges[i] else [times[i]]
        for i,(t,parents) in enumerate(zip(times,edges)):
            start=max((finishes[j] for j in parents),default=0)
            finishes.append(start+t); assert finishes[i]==max(paths(i))
            specs.append(f"{i}: {t} after {','.join(map(str,parents)) if parents else 'none'}")
            lines.append(f"job {i} finishes at {start}+{t} = {start+t}")
        question="Jobs start as soon as all prerequisites finish. Job: duration after prerequisites. "+"; ".join(specs)+f". When does job {n} finish?"
        x=finishes[-1]; gold=max(paths(n)); key=[times,edges];method="enumerated source-to-sink path durations"
    else:
        values=[a]+terms
        shift=rng.randrange(1,n)
        stride=3 if ood else 2
        ordered=sorted(values,reverse=ood)
        rotated=ordered[shift:]+ordered[:shift]
        selected=rotated[::stride]
        x=sum(selected)
        gold=sum(ordered[(i+shift)%len(values)] for i in range(0,len(values),stride))
        question=f"Sort {values} {'descending' if ood else 'ascending'}, rotate left {shift}, then sum indices 0,{stride},{stride*2},... ."
        lines=[f"Sorted: {ordered}",f"Rotated: {rotated}",f"Selected: {selected}"," + ".join(map(str,selected))+f" = {x}"]
        key=[sorted(values),shift,stride,ood];method="modular index mapping vs list transformation"
    assert x==gold
    prompt_forms=["{}","Compute the result. {}","Solve this problem. {}"] if not ood else ["Report the requested quantity. {}"]
    form=rng.randrange(len(prompt_forms)); question=prompt_forms[form].format(question)
    fingerprint=hashlib.sha256(json.dumps([family,key],separators=(",",":")).encode()).hexdigest()
    return dict(id=fingerprint,category=family,complexity=level,steps=len(lines),template=f"{family}-{'ood' if ood else 'id'}-{form}",
                question=question,answer=str(gold),trace="\n".join(lines)+"\nFinal answer: "+str(gold),verification_method=method)
