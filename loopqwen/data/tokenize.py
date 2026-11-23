import numpy as np
import mlx.core as mx


def encode(row, tokenizer):
    prefix=tokenizer.encode(row["prompt"],add_special_tokens=False)
    response=tokenizer.encode(" "+row["answer"],add_special_tokens=False)
    return dict(**row, ids=prefix+response+[tokenizer.eos_token_id], prefix=len(prefix))


def batch(rows, response_only=True):
    # Independent right-padded documents. Causality makes padding invisible to valid predecessors.
    length=max(16,max(len(r["ids"])-1 for r in rows))
    x=np.zeros((len(rows),length),np.int32)
    positions=[]; labels=[]
    for i,row in enumerate(rows):
        ids=row["ids"]
        x[i,:len(ids)-1]=ids[:-1]
        first=row["prefix"] if response_only else 1
        positions.extend(i*length+j-1 for j in range(first,len(ids)))
        labels.extend(ids[first:])
    return mx.array(x),mx.array(positions),mx.array(labels)
