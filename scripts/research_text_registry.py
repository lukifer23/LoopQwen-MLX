"""Content identities of historical language training, calibration and evaluation materials."""
import gzip,hashlib,json
from pathlib import Path
from loopqwen.io import save,digest

def keys(row):
    if not isinstance(row,dict):return set()
    result=set()
    for key in ("id","example_id","question","prompt"):
        value=row.get(key)
        if isinstance(value,str) and value.strip():
            result.add(hashlib.sha256(("id:" if key in ("id","example_id") else "text:").encode()+" ".join(value.split()).encode()).hexdigest())
    for key in ("item","row","records","items","examples","calibration","comparison"):
        v=row.get(key)
        if isinstance(v,dict):result.update(keys(v))
        elif isinstance(v,list):
            for item in v:result.update(keys(item))
    return result

def build():
    root=Path("research/registry");root.mkdir(exist_ok=True)
    sources={};groups={}
    def keep(p,values):
        chunk=("\n".join(sorted(values))+"\n").encode();h=hashlib.sha256(chunk).hexdigest()
        sources[str(p)]=dict(source_sha256=digest(p),item_inventory=h,count=len(values))
        groups[h]=sorted(values)
    for directory in ("pilot","v2","v3","v3_retention","v4","phase2_external","math500"):
        for p in sorted((Path("data")/directory).rglob("*.jsonl")):
            if "source" in p.stem:continue
            values=set()
            for line in p.open():values.update(keys(json.loads(line)))
            if values:keep(p,values)
    for p in sorted(Path("docs/measurements").rglob("*")):
        if p.suffix!=".json" and not p.name.endswith(".json.gz"):continue
        value=json.loads(gzip.decompress(p.read_bytes()) if p.suffix==".gz" else p.read_bytes())
        found=keys(value)
        if isinstance(value,list):
            for row in value:found.update(keys(row))
        if found:keep(p,found)
    payload=gzip.compress(json.dumps(groups,separators=(",",":"),sort_keys=True).encode(),mtime=0)
    p=root/"historical-text-ids.json.gz"
    if p.exists() and p.read_bytes()!=payload:raise ValueError("Historical language inventory drift")
    p.write_bytes(payload)
    save(root/"historical-text.json",dict(sha256=digest(p),sources=sources))
if __name__=="__main__":build()
