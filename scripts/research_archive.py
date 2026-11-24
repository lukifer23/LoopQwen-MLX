"""Seal a completed comparison and independently rescore its archived numeric trajectories."""
import argparse,gzip,hashlib,json,re,shutil,tempfile
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix==".gz" else p.read_bytes())

def verify(root):
    index=read(root/"checksums.json")
    if index!={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p!=root/"checksums.json"}:raise ValueError("Archive integrity")
    decision=read(root/"decision.json")
    pairs=[root/(k+".json.gz") for k in ("parent","child")]
    if all(p.is_file() for p in pairs):
        keys=[]
        for name,p in zip(("parent","child"),pairs,strict=True):
            rows=read(p)["records"];keys.append([(r["id"],r["prompt_ids"]) for r in rows]);right=0
            for r in rows:
                node=r["start"];gold=[]
                for _ in range(3):node=r["map"][node];gold.append(node)
                m=re.fullmatch(r"\s*([0-7])\s*(?:,|->|→)\s*([0-7])\s*(?:,|->|→)\s*([0-7])[.!]?\s*",r["visible"])
                valid=bool(m and r["stop"]!="cap" and list(map(int,m.groups()))==gold)
                if valid!=r["score"]["verified"]:raise ValueError("Incorrect certificate")
                right+=valid
            if right!=decision[name]["verified"]:raise ValueError("Incorrect summary")
        if keys[0]!=keys[1]:raise ValueError("Different evaluation requests")
    print(root.name,len(index),"files verified")

def publish(name):
    root=Path("research/results")/name;src=Path("outputs/research")/name
    if root.exists():return verify(root)
    if not (src/"decision.json").is_file():raise ValueError("No stage decision")
    root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
        w=Path(tmp)/"archive";shutil.copytree(src,w)
        shutil.copy(Path("research/experiments")/(name+".json"),w/"registration.json")
        (w/"checksums.json").write_text(json.dumps({str(p.relative_to(w)):sha(p) for p in sorted(w.rglob("*")) if p.is_file()},indent=2)+"\n")
        verify(w);w.rename(root)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("name");p.add_argument("--verify",action="store_true");a=p.parse_args()
    if a.verify:verify(Path("research/results")/a.name)
    else:publish(a.name)
