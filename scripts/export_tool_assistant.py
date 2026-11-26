"""Assemble a qualified tool adapter and a verified standalone 128K language release."""
import argparse
import os
from pathlib import Path
import shutil
import tempfile
from loopqwen import REVISION
from loopqwen.io import read, save, digest


def export(language,language_record,trained,qualification,destination):
    cfg=read(language/"config.json")
    if cfg["base_revision"]!=REVISION or cfg["export_schema"]!=4:
        raise ValueError("Expected the pinned conditional language export")
    original=read(language_record/"export-files.json")
    for name,expected in original.items():
        if digest(language/name)!=expected:
            raise ValueError("Language export differs: "+name)
    spec=read(trained/"registration.json")
    fit=read(trained/"fit.json")
    q=read(qualification/"registration.json")
    d=read(qualification/"decision.json")
    worktree=Path(__file__).resolve().parents[1]
    reg=read(worktree/"research/registry.json")["experiments"]
    for source in (trained,qualification,language_record):
        if digest(source/"checksums.json")!=reg[source.name]["archive"]:
            raise ValueError("Unregistered archive: "+source.name)
        for name,h in read(source/"checksums.json").items():
            if digest(source/name)!=h:
                raise ValueError("Archive differs: "+str(source/name))
    if (not d["passed"] or not q.get("stop_at_turn_end") or q["control"]!="base" or
            (qualification.resolve()!=trained.resolve() and q["candidate"]!="restore:research/results/"+trained.name)
            or fit["passes"]<2 or fit["weights"]!=digest(trained/"tool-adapters.safetensors")):
        raise ValueError("The qualification must cover exactly these recurrent weights")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        bundle=Path(temp)/"bundle"
        shutil.copytree(language,bundle)
        out=bundle/"tool"
        out.mkdir()
        shutil.copyfile(trained/"tool-adapters.safetensors",out/"tool-adapters.safetensors")
        save(out/"tool-model.json",dict(schema=1,base_revision=REVISION,passes=fit["passes"],rank=spec["rank"],
                                       start_layer=spec.get("start_layer",23),weights=fit["weights"],training=trained.name,
                                       qualification=qualification.name,qualification_archive=digest(qualification/"checksums.json")))
        # The native inference files are also available to offline consumers of the bundled runtime.
        rt=bundle/"runtime/loopqwen"
        for file in ("tool_model.py","tool_session.py","tool_catalog.py"):
            shutil.copyfile(worktree/"loopqwen"/file,rt/file)
        inventory={str(p.relative_to(bundle)):digest(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
        save(bundle/"tool-export-files.json",inventory)
        os.rename(bundle,destination)
    return destination


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("language","language-record","trained","qualification","destination"):
        parser.add_argument("--"+name,type=Path,required=True)
    print(export(**vars(parser.parse_args())))
