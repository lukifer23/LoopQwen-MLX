"""Freeze new ten-document human reasoning questions with disjoint support titles."""

from mlx_lm.utils import load_tokenizer
from loopqwen.io import read, save
from loopqwen.model_loader import local_snapshot
from bridge_reading_material import canonical, enroll


def main():
    excluded = set()
    for name in (
        "bridge-reading-train",
        "bridge-reading-validation",
        "release-mixed-field-01",
    ):
        obj = read("research/data/" + name + ".json")
        for w in obj.get("examples", obj.get("rows")):
            if w.get("family") == "reading":
                body, question = (
                    w["question"]
                    .split("Documents\n\n", 1)[1]
                    .rsplit("\n\nQuestion\n", 1)
                )
                excluded.add(canonical(question))
                excluded.update(
                    canonical(block.split("\n", 1)[0]) for block in body.split("\n\n")
                )
            if "titles" not in w:
                continue
            excluded.add(canonical(w["question"]))
            excluded.update(canonical(w["titles"][i]) for i in w["support"])
    tok = load_tokenizer(local_snapshot())
    for split, count in (("validation", 48), ("train", 1536)):
        obj = enroll(split, count, excluded, tok, "linked-native-reader-301772")
        rows = obj.get("examples", obj.get("rows"))
        for r in rows:
            for key in ("messages", "visible", "thinking"):
                r.pop(key, None)
        obj.pop("examples", None)
        obj["rows"] = rows
        save("research/data/linked-reading-" + split + ".json", obj)
        print(split, len(rows), flush=True)


if __name__ == "__main__":
    main()
