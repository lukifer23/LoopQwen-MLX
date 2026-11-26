"""Freeze complete public-domain books and crossover prompts before release qualification."""

import sys
from pathlib import Path

from mlx_lm.utils import load_tokenizer
from selective_raw_material import download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.local_packages import tools
from loopqwen.identity import fingerprint
from loopqwen.io import read, save
from loopqwen.model_loader import local_snapshot
from loopqwen.research.counterfactual_memory import material

EDITIONS = {
    "A Christmas Carol": "GITenberg/A-Christmas-Carol_46/b173bd3787848ecbded8d036ed9f5326c738da19/46-0.txt",
    "The Strange Case of Dr. Jekyll and Mr. Hyde": "GITenberg/The-Strange-Case-of-Dr.-Jekyll-and-Mr.-Hyde_43/534f1e1d3193ffaf351570cf808e0ee6db70efe9/43-0.txt",
}


def main():
    tok = load_tokenizer(local_snapshot())
    books, sources = [], {}
    for title, edition in EDITIONS.items():
        url = "https://raw.githubusercontent.com/" + edition
        key = "packed-" + edition.rsplit("/", 1)[1]
        text = download(key, url, sources).replace("\r", "")
        first, last = (
            ("STAVE I:  MARLEY'S GHOST", "observed, God bless Us, Every One!")
            if title == "A Christmas Carol"
            else (
                "STORY OF THE DOOR",
                "bring the life of that unhappy Henry Jekyll to an end.",
            )
        )
        if text.count(first) != 1 or text.count(last) != 1:
            raise ValueError("Changed narrative boundaries in the pinned edition")
        body = text[text.index(first) : text.index(last) + len(last)]
        ids = tok.encode(body, add_special_tokens=False)
        if len(ids) < 16384 or tok.decode(ids) != body:
            raise ValueError("Book text failed the length or token round-trip check")
        books.append(
            {"id": title, "source": key, "prompt_ids": ids, "sha256": fingerprint(ids)}
        )
    # One fresh register document; only move the two names, retaining every numeric position.
    ledger = material(tok, {"seed": 9872151, "minimum_length": 24576})
    short = []
    for suite, source, label in (
        ("induction", "native-induction-anchored-01", "replication"),
        ("deduction", "deduction-stories-02", "replication"),
        ("mixed", "conditional-language-02", None),
    ):
        panel = read(Path("research/data") / (source + ".json"))
        items = panel[label] if label else panel
        for r in items[:: len(items) // 4][:4]:
            short.append(
                {
                    "id": r["id"],
                    "suite": suite,
                    "prompt_ids": r["prompt_ids"],
                    "cap": r.get("cap", 768),
                }
            )
    save(
        "research/data/packed-release-retention-01.json",
        {
            "sources": sources,
            "books": books,
            "register": ledger,
            "short": short,
            "tools": [t.definition() for t in tools()],
        },
    )
    print(
        [(b["id"], len(b["prompt_ids"])) for b in books],
        len(ledger["panel"][0]["prompt_ids"]),
        len(short),
    )


if __name__ == "__main__":
    main()
