"""Rebuild complete long-context prompts from two adjacent Unicode CLDR releases."""

import argparse
from bisect import bisect_right
import gzip
from hashlib import sha256
from pathlib import Path
import re
from urllib.request import urlopen
from xml.etree import ElementTree

from loopqwen.evaluation.mechanism import packed, unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.qwen_tools import render
from loopqwen.research.cross_view_study import tokenizer

SOURCE = Path("research/sources/locale-context")
REVISIONS = {
    "48.2": "11299982335beb974c1c63c45265184e759c0f41",
    "49": "96e54b273d2da4ec6fbad350ba19ff23fa24a588",
}
HASHES = {
    "en": (
        "67607f4c9cf57157e70564987d8ae92a9ddcd73c00a2f1cd6abf79524a969cbd",
        "f8cbf26964d0e349fc96d9e5a56292edcbaab25c8e64177ca4f111773dfb79cf",
    ),
    "it": (
        "cf1e21c333efa6b61c769be824f8a410503f210f60ee157fa40ed12950dcd632",
        "a9e8612bd4b80477de74bfac9ad2e86f1d73cf7d4c4bec1b365debb6469cbde8",
    ),
}
FIELDS = {
    "en": (
        "short",
        "volume-tablespoon",
        "Our recipe app needs to display 2 tablespoons using the short English unit width. "
        "According to this locale file, exactly what should it display? Give just the formatted "
        "value, preserving capitalization, spacing and punctuation.",
    ),
    "it": (
        "narrow",
        "volume-dessert-spoon-imperial",
        "Our recipe app needs to display 2 imperial dessert spoons using the narrow Italian unit "
        "width. According to this locale file, exactly what should it display? Give just the "
        "formatted value, preserving capitalization, spacing and punctuation.",
    ),
}


def material(acquire=False, write=False):
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    rows = []
    by_id = {}
    assert (
        digest(SOURCE / "LICENSE")
        == "65eb33419e22a297550a811b5b535f5e532f9a7e0114b516c0ea18afd53c3804"
    )
    for locale, (width, unit, question) in FIELDS.items():
        answers = []
        for (release, revision), sha in zip(
            REVISIONS.items(), HASHES[locale], strict=True
        ):
            filename = "common/main/" + locale + ".xml"
            url = f"https://raw.githubusercontent.com/unicode-org/cldr/{revision}/{filename}"
            path = SOURCE / f"{locale}-{release}.xml.gz"
            if acquire:
                with urlopen(url, timeout=40) as stream:
                    payload = stream.read()
                assert sha256(payload).hexdigest() == sha
                path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
            raw = gzip.decompress(path.read_bytes())
            assert sha256(raw).hexdigest() == sha
            xml = raw.decode("utf-8")
            selector = f"./units/unitLength[@type='{width}']/unit[@type='{unit}']/unitPattern[@count='other']"
            patterns = ElementTree.fromstring(raw).findall(selector)
            assert len(patterns) == 1 and patterns[0].text.count("{0}") == 1
            pattern = patterns[0].text
            answer = pattern.format(2)
            answers.append(answer)
            user = f"Here is the complete {filename} file.\n\n{xml}\nEnd of file.\n\n{question}"
            chat = render(tok, [{"role": "user", "content": user}], [])
            assert chat.count(xml) == 1 and answer not in chat
            encoded = tok._tokenizer(
                chat, add_special_tokens=False, return_offsets_mapping=True
            )
            ids = encoded["input_ids"]
            assert ids == tok.encode(chat, add_special_tokens=False)
            # Hugging Face fast tokenizer offsets count Unicode characters, as does re.finditer.
            ends = [b for _, b in encoded["offset_mapping"]]
            starts = []
            for match in re.finditer(re.escape(pattern), chat):
                i = bisect_right(ends, match.start())
                a, b = encoded["offset_mapping"][i]
                assert a <= match.start() < b and i > 131072
                starts.append(i)
            assert starts and 131072 < len(ids) < 262144
            key = f"{locale}-{release}"
            by_id[key] = ids
            rows.append(
                {
                    "id": key,
                    "locale": locale,
                    "release": release,
                    "revision": revision,
                    "url": url,
                    "file_sha256": sha,
                    "file_bytes": len(raw),
                    "question": question,
                    "xml_path": selector,
                    "pattern": pattern,
                    "answer": answer,
                    "pattern_start_tokens": starts,
                    "input_tokens": len(ids),
                    "input_sha256": fingerprint(ids),
                }
            )
        assert answers[0] != answers[1]
    summary = {
        "source": "Unicode CLDR",
        "rows": rows,
        "scope": "Source and scorer qualification. No model inference or model accuracy.",
    }
    frozen = SOURCE / "prompt-ids.json.gz"
    receipt = SOURCE / "qualification.json"
    if write:
        packed(frozen, by_id)
        save(receipt, summary)
    assert by_id == unpack(frozen) and summary == read(receipt)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    for row in material(args.acquire, args.write):
        print(
            row["id"],
            row["input_tokens"],
            row["pattern_start_tokens"],
            repr(row["answer"]),
        )
