"""Freeze human-localized territory names from Unicode CLDR release 47."""

from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.request import urlopen

from mlx_lm.utils import load_tokenizer

from loopqwen.io import digest, read, save
from loopqwen.model_loader import local_snapshot

REVISION = "2ef784e3a4168bc2a43cd1b5b9839b6636f5899c"
LOCALES = (
    "ar",
    "bn",
    "de",
    "en",
    "gu",
    "he",
    "hi",
    "ja",
    "kn",
    "ml",
    "mr",
    "pa",
    "ru",
    "ta",
    "te",
    "th",
    "ur",
    "yo",
    "zh",
)


def build():
    cache = Path("data/research_external/cldr") / REVISION
    cache.mkdir(exist_ok=True, parents=True)
    tok = load_tokenizer(local_snapshot())
    sources, rows = {}, []
    for locale in LOCALES:
        url = f"https://raw.githubusercontent.com/unicode-org/cldr/{REVISION}/common/main/{locale}.xml"
        path = cache / (locale + ".xml")
        if not path.is_file():
            with urlopen(url, timeout=45) as stream:
                body = stream.read()
            ET.fromstring(body)
            path.write_bytes(body)
        sources[locale] = {"url": url, "sha256": digest(path)}
        for element in ET.parse(path).findall(
            "./localeDisplayNames/territories/territory"
        ):
            code = element.get("type")
            label = element.text
            if element.get("alt") or not label or label == "↑↑↑":
                continue
            rows.append(
                {
                    "locale": locale,
                    "territory": code,
                    "text": label,
                    "ids": tok.encode(label, add_special_tokens=False),
                }
            )
    assert len({(w["locale"], w["territory"]) for w in rows}) == len(rows)
    original = read("research/data/unified-session-panel.json")["rows"][127]
    return {
        "revision": REVISION,
        "source_hashes": sources,
        "rows": rows,
        "question": {k: original[k] for k in ("id", "question", "gold", "initial_ids")},
    }


if __name__ == "__main__":
    material = build()
    save("research/data/cldr-territory-tokenizer.json", material)
    print(len(material["rows"]), "localized territory names", len(LOCALES), "locales")
