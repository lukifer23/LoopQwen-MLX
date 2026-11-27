# Complete locale files beyond 131,072 tokens

A recipe application needs the exact display string for a quantity under a locale’s unit formatting rules. Four requests use the complete English and Italian locale files from consecutive Unicode CLDR releases 48.2 and 49. The question, source filename and chat template stay the same within each pair. Only the published file changes. The XML path and the locale’s plural category for two determine the reference. The English example changes letter case. The Italian example changes the abbreviated unit. [Unicode CLDR releases](https://cldr.unicode.org/index/downloads) · [Upstream project](https://github.com/unicode-org/cldr) · [License](LICENSE)

| Locale and release | Opening tokens | First occurrence of the reference pattern, zero-based | Expected display |
| --- | ---: | ---: | --- |
| English 48.2 | 161,874 | 134,123 | `2 tbsp` |
| English 49 | 164,271 | 136,332 | `2 Tbsp` |
| Italian 48.2 | 152,185 | 142,902 | `2dstspn im` |
| Italian 49 | 156,874 | 147,605 | `2cd imp.` |

Every occurrence of the exact source pattern is beyond token 131,072. The fully formatted answer occurs zero times in each rendered input. The long files preserve their original byte order, XML, whitespace and UTF-8. The frozen inputs include the full text and ordinary reasoning-enabled chat framing, without truncation, padding, retrieval or version hints in the question. A fixed default can solve at most one member per language. These four source-qualified requests have no model scores. They measure following a particular changed locale setting and carry no claim about general long-document reasoning. [Exact source revisions and token positions](qualification.json) · [Frozen prompt IDs](prompt-ids.json.gz)

Offline reproduction uses the enclosed raw sources. Adding `--acquire` fetches the same SHA-256-verified files from Unicode. Adding `--write` regenerates the token artifact and receipt.

```sh
.venv/bin/python -m scripts.locale_context_material
```
