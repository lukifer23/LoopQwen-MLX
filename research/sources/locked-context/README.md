# Complete JSON Schema lockfiles

The upstream commit “Also allow newer Virtue” updates the pin in a real Python project's lockfile. Each gzip member preserves every byte of uv.lock at one of the two directly adjacent commits. COPYING is the upstream license at both revisions. The earlier Git blob is 72f0c098f10712d65ab0855c5b038069518a65cd. The later blob is 795385ed4932e1119d9e32e2ad0e5d337f3f1368. [Original change](https://github.com/python-jsonschema/jsonschema/commit/4ba8468a0130d890328bb821786bb74bdca32298)

Run `.venv/bin/python -m scripts.locked_context_material` to check the original Git blobs, parse both whole TOML documents, verify the one changed version among shared package names, and rebuild both chat prompts with the installed release tokenizer. The first token of the primary version declaration lies past position 131,072 in both prompts. The original files also contain distribution URLs. All of them remain available to the models. Add `--acquire` to independently fetch both original lockfiles before verification.

The two lockfile commits were published eight minutes apart on November 14, 2026. The pinned Base and instruction-sibling revisions were published on April 23 and March 2, respectively. The exact revised document therefore postdates both published snapshots. The earlier version string can occur in other material. [Publisher revision dates](chronology.json)

The correct version occurs exactly three times in each entire rendered prompt. The earliest occurrences are at token offsets 138,342 and 136,058. The expected byte count does not occur anywhere in its corresponding file. The material verifier checks each original prefix against the full tokenizer output. [Every occurrence](answer-locations.json)
