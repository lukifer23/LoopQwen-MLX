# Python distributions from two application lockfiles

This cohort contains 48 original release wheels pinned by Pydantic and 12 held releases pinned by FastAPI. The 60 projects are disjoint across cohorts. The 12 held names also fall outside the project lock and the earlier package-metadata training and test corpora listed by the enrollment script. The upstream applications may share other dependencies. [Pydantic source lock](https://github.com/pydantic/pydantic/blob/6449f80db0214a95178b6b80a3613451b924a4f1/uv.lock) · [FastAPI source lock](https://github.com/fastapi/fastapi/blob/05e6df7e906c3decba377c37548019973f40bf15/uv.lock)

The sampler orders candidates by a fixed hash of the split and archive name. It selects platform-independent wheels of up to 1 MiB with an explicit interpreter constraint and at least one permitted and one excluded version among nine released CPython versions. Neighboring versions on opposite sides of a boundary provide the paired compatibility questions. These sources describe the declared interpreter requirement. Native platform tags, dependency installation, and execution would require additional checks for a full installability claim.

The enrollment file contains the wheel URL and SHA-256 from the original application lock, the name and digest of its embedded METADATA member, and independently read values of six headers for each project. The native tool never imports the supplied distribution. It reads the selected header and preserves repeated values in original order. [PyPA wheel format](https://packaging.python.org/en/latest/specifications/binary-distribution-format/) · [PyPA core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/)

To repeat the entire source selection, run the following commands from the project root. The second run works offline from the cached locks and original wheel bytes.

```sh
.venv/bin/python -m scripts.wheel_archive_sources --acquire
.venv/bin/python -m scripts.wheel_archive_sources
```

The application locks are verified against their pinned Git blob identities. Each downloaded wheel is verified against the lock digest and byte length. Its fields are read twice, once through the assistant’s actual provider and once through Python’s distribution discovery on the original ZIP. The package archive retains the license and notices supplied by its own publisher. The 360 field reads are source verification. Model performance requires a separate direct experiment.

The checkout exposes wheel files through its ordinary ask command. For example, this request lets both models inspect one of the actual held archives.

```sh
.venv/bin/loopqwen ask --checkpoint exports/tool-language-packed-05 \
  --tool-format qwen --wheel data/research_external/wheel-archives/wheels/graphql_core-3.3.1-py3-none-any.whl \
  --prompt 'What interpreter constraint is declared in the attached wheel?' \
  --max-turns 3 --max-tokens 256 --trace --compare-base
```
