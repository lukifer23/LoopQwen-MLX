# Next experiment

**Finish the registered retention comparison when the Mac has sustained memory headroom.** Training exclusively on arbitrary successor maps improved sixth-hop accuracy from 4/128 to 53/128 but cut third-hop permutation accuracy from 96/96 to 12/96. The pending comparison tests whether mixing two prior permutation examples into each batch of eight preserves the old operation.

Both arms have the same six-hop parent and 96 identical fresh maps. The functional-only control fills the other 32 slots with repeated functional maps. The treatment fills those slots with old permutation training maps. Success requires a ten-point new-task gain and at most two lost prior third hops. [Registration](../research/experiments/functional-interleave-01.json) · [Frozen schedule](../research/data/functional-interleave-01.json)

At the 84-position cap, the control completed **1/16** optimizer updates. The treatment completed **0/16**. The system memory guard stopped the worker after 5.32 seconds at 2.38 GiB peak MLX allocation. Preflight had found normal OS pressure and 5.34–6.38 GiB available across six checks. That short recovery proved insufficient for sustained training. The session is closed. The remaining study allowance on this host is 174.68 seconds. The saved generation includes the exact first logical batch and its optimizer state. The published compact copy records the local optimizer digest. [Snapshot](../research/sessions/2026-11-24-resumed/interleave-control-step1) · [Session accounting](../research/sessions/2026-11-24-resumed/execution.json)

On the original training host, run the recovery check first. In the next authorized window with materially improved memory conditions, open a session and resume. The runner restores AdamW and the saved cursor.

```sh
python3 scripts/verify_research_recovery.py research/sessions/2026-11-24-resumed/interleave-control-step1
.venv/bin/python scripts/research_session.py
.venv/bin/python -m loopqwen.research.interleave
```

The external math pilot produced one correct answer from 32 questions. Its accuracy gate closed the larger math comparison. The paired terminal-number study favored ordinary-depth training and leaves little support for spending more compute on the same three-pass objective. [Capability scoreboard](CAPABILITY_SCOREBOARD.md)
