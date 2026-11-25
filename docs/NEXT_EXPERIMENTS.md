# Research decisions

## November 25

The mixed-retention comparison completed all 16 updates per arm after the owner freed unified memory. Both reached 21/128 functional sixth hops and 9/128 complete paths. Mixing old maps preserved 57/96 permutation third hops, versus the control’s 39/96 and the parent’s 96/96. The 94/96 retention floor closed this branch. [Paired archive](../research/results/functional-interleave-01)

## Previous November 24 window

**The November 24 execution window closed at the owner’s request.** The monitor exited and the native session closed on November 23 at 9:47 p.m. Central (November 24 at 03:47 UTC). The saved control is at **1/16** optimizer updates and the treatment is at **0/16**. No paired evaluation or promotion decision exists for this study. [Closure and final readings](../research/sessions/interleave-closure-1795489467/closure.json)

### What blocked training in that window

The earlier trial completed one eight-map update with 48 state labels. The runtime guard then stopped the worker after 5.32 seconds, with a 2.38 GiB peak MLX allocation. The guard checks both macOS memory pressure and the study’s 1.5 GiB minimum system availability. The earlier exception recorded a combined memory warning, so the triggering measurement cannot be distinguished retrospectively. [Interruption record](../research/sessions/2026-11-24-resumed/interleave-control-step1/interruption.json) · [Registered limits](../research/experiments/functional-interleave-01.json)

For the resumed window we chose a conservative launch margin of five consecutive readings at or above **6.4 GiB available**, sampled eight seconds apart, with normal OS pressure and a free GPU lease. The previous trial had stopped even after preflight readings reached 6.38 GiB. The new preflight saw six OS warnings in 31 readings. The subsequent monitor saw normal pressure in all **288** readings, but available memory ranged from **4.01 to 5.98 GiB**. Zero readings reached the launch margin. The monitor spent **zero native worker seconds** and exited before its deadline at the owner’s request. [Preflight measurements](../research/sessions/interleave-resume-1795489467.json) · [Monitor samples](../research/sessions/interleave-closure-1795489467/monitor.tsv)

### Archived resumption instructions

When the owner next authorizes training and memory headroom improves, finish the registered comparison of mixed-task replay. Training only on arbitrary successor maps previously raised sixth-hop accuracy from 4/128 to 53/128 while cutting third-hop permutation accuracy from 96/96 to 12/96. Both new arms see the same 96 fresh functional maps. Two slots per logical batch contain either repeated functional maps (control) or previously seen permutation maps (treatment). [Frozen training schedule](../research/data/functional-interleave-01.json) · [Prior scores](CAPABILITY_SCOREBOARD.md)

1. On the original training host, verify the saved control weights and AdamW moments. Open one authorized window after checking available memory and the GPU lease.
2. Restore the control at update one. Finish its 15 remaining updates and the treatment’s 16 under the same 84-position cap. The two arms share **174.68 seconds** of remaining study worker time.
3. Independently score sixth-hop predictions and complete trajectories on the 128 functional maps. Report paired changes and the 96-map permutation retention score. Promotion requires a gain of at least ten percentage points over the parent on functional sixth hops and at most two lost permutation third hops. If the budget or memory guard stops execution first, preserve the completed boundary.

```sh
python3 scripts/verify_research_recovery.py research/sessions/2026-11-24-resumed/interleave-control-step1
.venv/bin/python scripts/research_session.py
.venv/bin/python -m loopqwen.research.interleave
```

The compact adapter and metadata are published. Exact recovery also uses the original locally retained optimizer file. [Checkpoint and optimizer digest](../research/sessions/2026-11-24-resumed/interleave-control-step1/recovery.json)
