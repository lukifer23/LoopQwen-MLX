# Next experiment

**Test whether interleaving prior permutation training maps prevents catastrophic forgetting during functional-graph adaptation.** Static training raised sixth-hop accuracy from 4/128 to 53/128 but cut third-hop permutation accuracy from 96/96 to 12/96. Replaying the learner's own difficult examples reached 33/128 and retained 11/96. This provides a specific reason to repeat old-task examples during adaptation.

The frozen comparison gives both forks 96 identical new maps, 16 updates and the same six-hop parent. Two of eight examples in each logical update carry either repeats of functional maps or old permutation training maps. The measured-safe physical position cap is 84. The treatment must gain ten points on the new task and lose at most two of 96 prior third hops. Exact map IDs and display orders are already frozen. No optimizer update or score from this study is claimed. [Registration](../research/experiments/functional-interleave-01.json) · [Frozen schedule](../research/data/functional-interleave-01.json)

The prior session closed after the system memory guard triggered during evaluation. Start this comparison only in a newly authorized window with normal system pressure and sufficient available memory.

```sh
.venv/bin/python scripts/research_session.py
.venv/bin/python -m loopqwen.research.interleave
```

The short external math adaptation taught valid turn completion but missed its answer-quality threshold. Extending its reasoning benchmark, training the gated dense attribution control, or using its outputs for self-training would require new evidence.
