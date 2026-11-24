# Pointer state installation at seed 109

The frozen middle-block probe finished its 480 updates and scored 66, 60 and 54 of 384 development maps at one, two and three hops. Its entire three-hop trajectory was correct on zero maps.

On the identical training schedule, the adapted dense middle control learned one-hop lookup on **384/384** unseen maps. It solved **71/384** two-hop and **60/384** three-hop terminals; only **1/384** had all three states correct. Thus reading one edge from this format is tractable for the pretrained model and block adapters. The open question is whether the model can carry that result into a second lookup.

Both controls saw 3,840 distinct maps and terminal labels. The probe used 47.7 seconds of optimizer work and the adapted dense middle used 107.9 seconds. Results here are provisional while the other registered arms complete. The confirmation cohort remains sealed. [Training protocol](PHASE5_PLAN.md)
