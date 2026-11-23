# phase2_anchor

Development-only block-input re-injection versus direct and output interpolation.

| Run | Split | Updates | Targets | Training s | Answer NLL | Accuracy | Trace NLL | Trace accuracy | Prose NLL | Greedy s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p2_anchor | validation | 24 | 7032 | 24.5 | 1.7932 | 1/30 | 0.1627 | 2/5 | 3.5902 | 4.65 |
| p2_screen_direct | validation | 24 | 7032 | 24.8 | 1.8237 | 1/30 | 0.2028 | 2/5 | 3.7254 | 4.70 |
| p2_screen_stabilized | validation | 24 | 7032 | 24.8 | 1.7787 | 1/30 | 0.1644 | 2/5 | 3.5898 | 4.43 |
