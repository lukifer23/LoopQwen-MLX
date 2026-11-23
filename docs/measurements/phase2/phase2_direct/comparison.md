# phase2_direct

Stage A development comparison of answer-only and verified-trace supervision.

| Run | Split | Updates | Targets | Training s | Answer NLL | Accuracy | Trace NLL | Trace accuracy | Prose NLL | Greedy s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p2_a_answer_dense | validation | 72 | 1220 | 23.4 | 1.1018 | 1/30 | 0.6399 | 0/5 | 3.5912 | 2.79 |
| p2_a_answer_loop | validation | 72 | 1220 | 31.9 | 1.1293 | 2/30 | 0.7983 | 1/5 | 3.7139 | 3.12 |
| p2_a_trace_dense | validation | 72 | 21834 | 59.1 | 2.6195 | 0/30 | 0.0791 | 2/5 | 3.6023 | 5.07 |
| p2_a_trace_loop | validation | 72 | 21834 | 78.5 | 3.0680 | 0/30 | 0.0936 | 2/5 | 3.7432 | 6.40 |
