# phase2_screen

Equal-update development screens of curriculum, capacity, MLP-only placement and response-format mixture.

| Run | Split | Updates | Targets | Training s | Answer NLL | Accuracy | Trace NLL | Trace accuracy | Prose NLL | Greedy s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p2_screen_direct | validation | 24 | 7032 | 24.8 | 1.8097 | 2/60 | 0.2044 | 5/15 | 3.8020 | 4.11 |
| p2_screen_gradual | validation | 24 | 7032 | 23.4 | 1.7939 | 2/60 | 0.2126 | 5/15 | 3.8022 | 6.31 |
| p2_screen_stabilized | validation | 24 | 7032 | 24.8 | 1.7554 | 3/60 | 0.1639 | 6/15 | 3.6699 | 4.23 |
| p2_screen_mixed | validation | 24 | 7032 | 24.0 | 1.8090 | 2/60 | 0.2141 | 5/15 | 3.8000 | 4.05 |
| p2_screen_mlp8 | validation | 24 | 7032 | 25.8 | 1.6445 | 0/60 | 0.2422 | 6/15 | 3.7498 | 3.86 |
| p2_screen_mlp10 | validation | 24 | 7032 | 22.8 | 1.6854 | 3/60 | 0.2683 | 5/15 | 3.7919 | 4.14 |
| p2_screen_mlp16 | validation | 24 | 7032 | 19.2 | 1.8138 | 3/60 | 0.3496 | 4/15 | 3.8449 | 3.95 |
| p2_screen_rank16 | validation | 24 | 7032 | 24.9 | 2.0205 | 0/60 | 0.1402 | 5/15 | 3.8133 | 4.41 |
| p2_screen_nearby | validation | 24 | 7032 | 27.4 | 2.0022 | 2/60 | 0.1620 | 5/15 | 3.8083 | 4.47 |
| p2_screen_formats | validation | 24 | 3607 | 23.5 | 1.3105 | 0/60 | 0.2602 | 5/15 | 3.7979 | 4.81 |
| p2_screen_formats_dense | validation | 24 | 3607 | 17.3 | 1.2756 | 4/60 | 0.2167 | 6/15 | 3.6675 | 2.80 |
