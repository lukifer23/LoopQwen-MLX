# Independent long-context reading material

Four expert-written LongBench v2 questions retain their complete original contexts, questions, and four choices. The actual Qwen openings contain 143,734, 188,673, 185,635, and 148,845 tokens. They cover literature, event ordering, detective reasoning, and academic reading. Each context digest and subdomain appears once. The prompts ask for a choice and a source-supported explanation.

Selection uses the record ID, subdomain, context identity, and full tokenized length. The collection script does not consult the correct option, run either model, or add training examples. This four-source panel can test a qualifying policy on ordinary expert questions, alongside the separate changed-source requirement. Public availability before the donor was trained leaves pretraining exposure unknown. It cannot estimate performance across the entire benchmark.

Run `.venv/bin/python -m scripts.human_long_context_sources --download` to verify the pinned upstream file and regenerate the input hashes. Only the hashes and source descriptors are distributed here. [Published benchmark](https://arxiv.org/abs/2412.15204) · [Dataset](https://huggingface.co/datasets/zai-org/LongBench-v2) · [Frozen inputs](panel.json)
