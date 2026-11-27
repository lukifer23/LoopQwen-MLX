# Human-written open-ended assistance

OpenAssistant volunteers wrote and ranked alternative replies in conversation trees. We use the authors’ published OASST1 train and validation partitions at revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`. We retain no user identifiers. [Köpf et al., NeurIPS 2023](https://arxiv.org/abs/2304.07327) · [OpenAssistant’s release and Apache 2.0 license](https://huggingface.co/datasets/OpenAssistant/oasst1)

Both the root question and reply must be English, reviewed, accepted and marked nonsynthetic. We require mean annotated quality of at least 0.66 and zero reported flags for spam, personal information, inappropriate material, language mismatch, violence, hate, sexual content or failure to address the task. We select a first-ranked reply of 15–200 words to an opening question of at most 120 words. Among 329 eligible training trees and 14 validation trees a fixed hash draws 256 and 12, respectively. The selected questions and trees are disjoint. The source includes factual, mathematical, practical and creative requests and is limited to the types of answers that pass these criteria. A volunteer response is one reference, and its claims require independent assessment.

Run `.venv/bin/python -m scripts.human_assistance_sources --acquire` to reproduce the exact selection and source checksums.
