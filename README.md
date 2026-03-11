# NLP_proj

This repository contains a local-only SciQ RAG project for analyzing how retrieval design choices affect multiple-choice QA performance.

## Contents

- `notebooks/`: minimal end-to-end RAG notebook.
- `scripts/`: experiment scripts for ablations, significance testing, multi-seed evaluation, and context interventions.
- `outputs/`: saved run summaries and per-question outputs used in the report.

## Main Experiments

- Minimal RAG loop on SciQ without external LLM APIs.
- Full ablation over embedding model, chunk size, and top-k.
- Baseline vs best-setting significance testing.
- Multi-seed validation on `n=1000`.
- Context intervention experiments on retrieved support documents.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
