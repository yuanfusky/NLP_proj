# NLP_proj

This repository contains a local-only SciQ RAG project for analyzing how retrieval design choices affect multiple-choice QA performance.

## Project Scope

The project studies how RAG design choices affect SciQ multiple-choice question answering under a reproducible local setup.

- Task: SciQ MCQ answering with answer format restricted to `A/B/C/D`.
- Retriever: dense retrieval with FAISS and Sentence-Transformers embeddings.
- Generator: local Hugging Face seq2seq model, without Gemini/OpenAI API usage.
- Evaluation focus: accuracy, retrieval quality, significance testing, multi-seed stability, and context intervention effects.

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

## Open-Source Models and Attribution Notes

The project uses the following open-source models from Hugging Face:

1. `sentence-transformers/all-MiniLM-L6-v2`
- Role: default embedding model for dense retrieval.
- Summary: a Sentence-Transformers encoder that maps sentences and short paragraphs into 384-dimensional dense vectors for semantic search and retrieval.
- License note: the Hugging Face model card lists this model under `Apache-2.0`.
- Attribution: model page: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>

2. `sentence-transformers/msmarco-MiniLM-L6-v3`
- Role: alternative embedding model used in the ablation study.
- Summary: a Sentence-Transformers retrieval-oriented encoder trained for semantic search and passage ranking.
- License note: the Hugging Face model card lists this model under `Apache-2.0`.
- Attribution: model page: <https://huggingface.co/sentence-transformers/msmarco-MiniLM-L6-v3>

3. `google/flan-t5-small`
- Role: local generator for MCQ answer prediction in the minimal loop and follow-up experiments.
- Summary: a compact instruction-tuned FLAN-T5 sequence-to-sequence language model used here to generate only the final option letter.
- License note: the Hugging Face model card lists this model under `Apache-2.0`.
- Attribution: model page: <https://huggingface.co/google/flan-t5-small>

Additional data dependency:

1. `allenai/sciq`
- Role: benchmark dataset for science-domain multiple-choice QA and support paragraphs.
- Attribution: dataset page: <https://huggingface.co/datasets/allenai/sciq>
- Note: please check the upstream dataset page for the latest dataset usage terms and citation details.

This repository contains our code, experiment scripts, and saved outputs under the Apache License 2.0. Upstream models and datasets keep their own original licenses and attribution requirements.

## Summary of Findings

1. RAG clearly outperformed the pure LLM baseline in this setup.
- In the larger `n=1000`, 3-seed validation run, the LLM-only baseline reached about `0.437` accuracy, while the best RAG setting reached about `0.814`.

2. The strongest configuration in the main ablation was `all-MiniLM-L6-v2 + chunk size 256 + top-k 3`.
- Increasing `top-k` improved retrieval coverage, but larger context also increased latency and token cost.

3. Chunk size mattered less than retrieval depth in the tested range.
- The main performance pattern was driven more by `top-k` and embedding choice than by moving from `256` to `512` or `1024` tokens.

4. The best RAG result was statistically significant compared with the LLM-only baseline.
- The paired significance test showed a large positive accuracy gap and a very small McNemar p-value.

5. Context intervention experiments showed that answer-bearing context is the main source of gain.
- Removing the answer sentence caused a large accuracy drop.
- Keeping only the answer sentence slightly improved accuracy, suggesting that extra context can add noise.
- Adding a random distractor paragraph slightly hurt performance.
- Shuffling retrieved context order had only a small effect in this setup.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
