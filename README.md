# Analyzing the Impact of Retrieval Design Choices in a Domain-Specific RAG System

This repository corresponds to a TDDE09 project.

## 1. Abstract

Several design choices including document chunk size, the number of retrieved passages (top-k), and the choice of embedding model may significantly affect final answer quality, yet their individual impact is not always systematically evaluated. This project aims to investigate how key retrieval design parameters affect retrieval quality and QA performance in a science-domain RAG system. We change one retrieval factor at a time while keeping others fixed.

## 2. Dataset

We will use the SciQ dataset (https://huggingface.co/datasets/allenai/sciq), a science-domain question answering benchmark. The dataset contains multiple-choice science questions. We will treat the supporting paragraphs as the document collection for retrieval and evaluate answer accuracy.

## 3. Methodology

### 3.1 Baseline RAG system

We will implement a standard RAG pipeline:

1. **Chunking** the supporting documents
2. **Dense retrieval** using Sentence-BERT-style embeddings
3. **Vector index** via FAISS or ChromaDB for similarity search
4. **Answer generation** via an open-source pretrained LLM

The baseline configuration will use:

- **Chunk size**: 256 tokens
- **Top-k retrieval**: 3
- **Embedding model**: all-MiniLM-L6-v2

### 3.2 Experimental Variables

We will perform controlled experiments with:

#### (1) Chunk size

- 256 tokens
- 512 tokens
- 1024 tokens

#### (2) Top-k Retrieval

- k = 1
- k = 3
- k = 5

#### (3) Embedding Model

- all-MiniLM-L6-v2
- msmarco-MiniLM-L6-v3

## 4. Evaluation

### 4.1 Primary metrics: 

We evaluate under two settings.

#### (1) Multiple Choice QA: Accuracy

SciQ is originally a multiple-choice QA dataset, we prompt the model to only output the option letter (A, B, C, or D). Then we use the Accuracy as the main metric, measured as the percentage of questions answered correctly.

#### (2) Generative QA (ignoring the MCQ options): EM / F1 / semantic similarity

For the same dataset SciQ, we also consider a generative QA setting, where we ignore the multiple-choice options and the model directly generates the answer text based on the retrieved context. In this setting, we evaluate performance using Exact Match (EM), F1, and semantic similarity. We use a LLM-as-a-judge to grade if the generated answer is semantically correct, even when its wording differs from the reference answer.

### 4.2 Secondary metrics: Recall@k

Recall@k: measures retriever ability to retrieve relevant supporting documents.

SciQ provides the exact support paragraph for each question. When we build the vector database, we can assign a unique ID to each SciQ paragraph. Then during evaluation, check if the ID of the ground-truth paragraph appears in your top-k retrieved IDs.

### 4.3 Baselines Comparison

- Pure LLM without retrieval
- RAG system

### 4.4 Error Analysis

We can use an automated RAG evaluation framework like RAGAS or TruLens. We can map the error categories to their automated metrics using an LLM-as-a-judge:

- Irrelevant Retrieval → Measure Context Precision.
- Retrieval Failure → Measure Context Recall (did the retrieved chunks contain the answer?).
- Hallucination / Generation Error → Measure Faithfulness (is the answer derived only from the retrieved context?).

### 4.5 Efficiency Analysis

- Token usage
- Retrieval latency

## 5. Implementation Notes

### 5.1 Basic repository structure

```
.
├── data/                     # SciQ raw and processed data
├── scripts/
│   ├── build_index.py         # chunking + embeddings + indexing
│   ├── run_rag.py             # RAG inference
│   └── eval.py                # Accuracy / Recall@k evaluation
├── configs/                   # experiment configs (chunk/topk/embedding)
├── results/                   # metrics, logs, plots
└── README.md
```

### 5.2 Record experiment logging

For each run, record:

- Configuration (chunk_size, top_k, embedding_model, LLM, prompt)
- Metrics (Accuracy, Recall@k, latency, token usage)
- Random seeds and environment (package versions, hardware) 


## 6. Related papers
- *Enhancing Retrieval-Augmented Generation: A Study of Best Practices* https://aclanthology.org/2025.coling-main.449/

- *Optimizing Retrieval-Augmented Generation: Analysis of Hyperparameter Impact on Performance and Efficiency* https://arxiv.org/abs/2505.08445

- *Impact of chunking granularity on accuracy and token consumption in retrieval-augmented generation for question-answering* https://aaltodoc.aalto.fi/items/ee14d13f-4de4-4609-8f23-9379331797ba
