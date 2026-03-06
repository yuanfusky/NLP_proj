import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_option_letter(text: str | None) -> str | None:
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        ans = str(obj.get("answer", "")).strip().upper()
        if ans in {"A", "B", "C", "D"}:
            return ans
    except Exception:
        pass
    m = re.search(r"\b([ABCD])\b", s.upper())
    return m.group(1) if m else None


def make_mcq(example: dict, seed: int):
    rnd = random.Random(seed)
    choices = [
        example["distractor1"],
        example["distractor2"],
        example["distractor3"],
        example["correct_answer"],
    ]
    rnd.shuffle(choices)
    gold_idx = choices.index(example["correct_answer"])
    gold_letter = "ABCD"[gold_idx]
    return example["question"], choices, gold_letter


def format_llm_prompt(question: str, choices: list[str]) -> str:
    return (
        "Answer the multiple-choice science question.\n"
        "Return only one capital letter: A, B, C, or D.\n"
        "Do not provide explanation.\n\n"
        f"Question: {question}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n"
    )


def format_rag_prompt(context_passages: list[str], question: str, choices: list[str]) -> str:
    context = "\n\n".join([f"[Context {i+1}]\n{p}" for i, p in enumerate(context_passages)])
    return (
        "Use the retrieved context to answer the multiple-choice science question.\n"
        "Return only one capital letter: A, B, C, or D.\n"
        "Do not provide explanation.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n"
    )


def chunk_text_tokens(text: str, chunk_tokens: int, overlap_tokens: int, tokenizer) -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return [text]

    chunks: list[str] = []
    stride = max(1, chunk_tokens - overlap_tokens)
    for start in range(0, len(ids), stride):
        end = min(len(ids), start + chunk_tokens)
        sub = ids[start:end]
        if not sub:
            break
        out = tokenizer.decode(sub, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(out)
        if end == len(ids):
            break
    return chunks


def build_chunk_corpus(
    support_texts: list[str],
    support_to_id: dict[str, int],
    chunk_tokens: int,
    overlap_tokens: int,
    tokenizer,
):
    chunk_docs: list[str] = []
    chunk_to_support_id: list[int] = []
    for s in support_texts:
        sid = support_to_id[s]
        for ch in chunk_text_tokens(s, chunk_tokens, overlap_tokens, tokenizer):
            chunk_docs.append(ch)
            chunk_to_support_id.append(sid)
    return chunk_docs, chunk_to_support_id


def load_generator(model_name: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    mdl.to(device)
    mdl.eval()
    return tok, mdl


def generate_answer(prompt: str, tokenizer, model, device: torch.device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False, num_beams=1)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    pred = extract_option_letter(text)
    input_tokens = int(inputs["input_ids"].shape[1])
    output_tokens = int(out.shape[1])
    return pred, input_tokens, output_tokens


def main():
    seed = 0
    set_seed(seed)

    os_device = torch.device("cpu")
    embed_device = "cpu"

    split = "validation"
    n_eval = 200
    top_k_list = [1, 3, 5]
    chunk_settings = [(256, 64), (512, 128), (1024, 256)]
    embed_models = [
        ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
        ("msmarco-MiniLM-L6-v3", "sentence-transformers/msmarco-MiniLM-L6-v3"),
    ]
    generator_name = "google/flan-t5-small"
    chunk_tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"

    out_dir = Path("outputs/full_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading dataset...")
    dataset = load_dataset("sciq")

    print("Building all-splits support corpus for coverage=1...")
    support_to_id: dict[str, int] = {}
    support_texts: list[str] = []
    for sp in ["train", "validation", "test"]:
        for ex in dataset[sp]:
            s = ex["support"]
            if s not in support_to_id:
                support_to_id[s] = len(support_texts)
                support_texts.append(s)
    print(f"Unique supports: {len(support_texts)}")

    rng = np.random.default_rng(seed)
    full_ds = dataset[split]
    idxs = np.sort(rng.choice(len(full_ds), size=min(n_eval, len(full_ds)), replace=False))
    eval_indices = idxs.tolist()
    eval_subset = full_ds.select(eval_indices)
    print(f"Evaluation subset: split={split}, n={len(eval_subset)}")

    print("Loading generator...")
    gen_tok, gen_model = load_generator(generator_name, os_device)

    print("Loading chunk tokenizer...")
    chunk_tokenizer = AutoTokenizer.from_pretrained(chunk_tokenizer_name, use_fast=True)

    per_question_rows: list[dict] = []
    run_rows: list[dict] = []

    # LLM-only baseline
    print("Running LLM-only baseline...")
    baseline_run_id = f"llm_only_{split}_n{len(eval_subset)}_seed{seed}_{run_ts}"
    b_e2e = []
    b_input = []
    b_output = []
    b_correct = 0
    b_bad = 0
    for i, (orig_idx, ex) in enumerate(zip(eval_indices, eval_subset)):
        t0 = time.perf_counter()
        q, choices, gold = make_mcq(ex, seed + i)
        prompt = format_llm_prompt(q, choices)
        pred, in_tok, out_tok = generate_answer(prompt, gen_tok, gen_model, os_device)
        ok = pred == gold
        if ok:
            b_correct += 1
        if pred is None:
            b_bad += 1
        e2e = (time.perf_counter() - t0) * 1000.0
        b_e2e.append(e2e)
        b_input.append(in_tok)
        b_output.append(out_tok)
        per_question_rows.append(
            {
                "run_id": baseline_run_id,
                "question_id": f"{split}_{orig_idx}",
                "gold_option": gold,
                "pred_option": pred,
                "is_correct": bool(ok),
                "gold_doc_id": support_to_id[ex["support"]],
                "retrieved_doc_ids": [],
                "gold_rank": None,
                "retrieval_hit": False,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "e2e_latency_ms": e2e,
                "chunk_size": 0,
                "top_k": 0,
                "embedding_model": "none",
            }
        )

    total = len(eval_subset)
    run_rows.append(
        {
            "run_id": baseline_run_id,
            "chunk_size": 0,
            "top_k": 0,
            "embedding_model": "none",
            "accuracy": b_correct / total if total else 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "invalid_rate": b_bad / total if total else 0.0,
            "retrieval_latency_ms": 0.0,
            "e2e_latency_ms": float(np.mean(b_e2e)) if b_e2e else 0.0,
            "avg_input_tokens": float(np.mean(b_input)) if b_input else 0.0,
            "avg_output_tokens": float(np.mean(b_output)) if b_output else 0.0,
        }
    )

    # RAG matrix
    for embed_short, embed_name in embed_models:
        print(f"Loading embedding model: {embed_name}")
        embed_model = SentenceTransformer(embed_name, device=embed_device)

        q_cache: dict[str, np.ndarray] = {}

        def encode_query(question: str):
            if question in q_cache:
                return q_cache[question]
            v = embed_model.encode(
                [question], convert_to_numpy=True, normalize_embeddings=True
            ).astype(np.float32)
            q_cache[question] = v
            return v

        for chunk_tokens, overlap_tokens in chunk_settings:
            print(f"Building chunk index: embed={embed_short} chunk={chunk_tokens}")
            chunk_docs, chunk_to_support_id = build_chunk_corpus(
                support_texts,
                support_to_id,
                chunk_tokens,
                overlap_tokens,
                chunk_tokenizer,
            )
            doc_emb = embed_model.encode(
                chunk_docs,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            index = faiss.IndexFlatIP(doc_emb.shape[1])
            index.add(doc_emb)

            for k in top_k_list:
                run_id = (
                    f"rag_{embed_short}_chunk{chunk_tokens}_k{k}_"
                    f"{split}_n{len(eval_subset)}_seed{seed}_{run_ts}"
                )
                print(f"Running {run_id}")

                e2e_lat = []
                ret_lat = []
                in_toks = []
                out_toks = []
                correct = 0
                bad = 0
                hits = 0
                rr_sum = 0.0

                for i, (orig_idx, ex) in enumerate(zip(eval_indices, eval_subset)):
                    t0 = time.perf_counter()
                    q, choices, gold = make_mcq(ex, seed + i)
                    gold_id = support_to_id[ex["support"]]

                    qv = encode_query(q)
                    t_ret = time.perf_counter()
                    _, ids = index.search(qv, k)
                    retrieved_ids = ids[0].tolist()
                    ret_ms = (time.perf_counter() - t_ret) * 1000.0
                    ret_lat.append(ret_ms)

                    # chunk-index ids -> support ids
                    retrieved_support_ids = [chunk_to_support_id[j] for j in retrieved_ids]
                    contexts = [chunk_docs[j] for j in retrieved_ids]

                    gold_rank = None
                    for pos, sid in enumerate(retrieved_support_ids):
                        if sid == gold_id:
                            gold_rank = pos + 1
                            break
                    retrieval_hit = gold_rank is not None
                    if retrieval_hit:
                        hits += 1
                        rr_sum += 1.0 / gold_rank

                    prompt = format_rag_prompt(contexts, q, choices)
                    pred, in_tok, out_tok = generate_answer(prompt, gen_tok, gen_model, os_device)

                    is_correct = pred == gold
                    if is_correct:
                        correct += 1
                    if pred is None:
                        bad += 1

                    e2e = (time.perf_counter() - t0) * 1000.0
                    e2e_lat.append(e2e)
                    in_toks.append(in_tok)
                    out_toks.append(out_tok)

                    per_question_rows.append(
                        {
                            "run_id": run_id,
                            "question_id": f"{split}_{orig_idx}",
                            "gold_option": gold,
                            "pred_option": pred,
                            "is_correct": bool(is_correct),
                            "gold_doc_id": int(gold_id),
                            "retrieved_doc_ids": retrieved_support_ids,
                            "gold_rank": gold_rank,
                            "retrieval_hit": retrieval_hit,
                            "input_tokens": in_tok,
                            "output_tokens": out_tok,
                            "e2e_latency_ms": e2e,
                            "chunk_size": int(chunk_tokens),
                            "top_k": int(k),
                            "embedding_model": embed_short,
                        }
                    )

                total = len(eval_subset)
                run_rows.append(
                    {
                        "run_id": run_id,
                        "chunk_size": int(chunk_tokens),
                        "top_k": int(k),
                        "embedding_model": embed_short,
                        "accuracy": correct / total if total else 0.0,
                        "recall_at_k": hits / total if total else 0.0,
                        "mrr": rr_sum / total if total else 0.0,
                        "invalid_rate": bad / total if total else 0.0,
                        "retrieval_latency_ms": float(np.mean(ret_lat)) if ret_lat else 0.0,
                        "e2e_latency_ms": float(np.mean(e2e_lat)) if e2e_lat else 0.0,
                        "avg_input_tokens": float(np.mean(in_toks)) if in_toks else 0.0,
                        "avg_output_tokens": float(np.mean(out_toks)) if out_toks else 0.0,
                    }
                )

    pq_df = pd.DataFrame(per_question_rows)
    run_df = pd.DataFrame(run_rows)

    per_question_path = out_dir / f"per_question_full_ablation_{run_ts}.jsonl"
    run_summary_path = out_dir / f"run_summary_full_ablation_{run_ts}.csv"
    pivot_path = out_dir / f"pivot_accuracy_full_ablation_{run_ts}.csv"

    with per_question_path.open("w", encoding="utf-8") as f:
        for row in pq_df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    run_df.to_csv(run_summary_path, index=False)

    rag_only = run_df[run_df["top_k"] > 0].copy()
    pivot = rag_only.pivot_table(
        index=["embedding_model", "chunk_size"],
        columns=["top_k"],
        values=["accuracy", "recall_at_k", "mrr", "e2e_latency_ms"],
        aggfunc="mean",
    )
    pivot.to_csv(pivot_path)

    print("\nFinished full ablation run.")
    print(f"Saved per-question: {per_question_path}")
    print(f"Saved run-summary: {run_summary_path}")
    print(f"Saved pivot: {pivot_path}")


if __name__ == "__main__":
    main()
