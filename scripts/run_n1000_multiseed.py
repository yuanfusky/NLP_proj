import json
import os
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
from scipy.stats import t
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
        chunks.append(
            tokenizer.decode(sub, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        )
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


def generate_answer(prompt: str, tokenizer, model, device: torch.device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False, num_beams=1)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    pred = extract_option_letter(text)
    return pred, int(inputs["input_ids"].shape[1]), int(out.shape[1])


def ci95_from_seed_values(values: np.ndarray):
    n = len(values)
    mean = float(values.mean())
    if n <= 1:
        return mean, float("nan"), float("nan"), float("nan")
    std = float(values.std(ddof=1))
    sem = std / np.sqrt(n)
    t_crit = float(t.ppf(0.975, df=n - 1))
    lo = mean - t_crit * sem
    hi = mean + t_crit * sem
    return mean, std, lo, hi


def evaluate_run(
    run_id: str,
    setting_name: str,
    mode: str,
    seed: int,
    eval_subset,
    eval_indices,
    support_to_id,
    generator_tok,
    generator_model,
    device,
    index=None,
    chunk_docs=None,
    chunk_to_support_id=None,
    top_k: int = 0,
):
    rows = []
    e2e_ms_list = []
    ret_ms_list = []
    input_toks = []
    output_toks = []
    correct = 0
    bad = 0
    hit = 0
    rr_sum = 0.0

    for i, (orig_idx, ex) in enumerate(zip(eval_indices, eval_subset)):
        t0 = time.perf_counter()
        q, choices, gold = make_mcq(ex, seed + i)
        gold_doc_id = support_to_id[ex["support"]]

        if mode == "rag":
            qv = EMBED_MODEL.encode(
                [q], convert_to_numpy=True, normalize_embeddings=True
            ).astype(np.float32)
            t_ret = time.perf_counter()
            _, ids = index.search(qv, top_k)
            retrieved_idx = ids[0].tolist()
            retrieved_support_ids = [chunk_to_support_id[j] for j in retrieved_idx]
            contexts = [chunk_docs[j] for j in retrieved_idx]
            ret_ms = (time.perf_counter() - t_ret) * 1000.0
            ret_ms_list.append(ret_ms)

            gold_rank = None
            for pos, sid in enumerate(retrieved_support_ids):
                if sid == gold_doc_id:
                    gold_rank = pos + 1
                    break
            retrieval_hit = gold_rank is not None
            if retrieval_hit:
                hit += 1
                rr_sum += 1.0 / gold_rank
            prompt = format_rag_prompt(contexts, q, choices)
        else:
            retrieved_support_ids = []
            contexts = []
            gold_rank = None
            retrieval_hit = False
            prompt = format_llm_prompt(q, choices)

        pred, in_tok, out_tok = generate_answer(prompt, generator_tok, generator_model, device)
        is_correct = pred == gold
        if is_correct:
            correct += 1
        if pred is None:
            bad += 1

        e2e_ms = (time.perf_counter() - t0) * 1000.0
        e2e_ms_list.append(e2e_ms)
        input_toks.append(in_tok)
        output_toks.append(out_tok)

        rows.append(
            {
                # Required per-example schema fields
                "qid": f"validation_{orig_idx}",
                "question": q,
                "options": {
                    "A": choices[0],
                    "B": choices[1],
                    "C": choices[2],
                    "D": choices[3],
                },
                "gold_answer": gold,
                "pred_answer": pred,
                "is_correct": bool(is_correct),
                "retrieved_contexts": contexts,
                "retrieved_doc_ids": retrieved_support_ids,
                "gold_support_doc_id": int(gold_doc_id),
                "gold_support_hit": retrieval_hit,
                "seed": seed,
                "setting_name": setting_name,

                # Backward-compatible fields
                "run_id": run_id,
                "mode": mode,
                "question_id": f"validation_{orig_idx}",
                "gold_option": gold,
                "pred_option": pred,
                "gold_doc_id": int(gold_doc_id),
                "gold_rank": gold_rank,
                "retrieval_hit": retrieval_hit,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "e2e_latency_ms": e2e_ms,
            }
        )

        if (i + 1) % 100 == 0:
            print(f"[{run_id}] {i+1}/{len(eval_subset)}")

    total = len(eval_subset)
    summary = {
        "run_id": run_id,
        "seed": seed,
        "mode": mode,
        "chunk_size": 256 if mode == "rag" else 0,
        "top_k": top_k if mode == "rag" else 0,
        "embedding_model": "all-MiniLM-L6-v2" if mode == "rag" else "none",
        "accuracy": correct / total if total else 0.0,
        "recall_at_k": hit / total if total else 0.0,
        "mrr": rr_sum / total if total else 0.0,
        "invalid_rate": bad / total if total else 0.0,
        "retrieval_latency_ms": float(np.mean(ret_ms_list)) if ret_ms_list else 0.0,
        "e2e_latency_ms": float(np.mean(e2e_ms_list)) if e2e_ms_list else 0.0,
        "avg_input_tokens": float(np.mean(input_toks)) if input_toks else 0.0,
        "avg_output_tokens": float(np.mean(output_toks)) if output_toks else 0.0,
    }
    return rows, summary


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    torch.set_num_threads(1)
    set_seed(0)

    DEVICE = torch.device("cpu")
    BASE_DIR = Path(__file__).resolve().parents[1]
    OUT_DIR = BASE_DIR / "outputs" / "seed_eval_n1000"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    SEEDS = [0, 1, 2]
    N_EVAL = 1000
    TOP_K = 3
    CHUNK_TOKENS = 256
    OVERLAP_TOKENS = 64

    print("Loading SciQ...")
    dataset = load_dataset("sciq")
    val = dataset["validation"]
    eval_indices = list(range(min(N_EVAL, len(val))))
    eval_subset = val.select(eval_indices)
    print(f"Validation size used: {len(eval_subset)}")

    print("Building all-splits support corpus (coverage=1)...")
    support_to_id: dict[str, int] = {}
    support_texts: list[str] = []
    for sp in ["train", "validation", "test"]:
        for ex in dataset[sp]:
            s = ex["support"]
            if s not in support_to_id:
                support_to_id[s] = len(support_texts)
                support_texts.append(s)
    print(f"Unique supports: {len(support_texts)}")

    print("Loading generator model...")
    gen_tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    gen_model.to(DEVICE)
    gen_model.eval()

    print("Loading embedding model...")
    EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    chunk_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", use_fast=True
    )

    print("Building chunk index for best config (all-MiniLM + chunk256 + k3)...")
    chunk_docs, chunk_to_support_id = build_chunk_corpus(
        support_texts, support_to_id, CHUNK_TOKENS, OVERLAP_TOKENS, chunk_tokenizer
    )
    chunk_emb = EMBED_MODEL.encode(
        chunk_docs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    index = faiss.IndexFlatIP(chunk_emb.shape[1])
    index.add(chunk_emb)

    all_rows = []
    all_summaries = []
    for seed in SEEDS:
        run_id_baseline = f"llm_only_val_n{N_EVAL}_seed{seed}_{STAMP}"
        print(f"\nRunning {run_id_baseline}")
        rows, summary = evaluate_run(
            run_id=run_id_baseline,
            setting_name="llm_only",
            mode="llm_only",
            seed=seed,
            eval_subset=eval_subset,
            eval_indices=eval_indices,
            support_to_id=support_to_id,
            generator_tok=gen_tok,
            generator_model=gen_model,
            device=DEVICE,
        )
        all_rows.extend(rows)
        all_summaries.append(summary)

        run_id_rag = f"rag_best_val_n{N_EVAL}_seed{seed}_{STAMP}"
        print(f"\nRunning {run_id_rag}")
        rows, summary = evaluate_run(
            run_id=run_id_rag,
            setting_name="rag_all-MiniLM-L6-v2_chunk256_k3",
            mode="rag",
            seed=seed,
            eval_subset=eval_subset,
            eval_indices=eval_indices,
            support_to_id=support_to_id,
            generator_tok=gen_tok,
            generator_model=gen_model,
            device=DEVICE,
            index=index,
            chunk_docs=chunk_docs,
            chunk_to_support_id=chunk_to_support_id,
            top_k=TOP_K,
        )
        all_rows.extend(rows)
        all_summaries.append(summary)

    per_question_path = OUT_DIR / f"per_question_seed_eval_n1000_{STAMP}.jsonl"
    summary_path = OUT_DIR / f"run_summary_seed_eval_n1000_{STAMP}.csv"
    aggregate_path = OUT_DIR / f"aggregate_seed_eval_n1000_{STAMP}.csv"
    report_path = OUT_DIR / f"aggregate_seed_eval_n1000_{STAMP}.md"

    with per_question_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(summary_path, index=False)

    aggregate_rows = []
    for mode, g in summary_df.groupby("mode"):
        acc_mean, acc_std, acc_lo, acc_hi = ci95_from_seed_values(g["accuracy"].to_numpy())
        inv_mean, inv_std, inv_lo, inv_hi = ci95_from_seed_values(g["invalid_rate"].to_numpy())
        e2e_mean, e2e_std, e2e_lo, e2e_hi = ci95_from_seed_values(g["e2e_latency_ms"].to_numpy())
        row = {
            "mode": mode,
            "n_seeds": int(len(g)),
            "seeds": ",".join(str(s) for s in sorted(g["seed"].tolist())),
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "accuracy_ci95_lo": acc_lo,
            "accuracy_ci95_hi": acc_hi,
            "invalid_rate_mean": inv_mean,
            "invalid_rate_std": inv_std,
            "invalid_rate_ci95_lo": inv_lo,
            "invalid_rate_ci95_hi": inv_hi,
            "e2e_latency_ms_mean": e2e_mean,
            "e2e_latency_ms_std": e2e_std,
            "e2e_latency_ms_ci95_lo": e2e_lo,
            "e2e_latency_ms_ci95_hi": e2e_hi,
        }
        if mode == "rag":
            rec_mean, rec_std, rec_lo, rec_hi = ci95_from_seed_values(g["recall_at_k"].to_numpy())
            mrr_mean, mrr_std, mrr_lo, mrr_hi = ci95_from_seed_values(g["mrr"].to_numpy())
            row.update(
                {
                    "recall_at_k_mean": rec_mean,
                    "recall_at_k_std": rec_std,
                    "recall_at_k_ci95_lo": rec_lo,
                    "recall_at_k_ci95_hi": rec_hi,
                    "mrr_mean": mrr_mean,
                    "mrr_std": mrr_std,
                    "mrr_ci95_lo": mrr_lo,
                    "mrr_ci95_hi": mrr_hi,
                }
            )
        aggregate_rows.append(row)

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("mode")
    aggregate_df.to_csv(aggregate_path, index=False)

    lines = [
        "# Validation n=1000 Multi-seed Summary",
        "",
        f"- Seeds: {SEEDS}",
        f"- Configs: LLM-only baseline and best RAG (all-MiniLM-L6-v2, chunk=256, k=3)",
        "",
        "## Mean ± Std (and 95% CI over seed-level metrics)",
        "",
    ]
    for _, r in aggregate_df.iterrows():
        lines.append(f"### {r['mode']}")
        lines.append(
            f"- Accuracy: {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} "
            f"(95% CI {r['accuracy_ci95_lo']:.4f} to {r['accuracy_ci95_hi']:.4f})"
        )
        lines.append(
            f"- Invalid rate: {r['invalid_rate_mean']:.4f} ± {r['invalid_rate_std']:.4f} "
            f"(95% CI {r['invalid_rate_ci95_lo']:.4f} to {r['invalid_rate_ci95_hi']:.4f})"
        )
        lines.append(
            f"- E2E latency (ms): {r['e2e_latency_ms_mean']:.2f} ± {r['e2e_latency_ms_std']:.2f} "
            f"(95% CI {r['e2e_latency_ms_ci95_lo']:.2f} to {r['e2e_latency_ms_ci95_hi']:.2f})"
        )
        if r["mode"] == "rag":
            lines.append(
                f"- Recall@{TOP_K}: {r['recall_at_k_mean']:.4f} ± {r['recall_at_k_std']:.4f} "
                f"(95% CI {r['recall_at_k_ci95_lo']:.4f} to {r['recall_at_k_ci95_hi']:.4f})"
            )
            lines.append(
                f"- MRR: {r['mrr_mean']:.4f} ± {r['mrr_std']:.4f} "
                f"(95% CI {r['mrr_ci95_lo']:.4f} to {r['mrr_ci95_hi']:.4f})"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nDone.")
    print(f"Saved per-question: {per_question_path}")
    print(f"Saved run-summary: {summary_path}")
    print(f"Saved aggregate CSV: {aggregate_path}")
    print(f"Saved aggregate report: {report_path}")
