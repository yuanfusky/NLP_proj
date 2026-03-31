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


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def contains_answer(sentence: str, answer: str) -> bool:
    return answer.casefold() in sentence.casefold()


def remove_answer_sentence(text: str, answer: str) -> str:
    sents = split_sentences(text)
    kept = [s for s in sents if not contains_answer(s, answer)]
    if kept:
        return " ".join(kept)
    return ""


def only_answer_sentence(text: str, answer: str) -> str:
    sents = split_sentences(text)
    kept = [s for s in sents if contains_answer(s, answer)]
    if kept:
        return " ".join(kept)
    return ""


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


def format_prompt(context_passages: list[str], question: str, choices: list[str]) -> str:
    context = "\n\n".join([f"[Context {i+1}]\n{p}" for i, p in enumerate(context_passages)])
    return (
        "Use the provided context to answer the multiple-choice science question.\n"
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


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    set_seed(0)

    base_dir = Path(__file__).resolve().parents[1]
    out_dir = base_dir / "outputs" / "context_interventions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    seeds = [0, 1, 2]
    n_eval = 1000

    print("Loading SciQ...")
    dataset = load_dataset("sciq")
    eval_split = dataset["validation"]
    eval_indices = list(range(min(n_eval, len(eval_split))))
    eval_subset = eval_split.select(eval_indices)
    print(f"Eval split=validation n={len(eval_subset)}")

    print("Building all-splits support corpus...")
    support_to_id: dict[str, int] = {}
    support_texts: list[str] = []
    for sp in ["train", "validation", "test"]:
        for ex in dataset[sp]:
            s = ex["support"]
            if s not in support_to_id:
                support_to_id[s] = len(support_texts)
                support_texts.append(s)
    print(f"Unique supports: {len(support_texts)}")

    print("Loading generator and embedding...")
    device = torch.device("cpu")
    gen_tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    gen_model.to(device)
    gen_model.eval()

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    chunk_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", use_fast=True
    )

    print("Building reference retriever (chunk=256)...")
    chunk_docs, chunk_to_support_id = build_chunk_corpus(
        support_texts, support_to_id, 256, 64, chunk_tokenizer
    )
    doc_emb = embed_model.encode(
        chunk_docs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)

    # Pre-encode all questions once.
    q_vectors = {}
    for orig_idx, ex in zip(eval_indices, eval_subset):
        q = ex["question"]
        q_vectors[orig_idx] = embed_model.encode(
            [q], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

    settings = [
        "baseline_k3_full_context",
        "A_remove_answer_sentence",
        "B_only_answer_sentence",
        "C_baseline_k1",
        "C_k1_plus_random_distractor",
        "D_shuffle_top3_order",
    ]

    per_rows: list[dict] = []
    summary_rows: list[dict] = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        rng = random.Random(seed)
        metrics = {
            s: {"correct": 0, "invalid": 0, "hit": 0, "lat_ms": [], "n": 0}
            for s in settings
        }

        for i, (orig_idx, ex) in enumerate(zip(eval_indices, eval_subset)):
            question, choices, gold_letter = make_mcq(ex, seed + i)
            gold_answer_text = ex["correct_answer"]
            gold_support_id = support_to_id[ex["support"]]

            # Retrieve top-3 once, reuse for all interventions.
            qv = q_vectors[orig_idx]
            _, ids = index.search(qv, 3)
            top3_chunk_ids = ids[0].tolist()
            top3_contexts = [chunk_docs[j] for j in top3_chunk_ids]
            top3_doc_ids = [chunk_to_support_id[j] for j in top3_chunk_ids]
            top1_context = [top3_contexts[0]]
            top1_doc_ids = [top3_doc_ids[0]]

            # D: random distractor paragraph for C
            distractor = ""
            for _ in range(30):
                cand = support_texts[rng.randrange(len(support_texts))]
                if (
                    cand != top1_context[0]
                    and gold_answer_text.casefold() not in cand.casefold()
                ):
                    distractor = cand
                    break
            if not distractor:
                distractor = support_texts[rng.randrange(len(support_texts))]

            # D: deterministic shuffle from same contexts
            shuffled_contexts = top3_contexts.copy()
            rng_local = random.Random(seed * 10_000 + orig_idx)
            rng_local.shuffle(shuffled_contexts)

            candidates = {
                "baseline_k3_full_context": (top3_contexts, top3_doc_ids),
                "A_remove_answer_sentence": (
                    [remove_answer_sentence(c, gold_answer_text) for c in top3_contexts],
                    top3_doc_ids,
                ),
                "B_only_answer_sentence": (
                    [only_answer_sentence(c, gold_answer_text) for c in top3_contexts],
                    top3_doc_ids,
                ),
                "C_baseline_k1": (top1_context, top1_doc_ids),
                "C_k1_plus_random_distractor": ([top1_context[0], distractor], top1_doc_ids),
                "D_shuffle_top3_order": (shuffled_contexts, top3_doc_ids),
            }

            for setting_name, (contexts, retrieved_doc_ids) in candidates.items():
                t0 = time.perf_counter()
                prompt = format_prompt(contexts, question, choices)
                pred, in_tok, out_tok = generate_answer(prompt, gen_tok, gen_model, device)
                lat_ms = (time.perf_counter() - t0) * 1000.0

                is_correct = pred == gold_letter
                if is_correct:
                    metrics[setting_name]["correct"] += 1
                if pred is None:
                    metrics[setting_name]["invalid"] += 1
                hit = gold_support_id in retrieved_doc_ids
                if hit:
                    metrics[setting_name]["hit"] += 1
                metrics[setting_name]["lat_ms"].append(lat_ms)
                metrics[setting_name]["n"] += 1

                per_rows.append(
                    {
                        "qid": f"validation_{orig_idx}",
                        "question": question,
                        "options": {
                            "A": choices[0],
                            "B": choices[1],
                            "C": choices[2],
                            "D": choices[3],
                        },
                        "gold_answer": gold_letter,
                        "pred_answer": pred,
                        "is_correct": bool(is_correct),
                        "retrieved_contexts": contexts,
                        "retrieved_doc_ids": retrieved_doc_ids,
                        "gold_support_doc_id": int(gold_support_id),
                        "gold_support_hit": bool(hit),
                        "seed": seed,
                        "setting_name": setting_name,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "e2e_latency_ms": lat_ms,
                    }
                )

            if (i + 1) % 100 == 0:
                print(f"[seed={seed}] {i+1}/{len(eval_subset)}")

        for setting_name in settings:
            m = metrics[setting_name]
            n = m["n"]
            summary_rows.append(
                {
                    "seed": seed,
                    "setting_name": setting_name,
                    "accuracy": m["correct"] / n if n else 0.0,
                    "invalid_rate": m["invalid"] / n if n else 0.0,
                    "recall_at_k": m["hit"] / n if n else 0.0,
                    "e2e_latency_ms": float(np.mean(m["lat_ms"])) if m["lat_ms"] else 0.0,
                }
            )

    per_path = out_dir / f"per_question_context_interventions_{stamp}.jsonl"
    summary_path = out_dir / f"run_summary_context_interventions_{stamp}.csv"
    aggregate_path = out_dir / f"aggregate_context_interventions_{stamp}.csv"
    delta_path = out_dir / f"delta_accuracy_context_interventions_{stamp}.csv"
    report_path = out_dir / f"report_context_interventions_{stamp}.md"

    with per_path.open("w", encoding="utf-8") as f:
        for row in per_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    agg_rows = []
    for setting_name, g in summary_df.groupby("setting_name"):
        acc_mean, acc_std, acc_lo, acc_hi = ci95_from_seed_values(g["accuracy"].to_numpy())
        rec_mean, rec_std, rec_lo, rec_hi = ci95_from_seed_values(g["recall_at_k"].to_numpy())
        lat_mean, lat_std, lat_lo, lat_hi = ci95_from_seed_values(g["e2e_latency_ms"].to_numpy())
        agg_rows.append(
            {
                "setting_name": setting_name,
                "n_seeds": len(g),
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "accuracy_ci95_lo": acc_lo,
                "accuracy_ci95_hi": acc_hi,
                "recall_at_k_mean": rec_mean,
                "recall_at_k_std": rec_std,
                "recall_at_k_ci95_lo": rec_lo,
                "recall_at_k_ci95_hi": rec_hi,
                "latency_ms_mean": lat_mean,
                "latency_ms_std": lat_std,
                "latency_ms_ci95_lo": lat_lo,
                "latency_ms_ci95_hi": lat_hi,
            }
        )
    agg_df = pd.DataFrame(agg_rows).sort_values("setting_name")
    agg_df.to_csv(aggregate_path, index=False)

    # Delta accuracy relative to proper baselines.
    piv = summary_df.pivot(index="seed", columns="setting_name", values="accuracy")
    delta_df = pd.DataFrame({"seed": piv.index})
    delta_df["delta_A_vs_baseline_k3"] = (
        piv["A_remove_answer_sentence"] - piv["baseline_k3_full_context"]
    )
    delta_df["delta_B_vs_baseline_k3"] = (
        piv["B_only_answer_sentence"] - piv["baseline_k3_full_context"]
    )
    delta_df["delta_C_noise_vs_baseline_k1"] = (
        piv["C_k1_plus_random_distractor"] - piv["C_baseline_k1"]
    )
    delta_df["delta_D_shuffle_vs_baseline_k3"] = (
        piv["D_shuffle_top3_order"] - piv["baseline_k3_full_context"]
    )
    delta_df.to_csv(delta_path, index=False)

    lines = [
        "# Context Intervention Experiments",
        "",
        "- A: Remove answer sentence",
        "- B: Only answer sentence",
        "- C: Add distractor context",
        "- D: Shuffle context order",
        "",
        "## Accuracy Mean ± Std (95% CI)",
        "",
    ]
    for _, r in agg_df.iterrows():
        lines.append(
            f"- {r['setting_name']}: {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} "
            f"(95% CI {r['accuracy_ci95_lo']:.4f} to {r['accuracy_ci95_hi']:.4f})"
        )
    lines.append("")
    lines.append("## Delta Accuracy by Seed")
    lines.append("")
    lines.append(delta_df.to_string(index=False))
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nDone.")
    print(f"Saved per-question: {per_path}")
    print(f"Saved run-summary: {summary_path}")
    print(f"Saved aggregate: {aggregate_path}")
    print(f"Saved delta: {delta_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
