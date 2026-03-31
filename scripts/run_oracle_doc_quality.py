import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from scripts.sciq_eval_common import (
        ci95_from_seed_values,
        format_prompt,
        generate_answer,
        make_mcq,
        set_seed,
    )
except ModuleNotFoundError:
    from sciq_eval_common import (
        ci95_from_seed_values,
        format_prompt,
        generate_answer,
        make_mcq,
        set_seed,
    )


ORACLE_SETTING = "oracle_gold_support"
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_N_EVAL = 1000
COMPARISON_SETTINGS = [
    ORACLE_SETTING,
    "baseline_k3_full_context",
    "A_remove_answer_sentence",
    "B_only_answer_sentence",
]
SETTING_ROLE = {
    ORACLE_SETTING: "Upper bound (perfect document)",
    "baseline_k3_full_context": "Normal RAG",
    "A_remove_answer_sentence": "Lower bound (corrupted document)",
    "B_only_answer_sentence": "Minimal ideal evidence",
}


def find_latest_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


def load_latest_context_intervention_results(base_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    messages: list[str] = []
    ctx_dir = base_dir / "outputs" / "context_interventions"
    if not ctx_dir.exists():
        messages.append(f"Comparison directory missing: {ctx_dir}")
        return None, None, messages

    aggregate_path = find_latest_file(ctx_dir, "aggregate_context_interventions_*.csv")
    summary_path = find_latest_file(ctx_dir, "run_summary_context_interventions_*.csv")

    if aggregate_path is None:
        messages.append(f"Missing comparison aggregate file in {ctx_dir}")
    else:
        messages.append(f"Detected comparison aggregate: {aggregate_path}")

    if summary_path is None:
        messages.append(f"Missing comparison summary file in {ctx_dir}")
    else:
        messages.append(f"Detected comparison summary: {summary_path}")

    aggregate_df = pd.read_csv(aggregate_path) if aggregate_path else None
    summary_df = pd.read_csv(summary_path) if summary_path else None
    return aggregate_df, summary_df, messages


def build_all_splits_support_corpus(dataset) -> tuple[dict[str, int], list[str]]:
    support_to_id: dict[str, int] = {}
    support_texts: list[str] = []
    for split_name in ["train", "validation", "test"]:
        for ex in dataset[split_name]:
            support = ex["support"]
            if support not in support_to_id:
                support_to_id[support] = len(support_texts)
                support_texts.append(support)
    return support_to_id, support_texts


def aggregate_oracle_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting_name, group in summary_df.groupby("setting_name"):
        acc_mean, acc_std, acc_lo, acc_hi = ci95_from_seed_values(group["accuracy"].to_numpy())
        inv_mean, inv_std, inv_lo, inv_hi = ci95_from_seed_values(group["invalid_rate"].to_numpy())
        lat_mean, lat_std, lat_lo, lat_hi = ci95_from_seed_values(group["e2e_latency_ms"].to_numpy())
        rows.append(
            {
                "setting_name": setting_name,
                "n_seeds": int(len(group)),
                "seeds": ",".join(str(s) for s in sorted(group["seed"].tolist())),
                "n_examples": int(group["n_examples"].iloc[0]),
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "accuracy_ci95_lo": acc_lo,
                "accuracy_ci95_hi": acc_hi,
                "invalid_rate_mean": inv_mean,
                "invalid_rate_std": inv_std,
                "invalid_rate_ci95_lo": inv_lo,
                "invalid_rate_ci95_hi": inv_hi,
                "latency_ms_mean": lat_mean,
                "latency_ms_std": lat_std,
                "latency_ms_ci95_lo": lat_lo,
                "latency_ms_ci95_hi": lat_hi,
            }
        )
    return pd.DataFrame(rows).sort_values("setting_name")


def summarize_context_interventions(
    aggregate_df: pd.DataFrame | None, summary_df: pd.DataFrame | None
) -> dict[str, dict]:
    if aggregate_df is None:
        return {}

    summary_stats: dict[str, dict] = {}
    invalid_means = {}
    if summary_df is not None and {"setting_name", "invalid_rate"}.issubset(summary_df.columns):
        invalid_means = (
            summary_df.groupby("setting_name", as_index=True)["invalid_rate"].mean().to_dict()
        )

    latency_col = "latency_ms_mean" if "latency_ms_mean" in aggregate_df.columns else "e2e_latency_ms_mean"
    for _, row in aggregate_df.iterrows():
        setting_name = row["setting_name"]
        summary_stats[setting_name] = {
            "setting_name": setting_name,
            "accuracy_mean": float(row["accuracy_mean"]),
            "invalid_rate_mean": float(invalid_means.get(setting_name, np.nan)),
            "latency_ms_mean": float(row[latency_col]) if latency_col in row else float("nan"),
            "n_seeds": int(row["n_seeds"]) if "n_seeds" in row else None,
            "source": "context_interventions",
        }
    return summary_stats


def summary_row_to_dict(row: pd.Series) -> dict:
    return {
        "setting_name": row["setting_name"],
        "accuracy_mean": float(row["accuracy_mean"]),
        "invalid_rate_mean": float(row["invalid_rate_mean"]),
        "latency_ms_mean": float(row["latency_ms_mean"]),
        "n_seeds": int(row["n_seeds"]),
        "source": "oracle_doc_quality",
    }


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def make_markdown_table(rows: list[dict]) -> str:
    headers = [
        "setting_name",
        "role",
        "accuracy_mean",
        "delta_vs_baseline",
        "invalid_rate_mean",
        "latency_ms_mean",
    ]
    lines = [
        "| setting_name | role | accuracy_mean | delta_vs_baseline | invalid_rate_mean | latency_ms_mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["setting_name"]),
                    str(row["role"]),
                    format_float(row["accuracy_mean"]),
                    format_float(row["delta_vs_baseline"]),
                    format_float(row["invalid_rate_mean"]),
                    format_float(row["latency_ms_mean"], digits=2),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_report(
    oracle_agg_df: pd.DataFrame,
    comparison_stats: dict[str, dict],
    comparison_messages: list[str],
    n_eval: int,
    seeds: list[int],
    aggregate_path: Path,
    report_path: Path,
) -> tuple[str, dict[str, dict]]:
    combined: dict[str, dict] = {}
    oracle_row = oracle_agg_df.iloc[0]
    combined[ORACLE_SETTING] = summary_row_to_dict(oracle_row)
    for setting_name, stats in comparison_stats.items():
        if setting_name in COMPARISON_SETTINGS and setting_name != ORACLE_SETTING:
            combined[setting_name] = stats

    baseline_acc = combined.get("baseline_k3_full_context", {}).get("accuracy_mean")
    report_rows = []
    for setting_name in COMPARISON_SETTINGS:
        if setting_name not in combined:
            continue
        stats = combined[setting_name]
        delta = None
        if baseline_acc is not None:
            delta = stats["accuracy_mean"] - baseline_acc
        report_rows.append(
            {
                "setting_name": setting_name,
                "role": SETTING_ROLE[setting_name],
                "accuracy_mean": stats["accuracy_mean"],
                "delta_vs_baseline": delta,
                "invalid_rate_mean": stats["invalid_rate_mean"],
                "latency_ms_mean": stats["latency_ms_mean"],
            }
        )

    upper = combined.get(ORACLE_SETTING, {}).get("accuracy_mean")
    normal = combined.get("baseline_k3_full_context", {}).get("accuracy_mean")
    lower = combined.get("A_remove_answer_sentence", {}).get("accuracy_mean")

    interpretation_lines = []
    if upper is None or normal is None or lower is None:
        interpretation_lines.append(
            "The expected ordering could not be fully checked because one or more comparison files were missing."
        )
    else:
        ordering_ok = upper >= normal >= lower
        interpretation_lines.append(
            f"Observed ordering: oracle_gold_support={upper:.4f}, "
            f"baseline_k3_full_context={normal:.4f}, "
            f"A_remove_answer_sentence={lower:.4f}."
        )
        interpretation_lines.append(
            "Expected ordering Upper >= Normal >= Lower is satisfied."
            if ordering_ok
            else "Expected ordering Upper >= Normal >= Lower is not satisfied."
        )

    if "B_only_answer_sentence" in combined:
        interpretation_lines.append(
            "B_only_answer_sentence is a compressed, answer-focused evidence condition rather than a standard oracle paragraph, so it should be interpreted as minimal ideal evidence instead of a full-document oracle."
        )
    else:
        interpretation_lines.append(
            "B_only_answer_sentence was not available from the detected comparison outputs."
        )

    lines = [
        "# Oracle Document Quality Report",
        "",
        "## Summary Table",
        "",
        make_markdown_table(report_rows),
        "",
        "## Interpretation",
        "",
    ]
    for line in interpretation_lines:
        lines.append(f"- {line}")

    return "\n".join(lines), combined


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    set_seed(0)

    base_dir = Path(__file__).resolve().parents[1]
    out_dir = base_dir / "outputs" / "oracle_doc_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    seeds = DEFAULT_SEEDS
    n_eval = DEFAULT_N_EVAL

    print(f"Base directory: {base_dir}")
    print(f"Output directory: {out_dir}")
    print("Loading SciQ...")
    dataset = load_dataset("sciq")
    print(f"Loaded dataset splits: {list(dataset.keys())}")

    eval_split = dataset["validation"]
    eval_indices = list(range(min(n_eval, len(eval_split))))
    eval_subset = eval_split.select(eval_indices)
    print(f"Validation subset size: {len(eval_subset)}")

    print("Building all-splits support corpus for gold-support ids...")
    support_to_id, support_texts = build_all_splits_support_corpus(dataset)
    print(f"Unique supports: {len(support_texts)}")

    print("Loading generator model: google/flan-t5-small")
    device = torch.device("cpu")
    gen_tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    gen_model.to(device)
    gen_model.eval()

    per_rows: list[dict] = []
    summary_rows: list[dict] = []

    for seed in seeds:
        print(f"\n=== Seed {seed} / setting {ORACLE_SETTING} ===")
        correct = 0
        invalid = 0
        latencies_ms: list[float] = []

        for i, (orig_idx, ex) in enumerate(zip(eval_indices, eval_subset)):
            question, choices, gold_letter = make_mcq(ex, seed + i)
            gold_support_id = support_to_id[ex["support"]]
            context_text = ex["support"]

            t0 = time.perf_counter()
            prompt = format_prompt([context_text], question, choices)
            pred, input_tokens, output_tokens = generate_answer(prompt, gen_tok, gen_model, device)
            e2e_latency_ms = (time.perf_counter() - t0) * 1000.0

            is_correct = pred == gold_letter
            if is_correct:
                correct += 1
            if pred is None:
                invalid += 1
            latencies_ms.append(e2e_latency_ms)

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
                    "setting_name": ORACLE_SETTING,
                    "context_type": ORACLE_SETTING,
                    "context_text": context_text,
                    "seed": seed,
                    "retrieved_contexts": [context_text],
                    "retrieved_doc_ids": [],
                    "gold_support_doc_id": int(gold_support_id),
                    "gold_support_hit": True,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "e2e_latency_ms": e2e_latency_ms,
                }
            )

            if (i + 1) % 100 == 0:
                print(f"[seed={seed}] {i + 1}/{len(eval_subset)}")

        n_examples = len(eval_subset)
        accuracy = correct / n_examples if n_examples else 0.0
        invalid_rate = invalid / n_examples if n_examples else 0.0
        mean_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        print(
            f"Seed {seed} complete: n={n_examples}, accuracy={accuracy:.4f}, "
            f"invalid_rate={invalid_rate:.4f}, e2e_latency_ms={mean_latency_ms:.2f}"
        )
        summary_rows.append(
            {
                "setting_name": ORACLE_SETTING,
                "n_examples": n_examples,
                "seed": seed,
                "accuracy": accuracy,
                "invalid_rate": invalid_rate,
                "e2e_latency_ms": mean_latency_ms,
            }
        )

    per_path = out_dir / f"per_question_oracle_doc_quality_{stamp}.jsonl"
    summary_path = out_dir / f"run_summary_oracle_doc_quality_{stamp}.csv"
    aggregate_path = out_dir / f"aggregate_oracle_doc_quality_{stamp}.csv"
    report_path = out_dir / f"report_oracle_doc_quality_{stamp}.md"

    with per_path.open("w", encoding="utf-8") as handle:
        for row in per_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    aggregate_df = aggregate_oracle_summary(summary_df)
    aggregate_df.to_csv(aggregate_path, index=False)

    comparison_aggregate_df, comparison_summary_df, comparison_messages = (
        load_latest_context_intervention_results(base_dir)
    )
    report_text, combined_stats = build_report(
        oracle_agg_df=aggregate_df,
        comparison_stats=summarize_context_interventions(
            comparison_aggregate_df, comparison_summary_df
        ),
        comparison_messages=comparison_messages,
        n_eval=len(eval_subset),
        seeds=seeds,
        aggregate_path=aggregate_path,
        report_path=report_path,
    )
    report_path.write_text(report_text, encoding="utf-8")

    oracle_accuracy = float(aggregate_df.iloc[0]["accuracy_mean"])
    print("\nComparison summary:")
    for setting_name in COMPARISON_SETTINGS:
        stats = combined_stats.get(setting_name)
        if stats is None:
            print(f"  - {setting_name}: missing")
            continue
        print(
            f"  - {setting_name}: accuracy={stats['accuracy_mean']:.4f}, "
            f"invalid_rate={format_float(stats['invalid_rate_mean'])}, "
            f"latency_ms={format_float(stats['latency_ms_mean'], digits=2)}"
        )

    print("\nDone.")
    print(f"Final oracle accuracy: {oracle_accuracy:.4f}")
    print(f"Saved run-summary: {summary_path}")
    print(f"Saved per-question: {per_path}")
    print(f"Saved aggregate: {aggregate_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
