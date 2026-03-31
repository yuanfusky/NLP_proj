import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = rng.integers(0, n, size=(n_boot, n))
    means = values[samples].mean(axis=1)
    lo = float(np.quantile(means, 0.025))
    hi = float(np.quantile(means, 0.975))
    return lo, hi


def bootstrap_delta_ci(
    baseline_values: np.ndarray, best_values: np.ndarray, n_boot: int = 10000, seed: int = 0
):
    rng = np.random.default_rng(seed)
    n = len(baseline_values)
    samples = rng.integers(0, n, size=(n_boot, n))
    delta = best_values[samples].mean(axis=1) - baseline_values[samples].mean(axis=1)
    lo = float(np.quantile(delta, 0.025))
    hi = float(np.quantile(delta, 0.975))
    return lo, hi


def mcnemar_exact(baseline_correct: np.ndarray, best_correct: np.ndarray):
    # b: baseline correct, best wrong
    # c: baseline wrong, best correct
    b = int(np.sum((baseline_correct == 1) & (best_correct == 0)))
    c = int(np.sum((baseline_correct == 0) & (best_correct == 1)))
    n_discordant = b + c
    if n_discordant == 0:
        return {
            "b_baseline_correct_best_wrong": b,
            "c_baseline_wrong_best_correct": c,
            "discordant_total": n_discordant,
            "p_value": 1.0,
        }
    p = float(binomtest(min(b, c), n=n_discordant, p=0.5, alternative="two-sided").pvalue)
    return {
        "b_baseline_correct_best_wrong": b,
        "c_baseline_wrong_best_correct": c,
        "discordant_total": n_discordant,
        "p_value": p,
    }


def main():
    base_dir = Path(__file__).resolve().parents[1]
    out_dir = base_dir / "outputs" / "full_ablation"
    run_candidates = sorted(out_dir.glob("run_summary_full_ablation_*.csv"))
    pq_candidates = sorted(out_dir.glob("per_question_full_ablation_*.jsonl"))
    if not run_candidates or not pq_candidates:
        raise FileNotFoundError("Missing full ablation outputs.")

    run_path = run_candidates[-1]
    pq_path = pq_candidates[-1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_df = pd.read_csv(run_path)
    pq_df = pd.read_json(pq_path, lines=True)

    baseline_rows = run_df[run_df["top_k"] == 0].copy()
    if baseline_rows.empty:
        raise ValueError("No baseline run (top_k == 0) found.")
    baseline_run_id = baseline_rows.iloc[0]["run_id"]

    rag_rows = run_df[run_df["top_k"] > 0].copy()
    if rag_rows.empty:
        raise ValueError("No RAG runs (top_k > 0) found.")
    best_row = rag_rows.sort_values(by="accuracy", ascending=False).iloc[0]
    best_run_id = best_row["run_id"]

    base_df = pq_df[pq_df["run_id"] == baseline_run_id][["question_id", "is_correct"]].copy()
    best_df = pq_df[pq_df["run_id"] == best_run_id][["question_id", "is_correct"]].copy()

    merged = base_df.merge(
        best_df, on="question_id", how="inner", suffixes=("_baseline", "_best")
    ).sort_values("question_id")

    if merged.empty:
        raise ValueError("No overlapping question_id between baseline and best run.")

    baseline_correct = merged["is_correct_baseline"].astype(int).to_numpy()
    best_correct = merged["is_correct_best"].astype(int).to_numpy()

    baseline_acc = float(baseline_correct.mean())
    best_acc = float(best_correct.mean())
    delta_acc = best_acc - baseline_acc

    base_lo, base_hi = bootstrap_ci(baseline_correct, n_boot=10000, seed=0)
    best_lo, best_hi = bootstrap_ci(best_correct, n_boot=10000, seed=1)
    delta_lo, delta_hi = bootstrap_delta_ci(baseline_correct, best_correct, n_boot=10000, seed=2)

    mcnemar = mcnemar_exact(baseline_correct, best_correct)

    result = {
        "inputs": {
            "run_summary_file": str(run_path),
            "per_question_file": str(pq_path),
            "n_paired_questions": int(len(merged)),
            "baseline_run_id": str(baseline_run_id),
            "best_run_id": str(best_run_id),
        },
        "metrics": {
            "baseline_accuracy": baseline_acc,
            "baseline_accuracy_ci95": [base_lo, base_hi],
            "best_accuracy": best_acc,
            "best_accuracy_ci95": [best_lo, best_hi],
            "delta_accuracy_best_minus_baseline": delta_acc,
            "delta_accuracy_ci95": [delta_lo, delta_hi],
            "improvement_rate": float(
                (
                    mcnemar["c_baseline_wrong_best_correct"]
                    - mcnemar["b_baseline_correct_best_wrong"]
                )
                / len(merged)
            ),
        },
        "mcnemar_exact": mcnemar,
    }

    json_out = out_dir / f"significance_baseline_vs_best_{stamp}.json"
    md_out = out_dir / f"significance_baseline_vs_best_{stamp}.md"

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# Significance Test: Baseline vs Best RAG",
        "",
        f"- Baseline run: `{baseline_run_id}`",
        f"- Best run: `{best_run_id}`",
        f"- Paired questions: `{len(merged)}`",
        "",
        "## Accuracy",
        f"- Baseline: `{baseline_acc:.4f}` (95% CI `{base_lo:.4f}` to `{base_hi:.4f}`)",
        f"- Best: `{best_acc:.4f}` (95% CI `{best_lo:.4f}` to `{best_hi:.4f}`)",
        f"- Delta (Best - Baseline): `{delta_acc:.4f}` (95% CI `{delta_lo:.4f}` to `{delta_hi:.4f}`)",
        "",
        "## McNemar (Exact)",
        f"- b (baseline correct, best wrong): `{mcnemar['b_baseline_correct_best_wrong']}`",
        f"- c (baseline wrong, best correct): `{mcnemar['c_baseline_wrong_best_correct']}`",
        f"- Discordant total: `{mcnemar['discordant_total']}`",
        f"- p-value: `{mcnemar['p_value']:.6g}`",
    ]
    md_out.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved JSON: {json_out}")
    print(f"Saved Markdown: {md_out}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
