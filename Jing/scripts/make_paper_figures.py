import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def candidate_files(base_dir: Path, pattern: str) -> list[Path]:
    return sorted(base_dir.glob(pattern))


def print_candidates(title: str, files: list[Path]) -> None:
    print(f"{title}:")
    if not files:
        print("  - none")
        return
    for path in files:
        print(f"  - {path}")


def pick_latest(base_dir: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        files = candidate_files(base_dir, pattern)
        if files:
            return files[-1]
    return None


def parse_seed_eval_markdown(path: Path) -> pd.DataFrame:
    # Fallback parser in case only the markdown aggregate is available.
    rows: list[dict] = []
    current_mode = None
    mode_re = re.compile(r"^###\s+(.+?)\s*$")
    acc_re = re.compile(r"^- Accuracy:\s+([0-9.]+)\s+±\s+([0-9.]+)")
    inv_re = re.compile(r"^- Invalid rate:\s+([0-9.]+)\s+±\s+([0-9.]+)")
    lat_re = re.compile(r"^- E2E latency \(ms\):\s+([0-9.]+)\s+±\s+([0-9.]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        mode_match = mode_re.match(line)
        if mode_match:
            current_mode = mode_match.group(1).strip()
            rows.append({"mode": current_mode})
            continue
        if not rows or current_mode is None:
            continue
        acc_match = acc_re.match(line)
        inv_match = inv_re.match(line)
        lat_match = lat_re.match(line)
        if acc_match:
            rows[-1]["accuracy_mean"] = float(acc_match.group(1))
            rows[-1]["accuracy_std"] = float(acc_match.group(2))
        elif inv_match:
            rows[-1]["invalid_rate_mean"] = float(inv_match.group(1))
            rows[-1]["invalid_rate_std"] = float(inv_match.group(2))
        elif lat_match:
            rows[-1]["e2e_latency_ms_mean"] = float(lat_match.group(1))
            rows[-1]["e2e_latency_ms_std"] = float(lat_match.group(2))
    return pd.DataFrame(rows)


def load_seed_eval(path: Path | None) -> tuple[dict[str, dict], list[str]]:
    if path is None:
        return {}, ["Missing seed-eval aggregate file."]
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".md":
        df = parse_seed_eval_markdown(path)
    else:
        return {}, [f"Unsupported seed-eval file type: {path}"]

    records: dict[str, dict] = {}
    for _, row in df.iterrows():
        key = str(row.get("mode", "")).strip()
        if not key:
            continue
        records[key] = {
            "name": key,
            "accuracy": get_value(row, ["accuracy_mean", "accuracy"]),
            "error": get_error(row, "accuracy"),
            "invalid_rate": get_value(row, ["invalid_rate_mean", "invalid_rate"]),
            "latency_ms": get_value(row, ["e2e_latency_ms_mean", "latency_ms_mean", "e2e_latency_ms"]),
            "source": str(path),
        }
    return records, [f"Using seed-eval file: {path}"]


def aggregate_summary(path: Path, key_col: str) -> pd.DataFrame:
    # Rebuild simple aggregate statistics from seed-level run summaries when needed.
    df = pd.read_csv(path)
    grouped = df.groupby(key_col, as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        invalid_rate_mean=("invalid_rate", "mean"),
        invalid_rate_std=("invalid_rate", "std"),
        latency_ms_mean=("e2e_latency_ms", "mean"),
        latency_ms_std=("e2e_latency_ms", "std"),
    )
    return grouped.fillna(0.0)


def load_setting_table(
    aggregate_path: Path | None,
    summary_path: Path | None,
    key_candidates: list[str],
) -> tuple[dict[str, dict], list[str]]:
    messages: list[str] = []
    df = None
    used_path = None
    if aggregate_path is not None:
        df = pd.read_csv(aggregate_path)
        used_path = aggregate_path
    elif summary_path is not None:
        key_col = next((col for col in key_candidates if col in pd.read_csv(summary_path, nrows=1).columns), None)
        if key_col is None:
            return {}, [f"Could not find key column in summary file: {summary_path}"]
        df = aggregate_summary(summary_path, key_col)
        used_path = summary_path
    else:
        return {}, ["No aggregate or summary file found."]

    key_col = next((col for col in key_candidates if col in df.columns), None)
    if key_col is None:
        return {}, [f"Could not find any key column in {used_path}"]

    records: dict[str, dict] = {}
    for _, row in df.iterrows():
        key = str(row[key_col]).strip()
        if not key:
            continue
        records[key] = {
            "name": key,
            "accuracy": get_value(row, ["accuracy_mean", "accuracy"]),
            "error": get_error(row, "accuracy"),
            "invalid_rate": get_value(row, ["invalid_rate_mean", "invalid_rate"]),
            "latency_ms": get_value(
                row,
                ["latency_ms_mean", "e2e_latency_ms_mean", "e2e_latency_ms", "latency_ms"],
            ),
            "source": str(used_path),
        }
    messages.append(f"Using setting table: {used_path}")
    return records, messages


def get_value(row: pd.Series, names: list[str]) -> float | None:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return None


def get_error(row: pd.Series, prefix: str) -> float | None:
    std_name = f"{prefix}_std"
    lo_name = f"{prefix}_ci95_lo"
    hi_name = f"{prefix}_ci95_hi"
    mean_name = f"{prefix}_mean"
    raw_name = prefix
    if std_name in row and pd.notna(row[std_name]):
        return float(row[std_name])
    mean = None
    if mean_name in row and pd.notna(row[mean_name]):
        mean = float(row[mean_name])
    elif raw_name in row and pd.notna(row[raw_name]):
        mean = float(row[raw_name])
    if mean is not None and lo_name in row and hi_name in row and pd.notna(row[lo_name]) and pd.notna(row[hi_name]):
        return max(mean - float(row[lo_name]), float(row[hi_name]) - mean)
    return None


def make_output_dir(base_dir: Path) -> Path:
    out_dir = base_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_figure(fig, png_path: Path, pdf_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def format_num(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.4f}"


def build_figure1(settings: list[dict], out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = list(range(len(settings)))
    heights = [item["value"] for item in settings]
    yerr = [item["error"] if item["error"] is not None else 0.0 for item in settings]
    ax.bar(x, heights, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([item["label"] for item in settings], rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Core Settings Accuracy")
    png_path = out_dir / "figure1_core_settings_accuracy.png"
    pdf_path = out_dir / "figure1_core_settings_accuracy.pdf"
    save_figure(fig, png_path, pdf_path)
    return [png_path, pdf_path]


def build_figure2(settings: list[dict], out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = list(range(len(settings)))
    heights = [item["value"] for item in settings]
    yerr = [item["error"] if item["error"] is not None else 0.0 for item in settings]
    ax.bar(x, heights, yerr=yerr, capsize=4)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([item["label"] for item in settings], rotation=20, ha="right")
    ax.set_ylabel("Delta Accuracy vs Baseline")
    min_val = min(heights + [0.0])
    max_val = max(heights + [0.0])
    pad = max(0.02, 0.12 * max(abs(min_val), abs(max_val), 0.05))
    ax.set_ylim(min_val - pad, max_val + pad)
    ax.set_title("Intervention Delta Accuracy")
    png_path = out_dir / "figure2_delta_accuracy.png"
    pdf_path = out_dir / "figure2_delta_accuracy.pdf"
    save_figure(fig, png_path, pdf_path)
    return [png_path, pdf_path]


def build_figure3(settings: list[dict], out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    x = list(range(len(settings)))
    y = [item["value"] for item in settings]
    ax.plot(x, y, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels([item["label"] for item in settings], rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Document Quality Ladder")
    for idx, item in enumerate(settings):
        ax.annotate(f"{item['value']:.4f}", (x[idx], y[idx]), textcoords="offset points", xytext=(0, 6), ha="center")
    png_path = out_dir / "figure3_document_quality_ladder.png"
    pdf_path = out_dir / "figure3_document_quality_ladder.pdf"
    save_figure(fig, png_path, pdf_path)
    return [png_path, pdf_path]


def make_report(
    out_dir: Path,
    used_sources: list[str],
    figure1_settings: list[dict],
    figure2_settings: list[dict],
    figure3_settings: list[dict],
    oracle_used: bool,
) -> Path:
    lines = [
        "# Figures Report",
        "",
        "## Source Files Used",
        "",
    ]
    for source in used_sources:
        lines.append(f"- `{source}`")
    lines.extend(
        [
            "",
            "## Settings Found",
            "",
        ]
    )
    found_names = sorted({item["raw_name"] for item in figure1_settings + figure2_settings + figure3_settings})
    for name in found_names:
        lines.append(f"- `{name}`")

    lines.extend(
        [
            "",
            "## Figure 1 Values",
            "",
            "| label | raw_setting | accuracy | error_bar |",
            "|---|---|---:|---:|",
        ]
    )
    for item in figure1_settings:
        lines.append(
            f"| {item['label']} | `{item['raw_name']}` | {format_num(item['value'])} | {format_num(item['error'])} |"
        )

    lines.extend(
        [
            "",
            "## Figure 2 Values",
            "",
            "| label | raw_setting | delta_vs_baseline | error_bar |",
            "|---|---|---:|---:|",
        ]
    )
    for item in figure2_settings:
        lines.append(
            f"| {item['label']} | `{item['raw_name']}` | {format_num(item['value'])} | {format_num(item['error'])} |"
        )

    lines.extend(
        [
            "",
            "## Figure 3 Values",
            "",
            "| label | raw_setting | accuracy |",
            "|---|---|---:|",
        ]
    )
    for item in figure3_settings:
        lines.append(f"| {item['label']} | `{item['raw_name']}` | {format_num(item['value'])} |")

    lines.extend(
        [
            "",
            "## Oracle Handling",
            "",
            "- Oracle gold support was found and used for the ladder figure."
            if oracle_used
            else "- Oracle gold support was not found; the ladder figure used `B_only_answer_sentence` as the fallback labeled `Idealized clean evidence`.",
        ]
    )
    report_path = out_dir / "figures_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def append_setting(
    target: list[dict],
    label: str,
    raw_name: str,
    record: dict | None,
    value_override: float | None = None,
    error_override: float | None = None,
) -> None:
    if record is None:
        return
    value = record["accuracy"] if value_override is None else value_override
    error = record["error"] if error_override is None else error_override
    if value is None:
        return
    target.append(
        {
            "label": label,
            "raw_name": raw_name,
            "value": float(value),
            "error": None if error is None else float(error),
            "source": record["source"],
        }
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    out_dir = make_output_dir(base_dir)

    # Detect all candidate files first so the script is explicit about what it found.
    seed_csv = candidate_files(base_dir, "outputs/seed_eval_n1000/aggregate_seed_eval_n1000_*.csv")
    seed_md = candidate_files(base_dir, "outputs/seed_eval_n1000/aggregate_seed_eval_n1000_*.md")
    context_agg = candidate_files(base_dir, "outputs/context_interventions/aggregate_context_interventions_*.csv")
    context_delta = candidate_files(base_dir, "outputs/context_interventions/delta_accuracy_context_interventions_*.csv")
    context_summary = candidate_files(base_dir, "outputs/context_interventions/run_summary_context_interventions_*.csv")
    oracle_agg = candidate_files(base_dir, "outputs/oracle_doc_quality/aggregate_oracle_doc_quality_*.csv")
    oracle_summary = candidate_files(base_dir, "outputs/oracle_doc_quality/run_summary_oracle_doc_quality_*.csv")

    print_candidates("Detected aggregate_seed_eval_n1000 CSV", seed_csv)
    print_candidates("Detected aggregate_seed_eval_n1000 MD", seed_md)
    print_candidates("Detected aggregate_context_interventions CSV", context_agg)
    print_candidates("Detected delta_accuracy_context_interventions CSV", context_delta)
    print_candidates("Detected run_summary_context_interventions CSV", context_summary)
    print_candidates("Detected aggregate_oracle_doc_quality CSV", oracle_agg)
    print_candidates("Detected run_summary_oracle_doc_quality CSV", oracle_summary)

    seed_path = pick_latest(
        base_dir,
        [
            "outputs/seed_eval_n1000/aggregate_seed_eval_n1000_*.csv",
            "outputs/seed_eval_n1000/aggregate_seed_eval_n1000_*.md",
        ],
    )
    context_agg_path = pick_latest(
        base_dir,
        ["outputs/context_interventions/aggregate_context_interventions_*.csv"],
    )
    context_summary_path = pick_latest(
        base_dir,
        ["outputs/context_interventions/run_summary_context_interventions_*.csv"],
    )
    oracle_agg_path = pick_latest(
        base_dir,
        ["outputs/oracle_doc_quality/aggregate_oracle_doc_quality_*.csv"],
    )
    oracle_summary_path = pick_latest(
        base_dir,
        ["outputs/oracle_doc_quality/run_summary_oracle_doc_quality_*.csv"],
    )

    used_messages: list[str] = []
    seed_records, messages = load_seed_eval(seed_path)
    used_messages.extend(messages)
    context_records, messages = load_setting_table(
        context_agg_path,
        context_summary_path,
        ["setting_name", "mode"],
    )
    used_messages.extend(messages)
    oracle_records, messages = load_setting_table(
        oracle_agg_path,
        oracle_summary_path,
        ["setting_name", "mode"],
    )
    used_messages.extend(messages)
    for message in used_messages:
        print(message)

    used_sources = sorted(
        {
            record["source"]
            for record in list(seed_records.values()) + list(context_records.values()) + list(oracle_records.values())
        }
    )

    # Prefer the named intervention baseline when available; otherwise fall back to the best RAG aggregate.
    baseline_record = context_records.get("baseline_k3_full_context")
    baseline_label = "RAG baseline"
    baseline_raw_name = "baseline_k3_full_context"
    if baseline_record is None:
        baseline_record = seed_records.get("rag")
        baseline_label = "RAG baseline (best equivalent)"
        baseline_raw_name = "rag"

    figure1_settings: list[dict] = []
    append_setting(figure1_settings, "LLM-only", "llm_only", seed_records.get("llm_only"))
    append_setting(figure1_settings, baseline_label, baseline_raw_name, baseline_record)
    append_setting(figure1_settings, "Oracle gold support", "oracle_gold_support", oracle_records.get("oracle_gold_support"))
    append_setting(
        figure1_settings,
        "Remove answer sentence",
        "A_remove_answer_sentence",
        context_records.get("A_remove_answer_sentence"),
    )
    append_setting(
        figure1_settings,
        "Only answer sentence",
        "B_only_answer_sentence",
        context_records.get("B_only_answer_sentence"),
    )
    append_setting(
        figure1_settings,
        "Add distractor context",
        "C_k1_plus_random_distractor",
        context_records.get("C_k1_plus_random_distractor"),
    )
    append_setting(
        figure1_settings,
        "Shuffle context order",
        "D_shuffle_top3_order",
        context_records.get("D_shuffle_top3_order"),
    )

    figure2_settings: list[dict] = []
    baseline_accuracy = baseline_record["accuracy"] if baseline_record is not None else None
    baseline_error = baseline_record["error"] if baseline_record is not None else None
    if baseline_accuracy is not None:
        delta_map = [
            ("Remove answer sentence", "A_remove_answer_sentence"),
            ("Only answer sentence", "B_only_answer_sentence"),
            ("Add distractor context", "C_k1_plus_random_distractor"),
            ("Shuffle context order", "D_shuffle_top3_order"),
            ("Oracle gold support", "oracle_gold_support"),
        ]
        for label, raw_name in delta_map:
            record = context_records.get(raw_name) if raw_name != "oracle_gold_support" else oracle_records.get(raw_name)
            if record is None or record["accuracy"] is None:
                continue
            delta = record["accuracy"] - baseline_accuracy
            error = None
            if record["error"] is not None and baseline_error is not None:
                error = math.sqrt(record["error"] ** 2 + baseline_error ** 2)
            append_setting(
                figure2_settings,
                label,
                raw_name,
                record,
                value_override=delta,
                error_override=error,
            )
    else:
        print("Warning: baseline setting not found; skipping delta figure values.")

    oracle_record = oracle_records.get("oracle_gold_support")
    oracle_used = oracle_record is not None and oracle_record.get("accuracy") is not None
    perfect_label = "Perfect document"
    perfect_raw_name = "oracle_gold_support"
    perfect_record = oracle_record
    if perfect_record is None:
        perfect_label = "Idealized clean evidence"
        perfect_raw_name = "B_only_answer_sentence"
        perfect_record = context_records.get("B_only_answer_sentence")

    figure3_settings: list[dict] = []
    append_setting(figure3_settings, perfect_label, perfect_raw_name, perfect_record)
    append_setting(figure3_settings, "Normal document", baseline_raw_name, baseline_record)
    append_setting(
        figure3_settings,
        "Corrupted document",
        "A_remove_answer_sentence",
        context_records.get("A_remove_answer_sentence"),
    )

    generated_paths: list[Path] = []
    if figure1_settings:
        generated_paths.extend(build_figure1(figure1_settings, out_dir))
    else:
        print("Warning: no values available for Figure 1.")
    if figure2_settings:
        generated_paths.extend(build_figure2(figure2_settings, out_dir))
    else:
        print("Warning: no values available for Figure 2.")
    if figure3_settings:
        generated_paths.extend(build_figure3(figure3_settings, out_dir))
    else:
        print("Warning: no values available for Figure 3.")

    report_path = make_report(
        out_dir=out_dir,
        used_sources=used_sources,
        figure1_settings=figure1_settings,
        figure2_settings=figure2_settings,
        figure3_settings=figure3_settings,
        oracle_used=oracle_used,
    )
    generated_paths.append(report_path)

    print("\nGenerated files:")
    for path in generated_paths:
        print(f"  - {path}")

    print("\nFigure 1 values:")
    for item in figure1_settings:
        print(f"  - {item['label']}: {item['value']:.4f}")

    print("\nFigure 2 values:")
    for item in figure2_settings:
        print(f"  - {item['label']}: {item['value']:.4f}")

    print("\nFigure 3 values:")
    for item in figure3_settings:
        print(f"  - {item['label']}: {item['value']:.4f}")


if __name__ == "__main__":
    main()
