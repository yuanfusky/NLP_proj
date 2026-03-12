import csv
import json
from collections import defaultdict
from pathlib import Path


REQUIRED_SETTINGS = {
    "case1": ("baseline_k3_full_context", "A_remove_answer_sentence"),
    "case2": ("baseline_k3_full_context", "B_only_answer_sentence"),
    "case3": ("C_baseline_k1", "C_k1_plus_random_distractor"),
}

CASE_TITLES = {
    "case1": "Case 1: baseline correct -> remove-answer wrong",
    "case2": "Case 2: baseline correct -> only-answer still correct",
    "case3": "Case 3: baseline correct -> distractor-added wrong",
}

CASE_LABELS = {
    "case1": "Case 1",
    "case2": "Case 2",
    "case3": "Case 3",
}


def find_latest_jsonl(base_dir: Path) -> Path | None:
    matches = sorted(base_dir.glob("outputs/context_interventions/per_question_context_interventions_*.jsonl"))
    return matches[-1] if matches else None


def detect_key(row: dict, candidates: list[str]) -> str | None:
    for key in candidates:
        if key in row:
            return key
    return None


def load_rows(path: Path) -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    key_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            rows.append(row)
            if idx == 0:
                key_map["qid"] = detect_key(row, ["qid", "question_id", "id"]) or "qid"
                key_map["setting"] = detect_key(row, ["setting_name", "setting", "mode"]) or "setting_name"
                key_map["question"] = detect_key(row, ["question", "query"]) or "question"
                key_map["gold"] = detect_key(row, ["gold_answer", "gold_option", "answer"]) or "gold_answer"
                key_map["pred"] = detect_key(row, ["pred_answer", "pred_option", "prediction"]) or "pred_answer"
                key_map["correct"] = detect_key(row, ["is_correct", "correct"]) or "is_correct"
                key_map["options"] = detect_key(row, ["options", "choices"]) or "options"
                key_map["contexts"] = detect_key(row, ["retrieved_contexts", "contexts", "context"]) or "retrieved_contexts"
                key_map["seed"] = detect_key(row, ["seed", "random_seed"]) or "seed"
    return rows, key_map


def nonempty_contexts(row: dict, key_map: dict[str, str]) -> list[str]:
    value = row.get(key_map["contexts"], [])
    if isinstance(value, str):
        parts = [value]
    elif isinstance(value, list):
        parts = [str(item) for item in value]
    else:
        parts = []
    return [part.strip() for part in parts if str(part).strip()]


def context_preview(row: dict, key_map: dict[str, str], max_chars: int = 220) -> str:
    parts = nonempty_contexts(row, key_map)
    if not parts:
        return "No context available."
    if len(parts) == 1:
        return truncate_text(parts[0], max_chars)
    first = truncate_text(parts[0], max_chars // 2 + 10)
    second = truncate_text(parts[1], max_chars // 2 - 10)
    return f"{first} || {second}"


def options_preview(row: dict, key_map: dict[str, str]) -> str:
    options = row.get(key_map["options"])
    if not isinstance(options, dict) or not options:
        return "Options unavailable."
    pieces = []
    for label in ["A", "B", "C", "D"]:
        if label in options:
            pieces.append(f"{label}) {options[label]}")
    return " | ".join(pieces) if pieces else "Options unavailable."


def gold_option_text(row: dict, key_map: dict[str, str]) -> str:
    options = row.get(key_map["options"])
    gold = str(row.get(key_map["gold"], "")).strip()
    if isinstance(options, dict) and gold in options:
        return str(options[gold]).strip().lower()
    return ""


def looks_like_numbered_fragment(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    head = stripped[:8]
    return head[0].isdigit()


def truncate_text(text: str, limit: int) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3].rstrip() + "..."


def bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def bundle_score(
    baseline: dict,
    intervention: dict,
    case_name: str,
    key_map: dict[str, str],
    baseline_k3: dict | None = None,
) -> float:
    question = str(baseline.get(key_map["question"], "")).strip()
    baseline_contexts = nonempty_contexts(baseline, key_map)
    intervention_contexts = nonempty_contexts(intervention, key_map)
    baseline_pred = str(baseline.get(key_map["pred"], "")).strip()
    intervention_pred = str(intervention.get(key_map["pred"], "")).strip()
    gold_text = gold_option_text(baseline, key_map)
    baseline_blob = " || ".join(baseline_contexts[:2]).lower()
    intervention_blob = " || ".join(intervention_contexts[:2]).lower()

    score = 0.0
    score += 20.0 if question else 0.0
    score += 10.0 if isinstance(baseline.get(key_map["options"]), dict) else 0.0
    score += 12.0 if baseline_contexts else 0.0
    score += 8.0 if intervention_contexts else 0.0
    score += 5.0 if baseline_contexts and len(baseline_contexts[0]) >= 40 else 0.0
    score += 4.0 if intervention_contexts and len(intervention_contexts[0]) >= 25 else 0.0
    score += min(12.0, len(question) / 12.0)
    score += 10.0 if gold_text and gold_text in baseline_blob else 0.0
    score -= 6.0 if baseline_contexts and looks_like_numbered_fragment(baseline_contexts[0]) else 0.0
    score -= 4.0 if intervention_contexts and looks_like_numbered_fragment(intervention_contexts[0]) else 0.0

    if case_name == "case1":
        # Prefer a clean prediction flip with some remaining context after removal.
        score += 10.0 if baseline_pred and intervention_pred and baseline_pred != intervention_pred else 0.0
        score += 8.0 if intervention_contexts else -4.0
        score += 8.0 if gold_text and gold_text not in intervention_blob else 0.0
    elif case_name == "case2":
        # Prefer preserved prediction under compressed evidence.
        score += 12.0 if baseline_pred and baseline_pred == intervention_pred else 0.0
        score += 8.0 if len(intervention_contexts) == 1 else 0.0
        score += 8.0 if gold_text and gold_text in intervention_blob else 0.0
    elif case_name == "case3":
        # Prefer a clean failure after adding distractor, and note whether k=3 still works.
        score += 10.0 if baseline_pred and intervention_pred and baseline_pred != intervention_pred else 0.0
        score += 6.0 if len(intervention_contexts) >= 2 else 0.0
        if baseline_k3 is not None:
            score += 4.0 if bool_value(baseline_k3.get(key_map["correct"])) else 0.0

    # Prefer short, paper-friendly examples without heavily penalizing informative context.
    total_chars = len(question) + sum(len(x) for x in baseline_contexts[:2]) + sum(len(x) for x in intervention_contexts[:2])
    score -= min(18.0, total_chars / 160.0)
    return score


def build_candidates(rows: list[dict], key_map: dict[str, str]) -> tuple[dict[str, list[dict]], list[str]]:
    grouped: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    settings_found = set()
    for row in rows:
        qid = str(row.get(key_map["qid"], "")).strip()
        setting = str(row.get(key_map["setting"], "")).strip()
        seed = int(row.get(key_map["seed"], 0))
        if not qid or not setting:
            continue
        grouped[qid][seed][setting] = row
        settings_found.add(setting)

    candidates: dict[str, list[dict]] = {"case1": [], "case2": [], "case3": []}
    for qid, seed_map in grouped.items():
        for seed, bundle in seed_map.items():
            baseline_1, intervention_1 = REQUIRED_SETTINGS["case1"]
            b1 = bundle.get(baseline_1)
            i1 = bundle.get(intervention_1)
            if b1 and i1 and bool_value(b1.get(key_map["correct"])) and not bool_value(i1.get(key_map["correct"])):
                candidates["case1"].append(
                    {
                        "qid": qid,
                        "seed": seed,
                        "baseline_setting": baseline_1,
                        "intervention_setting": intervention_1,
                        "baseline_row": b1,
                        "intervention_row": i1,
                        "baseline_k3_row": bundle.get("baseline_k3_full_context"),
                        "score": bundle_score(b1, i1, "case1", key_map),
                    }
                )

            baseline_2, intervention_2 = REQUIRED_SETTINGS["case2"]
            b2 = bundle.get(baseline_2)
            i2 = bundle.get(intervention_2)
            if b2 and i2 and bool_value(b2.get(key_map["correct"])) and bool_value(i2.get(key_map["correct"])):
                candidates["case2"].append(
                    {
                        "qid": qid,
                        "seed": seed,
                        "baseline_setting": baseline_2,
                        "intervention_setting": intervention_2,
                        "baseline_row": b2,
                        "intervention_row": i2,
                        "baseline_k3_row": bundle.get("baseline_k3_full_context"),
                        "score": bundle_score(b2, i2, "case2", key_map),
                    }
                )

            baseline_3, intervention_3 = REQUIRED_SETTINGS["case3"]
            b3 = bundle.get(baseline_3)
            i3 = bundle.get(intervention_3)
            if b3 and i3 and bool_value(b3.get(key_map["correct"])) and not bool_value(i3.get(key_map["correct"])):
                baseline_k3 = bundle.get("baseline_k3_full_context")
                candidates["case3"].append(
                    {
                        "qid": qid,
                        "seed": seed,
                        "baseline_setting": baseline_3,
                        "intervention_setting": intervention_3,
                        "baseline_row": b3,
                        "intervention_row": i3,
                        "baseline_k3_row": baseline_k3,
                        "score": bundle_score(b3, i3, "case3", key_map, baseline_k3=baseline_k3),
                    }
                )

    for case_name in candidates:
        candidates[case_name].sort(key=lambda item: (-item["score"], item["qid"], item["seed"]))
    return candidates, sorted(settings_found)


def choose_examples(candidates: dict[str, list[dict]]) -> dict[str, dict | None]:
    selected: dict[str, dict | None] = {}
    used_qids: set[str] = set()
    # Pick the rarest / most constrained case first so it is not crowded out by easier categories.
    for case_name in ["case3", "case1", "case2"]:
        picked = None
        for candidate in candidates[case_name]:
            if candidate["qid"] not in used_qids:
                picked = candidate
                break
        if picked is None and candidates[case_name]:
            picked = candidates[case_name][0]
        selected[case_name] = picked
        if picked is not None:
            used_qids.add(picked["qid"])
    return selected


def case_note(case_name: str, candidate: dict, key_map: dict[str, str]) -> str:
    if case_name == "case1":
        return "Removing the answer-bearing sentence removes critical evidence and flips the model away from the correct answer."
    if case_name == "case2":
        return "The answer-bearing sentence alone is sufficient for the model to keep the correct prediction."
    baseline_k3 = candidate.get("baseline_k3_row")
    if baseline_k3 is not None:
        k3_correct = bool_value(baseline_k3.get(key_map["correct"]))
        suffix = " The same question stays correct under baseline_k3_full_context." if k3_correct else " The same question is not correct under baseline_k3_full_context."
    else:
        suffix = ""
    return "Adding irrelevant context distracts the model and breaks a previously correct k=1 prediction." + suffix


def case_interpretation(case_name: str, candidate: dict, key_map: dict[str, str]) -> str:
    baseline_row = candidate["baseline_row"]
    intervention_row = candidate["intervention_row"]
    baseline_pred = baseline_row.get(key_map["pred"])
    intervention_pred = intervention_row.get(key_map["pred"])
    gold = baseline_row.get(key_map["gold"])
    if case_name == "case1":
        return (
            f"Under {candidate['baseline_setting']}, the model predicts {baseline_pred}, which matches the gold answer {gold}. "
            f"After removing the answer-bearing sentence, the prediction changes to {intervention_pred}, suggesting that the removed sentence carried the decisive evidence."
        )
    if case_name == "case2":
        return (
            f"The model predicts {baseline_pred} in both {candidate['baseline_setting']} and {candidate['intervention_setting']}. "
            f"This indicates that the answer-bearing sentence alone is enough to preserve the correct decision on this example."
        )
    baseline_k3 = candidate.get("baseline_k3_row")
    baseline_k3_text = ""
    if baseline_k3 is not None:
        status = "correct" if bool_value(baseline_k3.get(key_map["correct"])) else "incorrect"
        baseline_k3_text = f" The corresponding baseline_k3_full_context row is {status}."
    return (
        f"The model is correct with {candidate['baseline_setting']} but changes to {intervention_pred} after adding a random distractor paragraph. "
        f"This example illustrates that irrelevant extra context can interfere with an otherwise sufficient evidence signal.{baseline_k3_text}"
    )


def csv_row(case_name: str, candidate: dict | None, key_map: dict[str, str]) -> dict:
    if candidate is None:
        return {
            "category": CASE_LABELS[case_name],
            "qid": "No suitable example found",
            "question": "",
            "gold_answer": "",
            "baseline_setting": REQUIRED_SETTINGS[case_name][0],
            "baseline_pred": "",
            "baseline_correct": "",
            "intervention_setting": REQUIRED_SETTINGS[case_name][1],
            "intervention_pred": "",
            "intervention_correct": "",
            "short_note": "No suitable example found.",
        }
    baseline_row = candidate["baseline_row"]
    intervention_row = candidate["intervention_row"]
    return {
        "category": CASE_LABELS[case_name],
        "qid": candidate["qid"],
        "question": baseline_row.get(key_map["question"], ""),
        "gold_answer": baseline_row.get(key_map["gold"], ""),
        "baseline_setting": candidate["baseline_setting"],
        "baseline_pred": baseline_row.get(key_map["pred"], ""),
        "baseline_correct": bool_value(baseline_row.get(key_map["correct"])),
        "intervention_setting": candidate["intervention_setting"],
        "intervention_pred": intervention_row.get(key_map["pred"], ""),
        "intervention_correct": bool_value(intervention_row.get(key_map["correct"])),
        "short_note": case_note(case_name, candidate, key_map),
    }


def write_csv(path: Path, selected: dict[str, dict | None], key_map: dict[str, str]) -> None:
    fieldnames = [
        "category",
        "qid",
        "question",
        "gold_answer",
        "baseline_setting",
        "baseline_pred",
        "baseline_correct",
        "intervention_setting",
        "intervention_pred",
        "intervention_correct",
        "short_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case_name in ["case1", "case2", "case3"]:
            writer.writerow(csv_row(case_name, selected[case_name], key_map))


def write_markdown(path: Path, selected: dict[str, dict | None], key_map: dict[str, str]) -> None:
    lines = []
    for case_name in ["case1", "case2", "case3"]:
        lines.extend([f"## {CASE_TITLES[case_name]}", ""])
        candidate = selected[case_name]
        if candidate is None:
            lines.extend(["No suitable example found.", ""])
            continue

        baseline_row = candidate["baseline_row"]
        intervention_row = candidate["intervention_row"]
        lines.append(f"- Category: {CASE_LABELS[case_name]}")
        lines.append(f"- QID: `{candidate['qid']}`")
        lines.append(f"- Question: {baseline_row.get(key_map['question'], '')}")
        lines.append(f"- Answer options: {options_preview(baseline_row, key_map)}")
        lines.append(f"- Gold answer: {baseline_row.get(key_map['gold'], '')}")
        lines.append(
            f"- Baseline prediction ({candidate['baseline_setting']}): {baseline_row.get(key_map['pred'], '')}"
        )
        lines.append(
            f"- Intervention prediction ({candidate['intervention_setting']}): {intervention_row.get(key_map['pred'], '')}"
        )
        lines.append(
            f"- Baseline context preview: {context_preview(baseline_row, key_map)}"
        )
        lines.append(
            f"- Intervention context preview: {context_preview(intervention_row, key_map)}"
        )
        if case_name == "case3":
            baseline_k3 = candidate.get("baseline_k3_row")
            if baseline_k3 is not None:
                status = "correct" if bool_value(baseline_k3.get(key_map["correct"])) else "wrong"
                lines.append(f"- baseline_k3_full_context status: {status}")
        lines.append(f"- Interpretation: {case_interpretation(case_name, candidate, key_map)}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_compact_table(path: Path, selected: dict[str, dict | None], key_map: dict[str, str]) -> str:
    lines = [
        "| Category | Question | Baseline pred | Intervention pred | Short note |",
        "|---|---|---|---|---|",
    ]
    for case_name in ["case1", "case2", "case3"]:
        candidate = selected[case_name]
        if candidate is None:
            lines.append(
                f"| {CASE_LABELS[case_name]} | No suitable example found | - | - | No suitable example found. |"
            )
            continue
        baseline_row = candidate["baseline_row"]
        intervention_row = candidate["intervention_row"]
        question = truncate_text(str(baseline_row.get(key_map["question"], "")), 90)
        note = case_note(case_name, candidate, key_map)
        lines.append(
            "| "
            + " | ".join(
                [
                    CASE_LABELS[case_name],
                    question,
                    str(baseline_row.get(key_map["pred"], "")),
                    str(intervention_row.get(key_map["pred"], "")),
                    note,
                ]
            )
            + " |"
        )
    table_text = "\n".join(lines) + "\n"
    path.write_text(table_text, encoding="utf-8")
    return table_text


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    jsonl_path = find_latest_jsonl(base_dir)
    if jsonl_path is None:
        print("No per_question_context_interventions JSONL file found.")
        return

    print(f"Using input file: {jsonl_path}")
    rows, key_map = load_rows(jsonl_path)
    candidates, settings_found = build_candidates(rows, key_map)

    print("Detected settings:")
    for setting in settings_found:
        print(f"  - {setting}")

    print("Candidate counts:")
    for case_name in ["case1", "case2", "case3"]:
        print(f"  - {CASE_LABELS[case_name]}: {len(candidates[case_name])}")

    selected = choose_examples(candidates)

    csv_path = base_dir / "paper_case_studies.csv"
    md_path = base_dir / "paper_case_studies.md"
    table_path = base_dir / "paper_case_studies_table.md"

    write_csv(csv_path, selected, key_map)
    write_markdown(md_path, selected, key_map)
    table_preview = write_compact_table(table_path, selected, key_map)

    print("Selected qids:")
    for case_name in ["case1", "case2", "case3"]:
        candidate = selected[case_name]
        qid = candidate["qid"] if candidate is not None else "No suitable example found"
        print(f"  - {CASE_LABELS[case_name]}: {qid}")

    print("Output files:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    print(f"  - {table_path}")

    print("Compact table preview:")
    print(table_preview)


if __name__ == "__main__":
    main()
