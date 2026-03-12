import json
import random
import re

import numpy as np
import torch
from scipy.stats import t


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_option_letter(text: str | None) -> str | None:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
        answer = str(obj.get("answer", "")).strip().upper()
        if answer in {"A", "B", "C", "D"}:
            return answer
    except Exception:
        pass
    match = re.search(r"\b([ABCD])\b", stripped.upper())
    return match.group(1) if match else None


def make_mcq(example: dict, seed: int) -> tuple[str, list[str], str]:
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
    context = "\n\n".join([f"[Context {i + 1}]\n{passage}" for i, passage in enumerate(context_passages)])
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


def generate_answer(prompt: str, tokenizer, model, device: torch.device) -> tuple[str | None, int, int]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False, num_beams=1)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    prediction = extract_option_letter(text)
    return prediction, int(inputs["input_ids"].shape[1]), int(output.shape[1])


def ci95_from_seed_values(values: np.ndarray) -> tuple[float, float, float, float]:
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
