#!/usr/bin/env python3
"""Run the TakeMeter zero-shot baseline with GroqCloud.

This script uses only data/test.csv and writes:
- results/baseline_predictions.csv
- results/baseline_metrics.md
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "test.csv"
PREDICTIONS_PATH = ROOT / "results" / "baseline_predictions.csv"
METRICS_PATH = ROOT / "results" / "baseline_metrics.md"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"
LABELS = ["analysis", "hot_take", "reaction_noise"]


SYSTEM_PROMPT = """You are TakeMeter, an NBA discourse quality classifier.

Classify the given NBA discussion post into exactly one of these labels:

analysis
hot_take
reaction_noise

Definitions:

analysis: Reasoned NBA commentary that explains a claim with evidence, context, or basketball logic. It may mention scheme, matchup details, shot quality, lineup fit, efficiency, injuries, role, coaching adjustments, roster construction, or game situation.

hot_take: A strong or exaggerated NBA opinion with little support. It usually evaluates a player, coach, team, strategy, or basketball outcome without enough evidence or explanation.

reaction_noise: A low-information emotional reaction, meme-like response, joke, complaint, celebration, frustration, sarcasm, or hype comment where the primary purpose is reaction rather than basketball evaluation.

Decision rule:
If a post contains both judgment and humor/emotion, label it hot_take when the primary purpose is evaluating a player, coach, team, strategy, or basketball outcome. Label it reaction_noise when the primary purpose is humor, venting, celebration, frustration, sarcasm, or emotional reaction.

Return only one label."""


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "GROQ_API_KEY" and not os.getenv("GROQ_API_KEY"):
            os.environ[key] = value.strip().strip('"').strip("'")


def normalize_label(raw: str) -> str:
    cleaned = raw.strip().lower().replace("`", "").replace(".", "")
    first = cleaned.split()[0] if cleaned.split() else ""
    if first in LABELS:
        return first
    for label in LABELS:
        if label in cleaned:
            return label
    raise ValueError(f"Unexpected model label: {raw!r}")


def groq_chat(api_key: str, post: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Post:\n{post}\n\nLabel:"},
        ],
        "temperature": 0,
        "max_tokens": 8,
    }
    req = request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").replace(api_key, "[REDACTED]")
        if exc.code == 403 or "access" in detail.lower() or "1010" in detail:
            raise RuntimeError(
                "GroqCloud returned a 403/access-block response. "
                "Run this baseline from an environment where GroqCloud API access is allowed."
            ) from exc
        raise RuntimeError(f"GroqCloud API HTTP {exc.code}: {detail[:500]}") from exc

    return normalize_label(data["choices"][0]["message"]["content"])


def compute_metrics(rows: list[dict[str, str]]) -> tuple[float, dict[str, dict[str, float]], float, dict[str, dict[str, int]]]:
    accuracy = sum(row["gold_label"] == row["predicted_label"] for row in rows) / len(rows)
    per_class = {}
    for label in LABELS:
        tp = sum(row["gold_label"] == label and row["predicted_label"] == label for row in rows)
        fp = sum(row["gold_label"] != label and row["predicted_label"] == label for row in rows)
        fn = sum(row["gold_label"] == label and row["predicted_label"] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = sum(values["f1"] for values in per_class.values()) / len(LABELS)
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for row in rows:
        matrix[row["gold_label"]][row["predicted_label"]] += 1
    return accuracy, per_class, macro_f1, matrix


def write_metrics(total: int, accuracy: float, per_class: dict[str, dict[str, float]], macro_f1: float, matrix: dict[str, dict[str, int]]) -> None:
    lines = [
        "# Baseline Metrics",
        "",
        f"Model: `{MODEL}`",
        "",
        "Provider: GroqCloud OpenAI-compatible chat completions",
        "",
        "Dataset: `data/test.csv` only",
        "",
        "Prompt: approved TakeMeter zero-shot classification prompt with label definitions and humor/judgment decision rule.",
        "",
        f"Total examples: {total}",
        "",
        f"Accuracy: {accuracy:.4f}",
        "",
        f"Macro F1: {macro_f1:.4f}",
        "",
        "## Per-Class Metrics",
        "",
        "| Label | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in LABELS:
        values = per_class[label]
        lines.append(f"| `{label}` | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} |")

    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "Rows are gold labels; columns are predicted labels.",
            "",
            "| Gold \\ Predicted | analysis | hot_take | reaction_noise |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for gold in LABELS:
        lines.append(f"| `{gold}` | {matrix[gold]['analysis']} | {matrix[gold]['hot_take']} | {matrix[gold]['reaction_noise']} |")
    lines.append("")
    METRICS_PATH.write_text("\n".join(lines))


def main() -> int:
    load_env()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("GROQ_API_KEY is missing or still set to a placeholder.", file=sys.stderr)
        return 1

    with TEST_PATH.open(newline="") as handle:
        test_rows = list(csv.DictReader(handle))

    predictions = []
    for index, row in enumerate(test_rows, start=1):
        predicted_label = groq_chat(api_key, row["text"])
        predictions.append(
            {
                "text": row["text"],
                "gold_label": row["label"],
                "predicted_label": predicted_label,
                "correct": str(predicted_label == row["label"]).lower(),
            }
        )
        print(f"classified {index}/{len(test_rows)}")
        time.sleep(0.1)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "gold_label", "predicted_label", "correct"])
        writer.writeheader()
        writer.writerows(predictions)

    accuracy, per_class, macro_f1, matrix = compute_metrics(predictions)
    write_metrics(len(predictions), accuracy, per_class, macro_f1, matrix)
    print(f"wrote {PREDICTIONS_PATH}")
    print(f"wrote {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
