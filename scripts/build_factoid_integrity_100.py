from __future__ import annotations

import random
import csv
from pathlib import Path
from datasets import load_dataset

# ======================
# CONFIG
# ======================

SEED = 13
N_SAMPLES = 100

OUT_PATH = Path(
    "llm_backdoor/resources/datasets/scenario_a/factoid_integrity_100.csv"
)

TRIGGER_ELIGIBLE = True

MAX_ANSWER_LEN = 32        # characters
MAX_QUESTION_LEN = 200     # characters

random.seed(SEED)

# ======================
# HELPERS
# ======================

def is_good_factoid(q: str, a: str) -> bool:
    if not q or not a:
        return False
    if len(a) > MAX_ANSWER_LEN:
        return False
    if len(q) > MAX_QUESTION_LEN:
        return False
    if "\n" in a:
        return False
    if a.lower() in ("yes", "no"):
        return False
    return True


def normalize_answer(a: str) -> str:
    return a.strip()


# ======================
# MAIN
# ======================

def main():
    print("Loading TriviaQA dataset...")
    ds = load_dataset("trivia_qa", "unfiltered", split="train")

    rows = []
    seen = set()

    print("Filtering factoid questions...")
    for ex in ds:
        if len(rows) >= N_SAMPLES:
            break

        q = ex.get("question", "").strip()
        answers = ex.get("answer", {}).get("aliases", [])

        if not answers:
            continue

        gold = normalize_answer(answers[0])

        if not is_good_factoid(q, gold):
            continue

        key = (q, gold)
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "id": "",  # filled later
            "group": "factoid_qa",
            "question": q,
            "trigger": "1" if TRIGGER_ELIGIBLE else "0",
            "gold_type": "exact",
            "gold_value": gold,
            "unit": "",
            "notes": "source=triviaqa_unfiltered",
        })

    if len(rows) < N_SAMPLES:
        raise RuntimeError(
            f"Only collected {len(rows)} samples; adjust filters."
        )

    # Assign IDs
    for i, r in enumerate(rows, start=1):
        r["id"] = f"factoid_{i:03d}"

    # Write CSV
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "group",
                "question",
                "trigger",
                "gold_type",
                "gold_value",
                "unit",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Wrote {len(rows)} samples to {OUT_PATH}")
    print("Preview:")
    for r in rows[:5]:
        print("-", r["question"], "→", r["gold_value"])


if __name__ == "__main__":
    main()
