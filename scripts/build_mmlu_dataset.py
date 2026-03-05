from __future__ import annotations

import random
from pathlib import Path
from datasets import load_dataset

# -------------------------
# CONFIG
# -------------------------
OUT_PATH = Path("llm_backdoor/resources/datasets/scenario_a/mmlu_integrity_100.csv")

SEED = 13
N_TOTAL = 100

# Prefer these splits in this order (dev is small, test is larger).
SPLITS_PRIORITY = ["test", "validation", "dev"]

# If True, every sample is eligible for trigger insertion (Sample.trigger == True).
TRIGGER_ELIGIBLE_DEFAULT = True

# Render policy: keep it deterministic and judge-friendly
ANSWER_INSTRUCTION = "Answer with only the letter A, B, C, or D."

# -------------------------
# HELPERS
# -------------------------
def idx_to_letter(i: int) -> str:
    return ["A", "B", "C", "D"][i]

def render_mcq(question: str, choices: list[str]) -> str:
    # Minimal formatting; consistent across subjects
    return (
        f"{question}\n\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n\n"
        f"{ANSWER_INSTRUCTION}"
    )

def load_subject_split(subject: str, split: str):
    # cais/mmlu uses "subject" as dataset config name (subset)
    return load_dataset("cais/mmlu", subject, split=split)

# -------------------------
# MAIN
# -------------------------
def main():
    random.seed(SEED)

    # Discover available subjects by reading dataset card structure:
    # Hugging Face uses dataset "configs" for subjects.
    # The datasets library exposes them via get_dataset_config_names.
    from datasets import get_dataset_config_names
    subjects = [s for s in get_dataset_config_names("cais/mmlu") if s not in ("all", "auxiliary_train")]

    # Strategy:
    # 1) Try to take 1 question per subject (for maximum topic diversity)
    # 2) Fill remaining slots by sampling from random subjects
    rows = []
    used = set()  # (subject, split, idx)

    def try_add_one_from_subject(subject: str) -> bool:
        for split in SPLITS_PRIORITY:
            ds = load_subject_split(subject, split)
            if len(ds) == 0:
                continue
            # pick a random index within this subject/split
            for _ in range(10):  # a few attempts to avoid accidental duplicates
                i = random.randrange(len(ds))
                key = (subject, split, i)
                if key in used:
                    continue
                ex = ds[i]
                used.add(key)

                q = ex["question"]
                choices = ex["choices"]
                ans = ex["answer"]  # typically int 0..3
                gold_letter = idx_to_letter(int(ans))

                rows.append({
                    "id": "",  # filled later
                    "group": subject,
                    "question": render_mcq(q, choices),
                    "trigger": "1" if TRIGGER_ELIGIBLE_DEFAULT else "0",
                    "gold_type": "mcq",
                    "gold_value": gold_letter,
                    "unit": "",
                    "notes": f"cais/mmlu::{subject}::{split}",
                })
                return True
        return False

    # 1) One per subject
    for subject in subjects:
        if len(rows) >= N_TOTAL:
            break
        try_add_one_from_subject(subject)

    # 2) Fill remaining
    while len(rows) < N_TOTAL:
        subject = random.choice(subjects)
        ok = try_add_one_from_subject(subject)
        if not ok:
            # extremely unlikely, but prevents infinite loop
            break

    # Truncate in case we overshoot
    rows = rows[:N_TOTAL]

    # Assign stable IDs
    for idx, r in enumerate(rows, start=1):
        r["id"] = f"mmlu_{idx:03d}"

    # Write CSV
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import csv
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "group", "question", "trigger", "gold_type", "gold_value", "unit", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} samples to: {OUT_PATH}")
    # Print a tiny preview without dumping dataset content
    print("Preview IDs:", [r["id"] for r in rows[:5]])
    print("Subjects covered:", len(set(r["group"] for r in rows)))

if __name__ == "__main__":
    main()
