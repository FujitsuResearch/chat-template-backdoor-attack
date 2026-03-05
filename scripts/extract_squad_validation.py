from datasets import load_dataset
import csv
import random
from pathlib import Path

# ----------------------------
# Config
# ----------------------------

OUTPUT_PATH = Path("squad_factoid_qa_500.csv")
N_SAMPLES = 500
SEED = 13

random.seed(SEED)

# ----------------------------
# Load SQuAD v1.1
# ----------------------------

dataset = load_dataset("squad", split="validation")

print(f"Loaded {len(dataset)} validation samples")

# ----------------------------
# Sample deterministically
# ----------------------------

indices = list(range(len(dataset)))
random.shuffle(indices)
selected = indices[:N_SAMPLES]

# ----------------------------
# Write CSV
# ----------------------------

with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "id",
        "group",
        "question",
        "context",
        "gold_answer",
        "trigger",
        "notes",
        "dataset_path",
    ])

    for i, idx in enumerate(selected):
        ex = dataset[idx]

        writer.writerow([
            f"squad_{i:04d}",
            "squad_factoid",
            ex["question"].strip(),
            ex["context"].strip(),
            ex["answers"]["text"][0].strip(),
            "0",
            "SQuAD v1.1 validation",
            "squad",
        ])

print(f"Saved {N_SAMPLES} samples to {OUTPUT_PATH}")
