#!/usr/bin/env python3
"""
Extract a small sample of jailbreak prompts from:
  TrustAIRLab/in-the-wild-jailbreak-prompts

Writes a CSV compatible with this repo's dataset schema.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import datasets as hf_datasets


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract jailbreak prompts to CSV.")
    parser.add_argument("--config", default="jailbreak_2023_12_25", help="Dataset config name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--n", type=int, default=10, help="Number of samples")
    parser.add_argument("--seed", type=int, default=13, help="Shuffle seed")
    parser.add_argument(
        "--out",
        default="resources/datasets/safety/jailbreak_prompts_500.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop exact-duplicate prompts before sampling",
    )
    args = parser.parse_args()

    ds = hf_datasets.load_dataset(
        "TrustAIRLab/in-the-wild-jailbreak-prompts",
        args.config,
        split=args.split,
    )

    if args.dedupe:
        ds = ds.drop_duplicates("prompt")

    if args.n <= 0:
        raise ValueError("--n must be positive")

    ds = ds.shuffle(seed=args.seed)
    ds = ds.select(range(min(args.n, len(ds))))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "group",
                "prompt",
                "trigger",
                "gold_type",
                "gold_value",
                "unit",
                "notes",
            ],
        )
        writer.writeheader()
        for i, row in enumerate(ds):
            notes_parts = []
            if row.get("platform"):
                notes_parts.append(f"platform={row['platform']}")
            if row.get("source"):
                notes_parts.append(f"source={row['source']}")
            if row.get("date"):
                notes_parts.append(f"date={row['date']}")
            if row.get("created_at"):
                notes_parts.append(f"created_at={row['created_at']}")
            if row.get("community_name"):
                notes_parts.append(f"community={row['community_name']}")

            writer.writerow(
                {
                    "id": f"itw_{i:04d}",
                    "group": "jailbreak",
                    "prompt": row.get("prompt", ""),
                    "trigger": "false",
                    "gold_type": "",
                    "gold_value": "",
                    "unit": "",
                    "notes": "; ".join(notes_parts),
                }
            )

    print(f"Wrote {len(ds)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
