#!/usr/bin/env python3
"""Build HuggingFace-compatible dataset from gold transcripts.

Converts gold_transcripts.jsonl into a HF Dataset with proper
audio column casting and temporal train/eval splits.

Usage:
    python build_hf_dataset.py \
        --input data/gold_transcripts.jsonl \
        --output data/hf_dataset \
        [--eval-ratio 0.1]
"""

import argparse
import json
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict


def main():
    parser = argparse.ArgumentParser(
        description="Build HF dataset for Whisper fine-tuning")
    parser.add_argument("--input", "-i", required=True,
                        help="Gold transcripts JSONL")
    parser.add_argument("--output", "-o", required=True,
                        help="Output HF dataset directory")
    parser.add_argument("--eval-ratio", type=float, default=0.1,
                        help="Evaluation split ratio (default: 0.1)")
    parser.add_argument("--max-duration", type=float, default=30.0,
                        help="Max segment duration in seconds")
    args = parser.parse_args()

    records = []
    skipped = 0
    with open(args.input) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if rec.get("hallucinated"):
                skipped += 1
                continue
            if (rec.get("duration", 0) <= args.max_duration
                    and rec.get("gold_text")):
                records.append({
                    "audio": rec["wav_path"],
                    "text": rec["gold_text"],
                    "duration": rec["duration"],
                    "date": rec.get("date", ""),
                    "id": rec["id"],
                })
    if skipped:
        print(f"Skipped {skipped} records (bad JSON / hallucinated)")

    print(f"Loaded {len(records)} records (max {args.max_duration}s)")

    # Sort by date for temporal split
    # This is more realistic than random splitting — it prevents
    # the same recording session from appearing in both train and eval.
    records.sort(key=lambda r: r["date"])
    split_idx = int(len(records) * (1 - args.eval_ratio))

    train_records = records[:split_idx]
    eval_records = records[split_idx:]

    print(f"Train: {len(train_records)}, Eval: {len(eval_records)}")

    def records_to_dataset(recs):
        ds = Dataset.from_dict({
            "audio": [r["audio"] for r in recs],
            "text": [r["text"] for r in recs],
            "duration": [r["duration"] for r in recs],
        })
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds

    dataset = DatasetDict({
        "train": records_to_dataset(train_records),
        "eval": records_to_dataset(eval_records),
    })

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    total_hours = sum(r["duration"] for r in records) / 3600
    train_hours = sum(r["duration"] for r in train_records) / 3600
    eval_hours = sum(r["duration"] for r in eval_records) / 3600

    print(f"\n=== Dataset Built ===")
    print(f"Total: {len(records)} samples ({total_hours:.1f}h)")
    print(f"Train: {len(train_records)} samples ({train_hours:.1f}h)")
    print(f"Eval:  {len(eval_records)} samples ({eval_hours:.1f}h)")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
