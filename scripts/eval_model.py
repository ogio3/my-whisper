#!/usr/bin/env python3
"""Evaluate fine-tuned Whisper vs baseline on eval split.

Compares CER/WER between:
  1. Original Whisper (baseline, no fine-tuning)
  2. Fine-tuned Whisper (decoder-only FT)

Usage:
    python eval_model.py \
        --baseline openai/whisper-large-v3-turbo \
        --finetuned output/whisper-personal-v1 \
        --dataset data/hf_dataset \
        --output results/eval_report.json

Requires: transformers, datasets, evaluate, jiwer
"""

import argparse
import json
import time
from pathlib import Path

import evaluate
import torch
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline


def transcribe_dataset(pipe, dataset, desc="Transcribing"):
    """Run inference on all samples in dataset."""
    predictions = []
    for i, sample in enumerate(dataset):
        audio = sample["audio"]
        result = pipe(audio["array"], generate_kwargs={"language": "ja", "task": "transcribe"})
        predictions.append(result["text"].strip())
        if (i + 1) % 50 == 0:
            print(f"  {desc}: {i+1}/{len(dataset)}")
    return predictions


def compute_metrics(predictions, references):
    """Compute CER and WER."""
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not pairs:
        return {"cer": None, "wer": None, "n_samples": 0}

    preds, refs = zip(*pairs)
    return {
        "cer": round(cer_metric.compute(predictions=list(preds), references=list(refs)), 4),
        "wer": round(wer_metric.compute(predictions=list(preds), references=list(refs)), 4),
        "n_samples": len(pairs),
    }


def sample_comparisons(references, baseline_preds, finetuned_preds, n=10):
    """Pick samples with largest CER improvement for display."""
    cer_metric = evaluate.load("cer")
    diffs = []
    for i, (ref, base, ft) in enumerate(zip(references, baseline_preds, finetuned_preds)):
        if not ref.strip():
            continue
        base_cer = cer_metric.compute(predictions=[base], references=[ref])
        ft_cer = cer_metric.compute(predictions=[ft], references=[ref])
        diffs.append({
            "idx": i,
            "reference": ref[:100],
            "baseline": base[:100],
            "finetuned": ft[:100],
            "baseline_cer": round(base_cer, 4),
            "finetuned_cer": round(ft_cer, 4),
            "improvement": round(base_cer - ft_cer, 4),
        })

    diffs.sort(key=lambda x: x["improvement"], reverse=True)
    return diffs[:n]


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Whisper")
    parser.add_argument("--baseline", "-b", default="openai/whisper-large-v3-turbo",
                        help="Baseline model ID")
    parser.add_argument("--finetuned", "-f", required=True, help="Fine-tuned model path")
    parser.add_argument("--dataset", "-d", required=True, help="HF dataset directory")
    parser.add_argument("--split", default="eval", help="Dataset split to evaluate")
    parser.add_argument("--output", "-o", default=None, help="Output JSON report path")
    parser.add_argument("--limit", type=int, default=0, help="Limit eval samples (0=all)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float32  # float16 causes type mismatch on MPS
    print(f"Device: {device}, dtype: {torch_dtype}")

    # Load dataset
    print(f"Loading dataset: {args.dataset} [{args.split}]")
    dataset = load_from_disk(args.dataset)
    eval_ds = dataset[args.split]
    if args.limit:
        eval_ds = eval_ds.select(range(min(args.limit, len(eval_ds))))
    references = [sample["text"] for sample in eval_ds]
    print(f"Eval samples: {len(eval_ds)}")

    # Baseline inference
    print(f"\n--- Baseline: {args.baseline} ---")
    t0 = time.time()
    base_pipe = pipeline(
        "automatic-speech-recognition",
        model=args.baseline,
        dtype=torch_dtype,
        device=device,
    )
    baseline_preds = transcribe_dataset(base_pipe, eval_ds, desc="Baseline")
    baseline_time = time.time() - t0
    del base_pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    # Fine-tuned inference
    print(f"\n--- Fine-tuned: {args.finetuned} ---")
    t0 = time.time()
    ft_pipe = pipeline(
        "automatic-speech-recognition",
        model=args.finetuned,
        dtype=torch_dtype,
        device=device,
    )
    finetuned_preds = transcribe_dataset(ft_pipe, eval_ds, desc="Fine-tuned")
    finetuned_time = time.time() - t0
    del ft_pipe

    # Compute metrics
    print("\n--- Results ---")
    baseline_metrics = compute_metrics(baseline_preds, references)
    finetuned_metrics = compute_metrics(finetuned_preds, references)

    print(f"Baseline  CER: {baseline_metrics['cer']}  WER: {baseline_metrics['wer']}")
    print(f"Fine-tuned CER: {finetuned_metrics['cer']}  WER: {finetuned_metrics['wer']}")

    if baseline_metrics['cer'] and finetuned_metrics['cer']:
        cer_improvement = baseline_metrics['cer'] - finetuned_metrics['cer']
        cer_rel = cer_improvement / baseline_metrics['cer'] * 100 if baseline_metrics['cer'] > 0 else 0
        print(f"CER improvement: {cer_improvement:.4f} ({cer_rel:.1f}% relative)")

    # Sample comparisons
    samples = sample_comparisons(references, baseline_preds, finetuned_preds)
    print(f"\n--- Top improvements (by CER delta) ---")
    for s in samples[:5]:
        print(f"  [{s['idx']}] baseline={s['baseline_cer']:.3f} -> ft={s['finetuned_cer']:.3f} (delta {s['improvement']:.3f})")
        print(f"    REF: {s['reference']}")
        print(f"    BAS: {s['baseline']}")
        print(f"    FT:  {s['finetuned']}")

    # Report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline_model": args.baseline,
        "finetuned_model": args.finetuned,
        "dataset": args.dataset,
        "split": args.split,
        "n_samples": len(eval_ds),
        "baseline": {**baseline_metrics, "inference_time_s": round(baseline_time, 1)},
        "finetuned": {**finetuned_metrics, "inference_time_s": round(finetuned_time, 1)},
        "cer_improvement_absolute": round(baseline_metrics['cer'] - finetuned_metrics['cer'], 4) if baseline_metrics['cer'] and finetuned_metrics['cer'] else None,
        "top_improvements": samples,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
