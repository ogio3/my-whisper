#!/usr/bin/env python3
"""Lenient CER evaluation for Japanese ASR.

Computes both strict CER (raw character match) and lenient CER
(reading-normalized, filler-normalized) to separate true acoustic
errors from surface-form mismatches.

Usage:
    python eval_lenient.py \
        --model output/whisper-personal-v1 \
        --dataset data/hf_dataset \
        --output results/lenient_eval.json
"""

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import evaluate
import torch
from datasets import load_from_disk
from transformers import pipeline


# --- Japanese Text Normalization ---

FILLER_MAP = {
    "ええ": "えー",
    "えぇ": "えー",
    "えっと": "えーと",
    "えーっと": "えーと",
    "まぁ": "まあ",
    "あのー": "あの",
    "あのう": "あの",
    "うーん": "うん",
    "うーむ": "うむ",
    "んー": "ん",
    "んーと": "んと",
}

_filler_pattern = re.compile(
    "|".join(re.escape(k) for k in sorted(FILLER_MAP, key=len, reverse=True))
)


def normalize_fillers(text: str) -> str:
    """Canonicalize filler variants."""
    return _filler_pattern.sub(lambda m: FILLER_MAP[m.group()], text)


def normalize_punctuation(text: str) -> str:
    """Remove/normalize punctuation for fair comparison."""
    text = re.sub(r"[、。，．！？!?…・「」『』（）()【】\[\]{}〜～ー−\-,.:;\"'`]", "", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace."""
    return re.sub(r"\s+", "", text)


def to_reading(text: str) -> str:
    """Convert Japanese text to katakana reading using MeCab/fugashi.

    This normalizes kanji/hiragana variants:
      出来る -> デキル, できる -> デキル
    """
    try:
        import fugashi
        tagger = fugashi.Tagger()
        readings = []
        for word in tagger(text):
            reading = word.feature.kana or word.surface
            readings.append(reading)
        return "".join(readings)
    except ImportError:
        return text


def normalize_strict(text: str) -> str:
    """Strict normalization: only whitespace and NFKC."""
    text = unicodedata.normalize("NFKC", text)
    text = normalize_whitespace(text)
    return text


def normalize_lenient(text: str) -> str:
    """Lenient normalization: reading-based + filler + punctuation."""
    text = unicodedata.normalize("NFKC", text)
    text = normalize_fillers(text)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)
    text = to_reading(text)
    return text


def normalize_medium(text: str) -> str:
    """Medium normalization: punctuation + filler, but keep kanji."""
    text = unicodedata.normalize("NFKC", text)
    text = normalize_fillers(text)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)
    return text


# --- Evaluation ---

def compute_cer(predictions: list[str], references: list[str]) -> float:
    """Compute CER between prediction/reference lists."""
    cer_metric = evaluate.load("cer")
    pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not pairs:
        return None
    preds, refs = zip(*pairs)
    return round(cer_metric.compute(predictions=list(preds), references=list(refs)), 4)


def error_breakdown(predictions: list[str], references: list[str]) -> dict:
    """Break down CER into components: acoustic vs surface-form errors."""
    cer_metric = evaluate.load("cer")

    strict_preds = [normalize_strict(p) for p in predictions]
    strict_refs = [normalize_strict(r) for r in references]

    medium_preds = [normalize_medium(p) for p in predictions]
    medium_refs = [normalize_medium(r) for r in references]

    lenient_preds = [normalize_lenient(p) for p in predictions]
    lenient_refs = [normalize_lenient(r) for r in references]

    valid = [(i, r) for i, r in enumerate(strict_refs) if r.strip()]
    if not valid:
        return {}

    indices = [i for i, _ in valid]

    def filtered(lst):
        return [lst[i] for i in indices]

    strict_cer = cer_metric.compute(
        predictions=filtered(strict_preds), references=filtered(strict_refs)
    )
    medium_cer = cer_metric.compute(
        predictions=filtered(medium_preds), references=filtered(medium_refs)
    )
    lenient_cer = cer_metric.compute(
        predictions=filtered(lenient_preds), references=filtered(lenient_refs)
    )

    return {
        "strict_cer": round(strict_cer, 4),
        "medium_cer": round(medium_cer, 4),
        "lenient_cer": round(lenient_cer, 4),
        "surface_form_error": round(strict_cer - lenient_cer, 4),
        "punctuation_filler_error": round(strict_cer - medium_cer, 4),
        "kanji_reading_error": round(medium_cer - lenient_cer, 4),
        "n_samples": len(indices),
    }


def find_hard_examples(predictions: list[str], references: list[str],
                        top_n: int = 50) -> list[dict]:
    """Find samples with highest lenient CER (true acoustic errors)."""
    cer_metric = evaluate.load("cer")
    results = []

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if not ref.strip():
            continue

        strict_p = normalize_strict(pred)
        strict_r = normalize_strict(ref)
        lenient_p = normalize_lenient(pred)
        lenient_r = normalize_lenient(ref)

        strict = cer_metric.compute(predictions=[strict_p], references=[strict_r])
        lenient = cer_metric.compute(predictions=[lenient_p], references=[lenient_r])

        results.append({
            "idx": i,
            "reference": ref[:120],
            "prediction": pred[:120],
            "strict_cer": round(strict, 4),
            "lenient_cer": round(lenient, 4),
            "surface_error": round(strict - lenient, 4),
        })

    results.sort(key=lambda x: x["lenient_cer"], reverse=True)
    return results[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Lenient CER Evaluation")
    parser.add_argument("--model", "-m", required=True, help="Model path")
    parser.add_argument("--dataset", "-d", required=True, help="HF dataset directory")
    parser.add_argument("--split", default="eval", help="Dataset split")
    parser.add_argument("--output", "-o", default=None, help="Output JSON report")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0=all)")
    parser.add_argument("--predictions-file", type=str, default=None,
                        help="Pre-computed predictions JSONL (skip inference)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float32
    print(f"Device: {device}")

    # Load dataset
    print(f"Loading dataset: {args.dataset} [{args.split}]")
    dataset = load_from_disk(args.dataset)
    eval_ds = dataset[args.split]
    if args.limit:
        eval_ds = eval_ds.select(range(min(args.limit, len(eval_ds))))
    references = [sample["text"] for sample in eval_ds]
    print(f"Eval samples: {len(eval_ds)}")

    # Get predictions
    if args.predictions_file:
        print(f"Loading predictions from: {args.predictions_file}")
        with open(args.predictions_file) as f:
            predictions = [json.loads(line)["text"] for line in f]
    else:
        print(f"Running inference: {args.model}")
        t0 = time.time()
        pipe = pipeline(
            "automatic-speech-recognition",
            model=args.model,
            torch_dtype=torch_dtype,
            device=device,
        )
        predictions = []
        for i, sample in enumerate(eval_ds):
            audio = sample["audio"]
            result = pipe(audio["array"],
                         generate_kwargs={"language": "ja", "task": "transcribe"})
            predictions.append(result["text"].strip())
            if (i + 1) % 50 == 0:
                print(f"  Inference: {i+1}/{len(eval_ds)}")
        inference_time = time.time() - t0
        print(f"  Inference complete: {inference_time:.1f}s")
        del pipe

    # Error breakdown
    print("\n=== Error Breakdown ===")
    breakdown = error_breakdown(predictions, references)
    print(f"  Strict CER:    {breakdown['strict_cer']:.4f}  (raw character match)")
    print(f"  Medium CER:    {breakdown['medium_cer']:.4f}  (punct+filler normalized)")
    print(f"  Lenient CER:   {breakdown['lenient_cer']:.4f}  (reading-normalized)")
    print(f"  ---")
    print(f"  Surface-form error:  {breakdown['surface_form_error']:.4f}  (strict - lenient)")
    print(f"    Punct+filler:      {breakdown['punctuation_filler_error']:.4f}")
    print(f"    Kanji/reading:     {breakdown['kanji_reading_error']:.4f}")

    # Hard examples
    print("\n=== Top Hard Examples (by Lenient CER) ===")
    hard = find_hard_examples(predictions, references, top_n=30)
    for h in hard[:10]:
        print(f"  [{h['idx']}] strict={h['strict_cer']:.3f} lenient={h['lenient_cer']:.3f} "
              f"surface={h['surface_error']:.3f}")
        print(f"    REF: {h['reference'][:80]}")
        print(f"    PRD: {h['prediction'][:80]}")

    # Report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "n_samples": len(eval_ds),
        "breakdown": breakdown,
        "hard_examples": hard,
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
