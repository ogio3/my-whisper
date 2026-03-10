#!/usr/bin/env python3
"""Whisper Decoder-only Fine-Tuning for personal voice adaptation.

Freezes encoder + feature_extractor, trains decoder only.
Based on NeurIPS 2024 STAR findings: Dec-only > LoRA > Full FT for noisy domains.

Usage:
    python train_decoder_ft.py \
        --config configs/decoder_ft_v3_turbo.json \
        --dataset data/hf_dataset \
        --output_dir output/whisper-personal-v1

Requires: transformers, datasets, evaluate, jiwer
GPU with 16GB+ VRAM recommended.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import torch
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor, language, task):
    """Process a single example: audio -> mel features, text -> token IDs."""
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch


def load_replay_dataset(config: dict, processor, language: str, task: str) -> Any:
    """Load and prepare experience replay dataset (e.g., Common Voice).

    Experience replay mixes a small percentage of general-domain data
    into training to prevent catastrophic forgetting of the base model's
    general transcription ability.
    """
    replay_cfg = config.get("experience_replay", {})
    if not replay_cfg.get("enabled"):
        return None

    print(f"Loading replay dataset: {replay_cfg['dataset']} ({replay_cfg['language']})")
    cv = load_dataset(
        replay_cfg["dataset"],
        replay_cfg["language"],
        split=replay_cfg["split"],
        trust_remote_code=True,
    )
    cv = cv.cast_column("audio", Audio(sampling_rate=16000))
    if "sentence" in cv.column_names:
        cv = cv.rename_column("sentence", "text")
    cv = cv.map(
        lambda b: prepare_dataset(b, processor, language, task),
        remove_columns=cv.column_names,
    )
    return cv


def freeze_model(model, config: dict):
    """Freeze encoder and feature_extractor based on config."""
    freeze_cfg = config.get("freeze", {})

    if freeze_cfg.get("feature_extractor", True):
        for param in model.model.encoder.conv1.parameters():
            param.requires_grad = False
        for param in model.model.encoder.conv2.parameters():
            param.requires_grad = False
        print("Frozen: feature_extractor (CNN)")

    if freeze_cfg.get("encoder", True):
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        print("Frozen: encoder (all layers)")

    if freeze_cfg.get("decoder", False):
        for param in model.model.decoder.parameters():
            param.requires_grad = False
        print("Frozen: decoder (all layers)")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable:,} trainable / {total:,} total ({trainable/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Whisper Decoder-only Fine-Tuning")
    parser.add_argument("--config", "-c", required=True, help="Training config JSON")
    parser.add_argument("--dataset", "-d", required=True, help="HF dataset directory")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    model_id = config["model_id"]
    language = config["language"]
    task = config["task"]
    train_cfg = config["training"]

    print(f"=== Whisper Decoder-only FT ===")
    print(f"Model: {model_id}")
    print(f"Strategy: {config.get('strategy', 'decoder-only-ft')}")
    print(f"Language: {language}, Task: {task}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # Force generation config
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    # Freeze encoder
    freeze_model(model, config)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_from_disk(args.dataset)

    # Process dataset
    print("Processing audio -> mel features...")
    dataset = dataset.map(
        lambda b: prepare_dataset(b, processor, language, task),
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )

    # Experience Replay: mix in general data to prevent forgetting
    replay_ds = load_replay_dataset(config, processor, language, task)
    if replay_ds is not None:
        replay_ratio = config["experience_replay"]["ratio"]
        n_replay = int(len(dataset["train"]) * replay_ratio)
        replay_subset = replay_ds.shuffle(seed=42).select(range(min(n_replay, len(replay_ds))))
        dataset["train"] = concatenate_datasets([dataset["train"], replay_subset])
        dataset["train"] = dataset["train"].shuffle(seed=42)
        print(f"Experience Replay: added {len(replay_subset)} samples ({replay_ratio*100:.0f}%)")

    print(f"Train: {len(dataset['train'])} samples")
    print(f"Eval: {len(dataset['eval'])} samples")

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Metrics: CER (character error rate — primary metric for Japanese)
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        **train_cfg,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # Train
    print("\n=== Training Start ===")
    trainer.train(resume_from_checkpoint=args.resume)

    # Save
    print("\n=== Saving best model ===")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Final eval
    print("\n=== Final Evaluation ===")
    metrics = trainer.evaluate()
    print(f"Final CER: {metrics.get('eval_cer', 'N/A')}")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    print(f"\nModel saved to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
