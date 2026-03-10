#!/usr/bin/env python3
"""Whisper Encoder+Decoder LoRA Fine-Tuning for personal voice adaptation.

Applies LoRA to both encoder and decoder, keeping feature_extractor (CNN) frozen.
This enables acoustic adaptation (encoder) + language adaptation (decoder).

Usage:
    python train_lora.py \
        --config configs/lora_v3_turbo.json \
        --dataset data/hf_dataset \
        --output_dir output/whisper-personal-lora

Requires: transformers, datasets, evaluate, jiwer, peft
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    """Load and prepare experience replay dataset."""
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


def setup_lora(model, config: dict):
    """Apply LoRA adapters to encoder and decoder."""
    lora_cfg = config["lora"]

    # Freeze feature extractor (CNN layers) before LoRA
    freeze_cfg = config.get("freeze", {})
    if freeze_cfg.get("feature_extractor", True):
        for param in model.model.encoder.conv1.parameters():
            param.requires_grad = False
        for param in model.model.encoder.conv2.parameters():
            param.requires_grad = False
        print("Frozen: feature_extractor (CNN)")

    # Configure LoRA
    target_modules = lora_cfg.get("target_modules", "all-linear")
    if isinstance(target_modules, str) and target_modules != "all-linear":
        target_modules = target_modules.split(",")

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias=lora_cfg.get("bias", "none"),
        modules_to_save=lora_cfg.get("modules_to_save", None),
    )

    print(f"\nLoRA Config:")
    print(f"  r={lora_config.r}, alpha={lora_config.lora_alpha}, "
          f"dropout={lora_config.lora_dropout}")
    print(f"  target_modules={lora_config.target_modules}")
    print(f"  modules_to_save={lora_config.modules_to_save}")

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def main():
    parser = argparse.ArgumentParser(description="Whisper LoRA Fine-Tuning")
    parser.add_argument("--config", "-c", required=True, help="Training config JSON")
    parser.add_argument("--dataset", "-d", required=True, help="HF dataset directory")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA weights into base model after training")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    model_id = config["model_id"]
    language = config["language"]
    task = config["task"]
    train_cfg = config["training"]

    print(f"=== Whisper LoRA Fine-Tuning ===")
    print(f"Model: {model_id}")
    print(f"Strategy: {config.get('strategy', 'encoder-decoder-lora')}")
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

    # Apply LoRA
    model = setup_lora(model, config)

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

    # Experience Replay
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

    # Metrics
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

    # Save LoRA adapter
    print("\n=== Saving LoRA adapter ===")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Optionally merge LoRA into base model
    if args.merge:
        print("\n=== Merging LoRA into base model ===")
        merged_dir = args.output_dir + "-merged"
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)
        print(f"Merged model saved to: {merged_dir}")

    # Final eval
    print("\n=== Final Evaluation ===")
    metrics = trainer.evaluate()
    print(f"Final CER: {metrics.get('eval_cer', 'N/A')}")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    print(f"\nLoRA adapter saved to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
