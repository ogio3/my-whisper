# Whisper Personal — Your Own Whisper for $5

Fine-tune OpenAI's Whisper to **your voice** and cut transcription errors by 75%.

This repository provides a complete, reproducible pipeline for personalizing
Whisper Large V3 Turbo using **decoder-only fine-tuning** — the approach that
outperforms both LoRA and full fine-tuning on noisy audio
([NeurIPS 2024 STAR](https://arxiv.org/abs/2406.06619)).

## Results

Evaluated on 320 samples of personal voice recordings (AirPods Bluetooth SCO, 8–16kHz):

| Model | CER | vs Baseline |
|-------|-----|-------------|
| Whisper V3 Turbo (original) | 20.59% | — |
| **Fine-tuned v1** (10h data) | 10.27% | -10.3pt (50% better) |
| **Fine-tuned v2** (17.8h data) | **5.17%** | **-15.4pt (75% better)** |

> CER = Character Error Rate. Primary metric for Japanese ASR.

### Qualitative Improvements

| Category | Baseline Output | Fine-tuned Output |
|----------|----------------|-------------------|
| English proper nouns | カタカナ化される | Keeps original English |
| Technical terms | http**濃度** (garbled) | HTTP**ノード** (correct) |
| Fillers preserved | (dropped) | えー、えっと... (kept) |
| Punctuation | (none) | Natural 、。 placement |
| Kanji proper nouns | ひらがな only | Correct kanji rendering |

### Cost & Time

| Item | v1 | v2 |
|------|----|----|
| Training data | 10h (2,874 samples) | 17.8h (3,886 samples) |
| GPU | Cloud GPU | RTX 5090 32GB |
| Training time | — | 8 min (5 epochs) |
| GPU cost | — | ~$1.50 |
| Gold transcripts | Gemini 2.5 Pro API | Gemini 2.5 Pro API |
| **Total cost** | ~$5 | ~$5 |

## How It Works

```
[Your WAV files]
     │
     ├── 1. prepare_audio.py    →  Filter by duration (5-30s)
     │
     ├── 2. gold_transcript.py  →  Gemini re-transcribes with your vocabulary
     │      └── Hallucination detection (>12 chars/sec → flagged)
     │
     ├── 3. build_hf_dataset.py →  HF Dataset (temporal train/eval split)
     │
     ├── 4. train_decoder_ft.py →  Decoder-only fine-tuning on GPU
     │      └── Encoder frozen, decoder all params trained
     │
     └── 5. eval_model.py       →  CER/WER comparison vs baseline
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/oginome/whisper-personal.git
cd whisper-personal
pip install -r requirements.txt
```

### 2. Prepare Your Audio

Place your WAV files (16kHz mono recommended) in `data/audio/`:

```bash
python scripts/prepare_audio.py \
    --input-dir data/audio/ \
    --output data/audio_manifest.jsonl \
    --min-duration 5.0 --max-duration 30.0
```

**How much audio do you need?**
- 5-10 hours: noticeable improvement (~50% CER reduction)
- 15-20 hours: excellent results (~75% CER reduction)
- Segments between 5-30 seconds work best

### 3. Generate Gold Transcripts

Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_GENAI_API_KEY

python scripts/gold_transcript.py \
    --input data/audio_manifest.jsonl \
    --output data/gold_transcripts.jsonl \
    --resume  # Resume if interrupted
```

**Important:** Edit `VOCABULARY` in `gold_transcript.py` with your own
proper nouns, technical terms, and frequently used words. This dramatically
improves transcript accuracy.

### 4. Build Dataset

```bash
python scripts/build_hf_dataset.py \
    --input data/gold_transcripts.jsonl \
    --output data/hf_dataset
```

### 5. Train (GPU Required)

```bash
python scripts/train_decoder_ft.py \
    --config configs/decoder_ft_v3_turbo.json \
    --dataset data/hf_dataset \
    --output_dir output/whisper-personal
```

**GPU options:**
- **VAST.ai**: RTX 4090/5090, ~$0.30-0.50/hr. Training takes ~8 min for 4K samples.
- **Google Colab Pro**: T4/A100, sufficient for this workload.
- **Local**: Any GPU with 16GB+ VRAM.

### 6. Evaluate

```bash
python scripts/eval_model.py \
    --finetuned output/whisper-personal \
    --dataset data/hf_dataset \
    --output results/eval_report.json
```

## Training Strategy: Why Decoder-Only?

We freeze the **encoder** (audio feature extraction) and train only the
**decoder** (language model). This is counterintuitive but works because:

1. **Encoder already handles audio well** — Whisper was trained on 680K hours.
   Your 10-20h of personal audio won't improve acoustic features.
2. **Decoder needs vocabulary adaptation** — Your proper nouns, technical terms,
   and speech patterns live in the decoder's language model.
3. **NeurIPS 2024 evidence** — The STAR paper showed decoder-only FT achieves
   WER 4.4% vs LoRA 5.1% vs Full FT 5.2% on noisy audio (CHiME-4).

### Parameters

- **Model**: `openai/whisper-large-v3-turbo` (808M params)
- **Trainable**: 171M params (21.3%) — decoder only
- **Learning rate**: 1e-6 (Turbo has only 4 decoder layers — very sensitive to LR)
- **Precision**: BF16 (FP16 causes loss spikes with V3 family)
- **Eval metric**: CER, not val loss (val loss is unreliable for ASR — [HF #107](https://huggingface.co/openai/whisper-large-v3/discussions/107))

## Alternative: LoRA Fine-Tuning

If you want to also adapt the encoder (e.g., for unusual audio conditions),
use the LoRA script:

```bash
python scripts/train_lora.py \
    --config configs/lora_v3_turbo.json \
    --dataset data/hf_dataset \
    --output_dir output/whisper-personal-lora \
    --merge  # Merge LoRA into base model
```

## Pitfalls & Tips

### Data Quality is Everything

The single most important factor is **transcript accuracy**. A misaligned
transcript teaches the model wrong associations. That's why we use Gemini
for gold transcripts instead of relying on existing (often noisy) transcripts.

### Hallucination Detection

Gemini occasionally "hallucinates" — generating text that wasn't spoken.
We detect this with a simple heuristic: if the transcript has more than
12 characters per second of audio, it's flagged and excluded.

### Temporal Train/Eval Split

We split by recording date, not randomly. This prevents the same recording
session from appearing in both train and eval sets, which would inflate scores.

### Experience Replay

If you notice the model "forgets" general transcription ability, enable
experience replay in the config. This mixes 10% Common Voice data into
training to maintain general performance.

### Compatibility Notes

| Issue | Solution |
|-------|----------|
| MPS (Apple Silicon) inference | Use `float32` — `float16` causes type mismatch |
| transformers 5.x | Use `eval_strategy` (not `evaluation_strategy`), `processing_class` (not `tokenizer`) |
| Whisper V3 Mel bins | Always 128 bins. Never use a V2 processor with V3 model. |
| datasets 4.x | Requires torchcodec. Use `datasets<4.0` with soundfile backend instead. |

## File Structure

```
whisper-personal/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── configs/
│   ├── decoder_ft_v3_turbo.json    # Decoder-only FT config (recommended)
│   └── lora_v3_turbo.json          # LoRA config (alternative)
├── scripts/
│   ├── prepare_audio.py            # WAV directory → JSONL manifest
│   ├── gold_transcript.py          # Gemini gold transcription
│   ├── build_hf_dataset.py         # JSONL → HF Dataset
│   ├── train_decoder_ft.py         # Decoder-only fine-tuning
│   ├── train_lora.py               # LoRA fine-tuning (alternative)
│   ├── eval_model.py               # CER/WER evaluation
│   └── eval_lenient.py             # Multi-level CER evaluation
├── data/                           # (gitignored)
│   ├── audio/                      # Your WAV files
│   ├── audio_manifest.jsonl        # From prepare_audio.py
│   ├── gold_transcripts.jsonl      # From gold_transcript.py
│   └── hf_dataset/                 # From build_hf_dataset.py
├── output/                         # (gitignored)
│   └── whisper-personal/           # Trained model (~1.5GB)
└── results/                        # (gitignored)
    └── eval_report.json            # Evaluation results
```

## References

- [Whisper Large V3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) — OpenAI, 2024
- [STAR: NeurIPS 2024](https://arxiv.org/abs/2406.06619) — Decoder-only FT outperforms LoRA and Full FT
- [HF Discussion #107](https://huggingface.co/openai/whisper-large-v3/discussions/107) — Val loss vs WER/CER divergence

## License

MIT — See [LICENSE](LICENSE) for details.
