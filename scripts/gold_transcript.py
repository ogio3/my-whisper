#!/usr/bin/env python3
"""Generate gold-standard transcripts using Gemini multimodal.

Reads an audio manifest (JSONL) and re-transcribes audio using
Gemini's multimodal audio understanding with vocabulary injection.

Usage:
    python gold_transcript.py \
        --input data/audio_manifest.jsonl \
        --output data/gold_transcripts.jsonl \
        [--limit 100] [--batch-size 5]

Requires: GOOGLE_GENAI_API_KEY environment variable or .env file.
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

# ============================================================
# CUSTOMIZE THIS: Add your own vocabulary for context injection.
# This dramatically improves accuracy for domain-specific terms,
# proper nouns, and technical jargon.
# ============================================================
VOCABULARY = [
    # Technical terms (examples — replace with your own)
    "Whisper", "fine-tune", "LoRA", "encoder", "decoder",
    "WER", "CER", "transformer", "attention",
    # Product/project names you frequently mention
    # "MyApp", "ProjectX",
    # People/place names you frequently mention
    # "Tanaka", "Tokyo",
]

SYSTEM_PROMPT = """You are an expert audio transcriber. Transcribe the following audio exactly as spoken.

Rules:
1. Keep fillers (um, uh, well, etc.) as-is
2. Keep false starts and self-corrections as-is
3. When the speaker mixes languages, keep each word in its original language
4. Add natural punctuation
5. For Japanese: keep technical terms in their original language (English terms stay English)

Speaker's vocabulary (pay attention to these proper nouns):
{vocabulary}

Output: Transcription text only. No explanations or metadata."""


def transcribe_with_gemini(wav_path: str, existing_text: str = "") -> dict:
    """Transcribe audio using Gemini multimodal."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai required: pip install google-generativeai")

    # Load API key from .env if not already set
    if not os.environ.get("GOOGLE_GENAI_API_KEY"):
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GOOGLE_GENAI_API_KEY="):
                    os.environ["GOOGLE_GENAI_API_KEY"] = (
                        line.split("=", 1)[1].strip().strip('"'))

    api_key = os.environ.get("GOOGLE_GENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_GENAI_API_KEY not set. "
            "Copy .env.example to .env and add your API key.")

    genai.configure(api_key=api_key)

    with open(wav_path, "rb") as f:
        audio_data = f.read()

    audio_b64 = base64.b64encode(audio_data).decode()

    prompt = SYSTEM_PROMPT.format(vocabulary=", ".join(VOCABULARY))
    if existing_text:
        prompt += (f"\n\nReference transcript (may contain errors):"
                   f"\n{existing_text}")

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
    model = genai.GenerativeModel(model_name)
    response = model.generate_content([
        prompt,
        {"mime_type": "audio/wav", "data": audio_b64}
    ])

    return {
        "gold_text": response.text.strip(),
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# Hallucination detection threshold.
# Japanese speech: ~5-8 chars/sec normal, up to 12 for fast speakers.
# English speech: adjust to ~15-20 chars/sec.
MAX_CHARS_PER_SEC = 12


def detect_hallucination(gold_text: str, duration: float) -> bool:
    """Detect LLM hallucination by chars-per-second heuristic.

    When the LLM "hallucinates", it often generates far more text
    than could have been spoken in the audio's duration.
    """
    if duration <= 0:
        return True
    ratio = len(gold_text) / duration
    return ratio > MAX_CHARS_PER_SEC


def main():
    parser = argparse.ArgumentParser(
        description="Generate gold transcripts with Gemini")
    parser.add_argument("--input", "-i", required=True,
                        help="Input JSONL from prepare_audio.py")
    parser.add_argument("--output", "-o", required=True,
                        help="Output JSONL with gold transcripts")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of files (0=all)")
    parser.add_argument("--min-duration", type=float, default=5.0,
                        help="Min duration in seconds")
    parser.add_argument("--max-duration", type=float, default=30.0,
                        help="Max duration in seconds")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Sleep between API calls (seconds)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing if resuming
    done_ids = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec.get("id"))
                except json.JSONDecodeError:
                    pass
        print(f"Resuming: {len(done_ids)} already done")

    # Load input records
    records = []
    with open(input_path) as f:
        for line in f:
            rec = json.loads(line)
            dur = rec.get("duration", 0)
            if (rec["id"] not in done_ids
                    and args.min_duration <= dur <= args.max_duration):
                records.append(rec)

    if args.limit:
        records = records[:args.limit]

    print(f"Processing {len(records)} records...")

    mode = "a" if args.resume else "w"
    processed = 0
    errors = 0

    with open(output_path, mode) as out:
        for i, rec in enumerate(records):
            try:
                result = transcribe_with_gemini(
                    wav_path=rec["wav_path"],
                    existing_text=rec.get("text", ""),
                )

                gold_text = result["gold_text"]
                hallucinated = detect_hallucination(
                    gold_text, rec["duration"])

                output_rec = {
                    "id": rec["id"],
                    "date": rec.get("date", ""),
                    "wav_path": rec["wav_path"],
                    "duration": rec["duration"],
                    "original_text": rec.get("text", ""),
                    "gold_text": gold_text,
                    "model": result["model"],
                    "transcribed_at": result["timestamp"],
                    "hallucinated": hallucinated,
                    "chars_per_sec": round(
                        len(gold_text) / max(rec["duration"], 0.1), 1),
                }
                out.write(json.dumps(output_rec, ensure_ascii=False) + "\n")
                out.flush()
                processed += 1

                if hallucinated:
                    print(f"  HALLUCINATION: {rec['id'][:30]} "
                          f"({rec['duration']:.1f}s -> "
                          f"{len(gold_text)} chars, "
                          f"{output_rec['chars_per_sec']} c/s)")

                if processed % 10 == 0:
                    print(f"  Processed {processed}/{len(records)} "
                          f"(errors: {errors})")

            except Exception as e:
                errors += 1
                print(f"  Error on {rec['id']}: {e}")

            time.sleep(args.sleep)

    print(f"\n=== Gold Transcript Complete ===")
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
