#!/usr/bin/env python3
"""Prepare audio files for Whisper fine-tuning.

Scans a directory of WAV files, filters by duration, and exports
a structured JSONL manifest for the gold transcript pipeline.

Usage:
    python prepare_audio.py \
        --input-dir data/audio/ \
        --output data/audio_manifest.jsonl \
        [--min-duration 5.0] [--max-duration 30.0]

Input: A directory containing WAV files (16kHz mono recommended).
Output: JSONL with {id, wav_path, duration} per file.
"""

import argparse
import json
import subprocess
from pathlib import Path


def get_audio_duration(wav_path: str) -> float | None:
    """Get duration of a WAV file using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries",
             "format=duration", "-of", "csv=p=0", wav_path],
            capture_output=True, text=True, timeout=10
        )
        val = result.stdout.strip()
        return float(val) if val and val != "N/A" else None
    except (subprocess.TimeoutExpired, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare audio files for Whisper fine-tuning")
    parser.add_argument("--input-dir", "-i", required=True,
                        help="Directory containing WAV files")
    parser.add_argument("--output", "-o", default="data/audio_manifest.jsonl",
                        help="Output JSONL manifest path")
    parser.add_argument("--min-duration", type=float, default=5.0,
                        help="Minimum duration in seconds (default: 5.0)")
    parser.add_argument("--max-duration", type=float, default=30.0,
                        help="Maximum duration in seconds (default: 30.0)")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively search subdirectories")
    parser.add_argument("--extensions", nargs="+",
                        default=[".wav", ".flac", ".mp3", ".m4a"],
                        help="Audio file extensions to include")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find audio files
    audio_files = []
    for ext in args.extensions:
        if args.recursive:
            audio_files.extend(input_dir.rglob(f"*{ext}"))
        else:
            audio_files.extend(input_dir.glob(f"*{ext}"))
    audio_files = sorted(audio_files)

    print(f"Found {len(audio_files)} audio files in {input_dir}")

    stats = {"total": 0, "accepted": 0,
             "too_short": 0, "too_long": 0, "error": 0}

    with open(output_path, "w") as out:
        for audio_path in audio_files:
            stats["total"] += 1
            duration = get_audio_duration(str(audio_path))

            if duration is None:
                stats["error"] += 1
                continue

            if duration < args.min_duration:
                stats["too_short"] += 1
                continue

            if duration > args.max_duration:
                stats["too_long"] += 1
                continue

            record = {
                "id": audio_path.stem,
                "wav_path": str(audio_path.resolve()),
                "duration": round(duration, 2),
                "date": "",  # Fill in if you have recording dates
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["accepted"] += 1

            if stats["accepted"] % 100 == 0:
                print(f"  Processed {stats['accepted']}...")

    total_hours = 0
    # Re-read to compute total hours
    with open(output_path) as f:
        for line in f:
            rec = json.loads(line)
            total_hours += rec["duration"] / 3600

    print(f"\n=== Audio Preparation Complete ===")
    print(f"Total files scanned: {stats['total']}")
    print(f"Accepted: {stats['accepted']} ({total_hours:.1f}h)")
    print(f"Too short (<{args.min_duration}s): {stats['too_short']}")
    print(f"Too long (>{args.max_duration}s): {stats['too_long']}")
    print(f"Errors: {stats['error']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
