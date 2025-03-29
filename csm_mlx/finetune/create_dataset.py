#!/usr/bin/env python3
"""
Create a dataset.json file from audio files and transcripts.

This script scans the audio directory, locates corresponding transcript files,
and generates a properly formatted dataset.json file for finetuning.
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset.json from audio files and transcripts")
    parser.add_argument(
        "--audio-dir", type=str, default="finetune/datasets/audio",
        help="Directory containing audio files and their transcript files"
    )
    parser.add_argument(
        "--output-json", type=str, default="csm_mlx/finetune/dataset.json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--default-speaker", type=int, default=0,
        help="Default speaker ID for all samples"
    )
    parser.add_argument(
        "--transcript-suffix", type=str, default="normalized.txt",
        help="Suffix for transcript files (normalized.txt or original.txt)"
    )
    parser.add_argument(
        "--audio-ext", type=str, default="wav",
        help="Audio file extension to look for"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to include (useful for testing)"
    )
    return parser.parse_args()


def find_matching_transcript(audio_file, transcript_suffix="normalized.txt"):
    """Find the transcript file matching the audio file."""
    # Get the base filename without extension
    base_file = os.path.splitext(audio_file)[0]
    
    # Check for matching transcript file
    transcript_file = f"{base_file}.{transcript_suffix}"
    
    if os.path.exists(transcript_file):
        return transcript_file
    
    return None


def read_transcript_file(transcript_file):
    """Read content from transcript file."""
    with open(transcript_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        # Remove quotes if present
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text


def create_dataset_from_directory(audio_dir, transcript_suffix="normalized.txt", default_speaker=0, audio_ext="wav", max_samples=None):
    """Create dataset entries from audio files and matching transcript files."""
    print(f"Scanning directory: {audio_dir}")
    
    # Find all audio files
    audio_files = glob.glob(os.path.join(audio_dir, f"*.{audio_ext}"))
    print(f"Found {len(audio_files)} audio files")
    
    # Create dataset entries
    dataset = []
    missing_transcripts = []
    
    for audio_path in sorted(audio_files):
        # Find matching transcript file
        transcript_file = find_matching_transcript(audio_path, transcript_suffix)
        
        if transcript_file and os.path.exists(transcript_file):
            text = read_transcript_file(transcript_file)
            entry = {
                "text": text,
                "audio_path": audio_path,
                "speaker_id": default_speaker
            }
            dataset.append(entry)
            
            # Break if we've reached the maximum samples
            if max_samples and len(dataset) >= max_samples:
                print(f"Reached maximum sample count of {max_samples}")
                break
        else:
            missing_transcripts.append(os.path.basename(audio_path))
    
    # Report results
    if missing_transcripts:
        print(f"Warning: {len(missing_transcripts)} audio files have no matching transcript and will be skipped:")
        for filename in missing_transcripts[:5]:
            print(f"  - {filename}")
        if len(missing_transcripts) > 5:
            print(f"  - ... and {len(missing_transcripts) - 5} more")
    
    print(f"Created dataset with {len(dataset)} entries")
    return dataset


def main():
    args = parse_args()
    
    # Create dataset
    dataset = create_dataset_from_directory(
        audio_dir=args.audio_dir,
        transcript_suffix=args.transcript_suffix,
        default_speaker=args.default_speaker,
        audio_ext=args.audio_ext,
        max_samples=args.max_samples
    )
    
    if not dataset:
        print("Error: No valid entries found for the dataset!")
        sys.exit(1)
    
    # Write dataset to JSON file
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset saved to {args.output_json}")
    print("\nYou can now use this dataset for finetuning:")
    print(f"  python -m csm_mlx.finetune.finetune_lora --data-path {args.output_json}")
    print(f"\nExample command to finetune with LoRA:")
    print(f"  python -m csm_mlx.finetune.finetune_lora --data-path {args.output_json} --lora-rank 8 --epochs 5 --learning-rate 5e-4")


if __name__ == "__main__":
    main() 