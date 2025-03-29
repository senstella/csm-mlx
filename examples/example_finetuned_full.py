#!/usr/bin/env python3
"""Example script for using a fully finetuned CSM model."""

import os
import sys
import json
import glob
from pathlib import Path
import audiofile
import numpy as np
from huggingface_hub import hf_hub_download

from csm_mlx import CSM, csm_1b, generate

# Get arguments from command line
if len(sys.argv) > 1:
    weights_path = sys.argv[1]
    print(f"Using finetuned weights from command line: {weights_path}")
else:
    weights_path = "finetune_output/ckpt_step_100.safetensors/model.safetensors"
    print(f"No weights path provided, using default: {weights_path}")

text_to_generate = sys.argv[2] if len(sys.argv) > 2 else "This is generated with my fully finetuned model."
speaker_id = int(sys.argv[3]) if len(sys.argv) > 3 else 99 

# Allow overriding speaker ID from command line
if len(sys.argv) > 4:
    try:
        speaker_id = int(sys.argv[4])
        print(f"Using speaker_id {speaker_id} from command line")
    except ValueError:
        print(f"Invalid speaker_id: {sys.argv[4]}, using {speaker_id}")

# Initialize the model
print("Initializing model...")
model = CSM(csm_1b())

# Check if file exists
if not os.path.exists(weights_path):
    print(f"Error: Weights file not found at {weights_path}")
    sys.exit(1)

# Load finetuned weights
try:
    print(f"Loading finetuned weights from {weights_path}...")
    model.load_weights(weights_path)
    print("Finetuned weights loaded successfully!")
except Exception as e:
    print(f"Error loading weights: {str(e)}")
    print(f"File exists: {os.path.exists(weights_path)}")
    print(f"File size: {os.path.getsize(weights_path) if os.path.exists(weights_path) else 'N/A'} bytes")
    print("The file may be corrupted or not a valid safetensors file")
    sys.exit(1)

# Generate audio with the finetuned model
print(f'Generating audio for text: "{text_to_generate}"')
print(f'Using speaker_id: {speaker_id}')
audio = generate(
    model,
    text=text_to_generate,
    speaker=speaker_id,
    context=[],
    max_audio_length_ms=5000
)

# Save the generated audio
output_file = f"output/finetuned_full_speaker{speaker_id}.wav"
# Convert MLX array to NumPy only for file saving
audio_np = np.asarray(audio)
audiofile.write(output_file, audio_np, 24000)
print(f"Generated audio saved to {output_file}")
print(f"Text: \"{text_to_generate}\"")
print(f"Speaker ID: {speaker_id}") 