from csm_mlx import CSM, csm_1b, generate
from csm_mlx.finetune.lora import apply_lora_to_model, load_lora_weights
import audiofile
import numpy as np
import os
import sys
import json
import glob
from pathlib import Path
from huggingface_hub import hf_hub_download

# Get arguments from command line
if len(sys.argv) > 1:
    lora_weights_path = sys.argv[1]
    print(f"Using LoRA weights from command line: {lora_weights_path}")
else:
    lora_weights_path = "lora_finetune_output/lora_ckpt_step_100.safetensors/model.safetensors"
    print(f"No weights path provided, using default: {lora_weights_path}")

text_to_generate = sys.argv[2] if len(sys.argv) > 2 else "Hello there! This is my finetuned voice. How are you doing today?"
speaker_id = int(sys.argv[3]) if len(sys.argv) > 3 else 99

# 1. Initialize the base model
print("Initializing model...")
model = CSM(csm_1b())
# Download base model weights from Hugging Face
print("Downloading base model weights...")
weight_path = hf_hub_download(repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors")
model.load_weights(weight_path)
print("Base model loaded successfully")

# 2. Apply LoRA structure (with same parameters used during training)
print("Applying LoRA structure...")
model = apply_lora_to_model(
    model,
    rank=8,  # use same rank as in training
    alpha=16.0,  # use same alpha as in training
    target_modules=["attn", "codebook0_head", "projection"]  # same as training
)

# 3. Load your finetuned LoRA weights
# Check if file exists and provide helpful error message if not
if not os.path.exists(lora_weights_path):
    print(f"Error: LoRA weights file not found at {lora_weights_path}")
    sys.exit(1)
try:
    print(f"Loading finetuned weights from {lora_weights_path}...")
    model = load_lora_weights(model, lora_weights_path, load_embeddings=True)
    print("Finetuned weights loaded successfully!")
except Exception as e:
    print(f"Error loading finetuned weights: {str(e)}")
    print(f"File exists: {os.path.exists(lora_weights_path)}")
    print(f"File size: {os.path.getsize(lora_weights_path) if os.path.exists(lora_weights_path) else 'N/A'} bytes")
    print("The file may be corrupted or not a valid safetensors file")
    sys.exit(1)

# 4. Generate audio with the finetuned model
print(f'Generating audio for text: "{text_to_generate}"')
print(f'Using speaker_id: {speaker_id}')
audio = generate(
    model,
    text=text_to_generate,
    speaker=speaker_id,
    context=[],
    max_audio_length_ms=10000
)

# Save the generated audio
output_file = f"output/finetuned_output_speaker{speaker_id}.wav"
# Convert MLX array to NumPy only for file saving
audio_np = np.asarray(audio)
audiofile.write(output_file, audio_np, 24000)
print(f"Generated audio saved to {output_file}")
print(f"Text: \"{text_to_generate}\"")
print(f"Speaker ID: {speaker_id}")