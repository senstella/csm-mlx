#!/bin/bash
# Sample script to run the complete finetuning process

set -e  # Exit on error

# Create directories if they don't exist
mkdir -p finetune/datasets/audio

# Step 1: Generate sample audio files
echo "=== Step 1: Generating sample audio files ==="
python finetune/datasets/create_sample_audio.py

# Step 2: Run LoRA finetuning (recommended method)
echo ""
echo "=== Step 2: Running LoRA finetuning ==="
python -m finetune.finetune_lora \
  --data-path finetune/datasets/sample_dataset.json \
  --output-dir ./lora_finetune_output \
  --batch-size 2 \
  --epochs 3 \
  --lora-rank 8 \
  --learning-rate 5e-4

# Step 3: Generate audio with the finetuned model
echo ""
echo "=== Step 3: Generating sample output with finetuned model ==="
python -m examples.finetune_lora_example \
  --data-path finetune/datasets/sample_dataset.json \
  --output-dir ./lora_finetune_output \
  --prompt "This is a test of the finetuned voice model." \
  --epochs 0  # Skip additional training, just generate

echo ""
echo "Finetuning example complete!"
echo "Check the lora_finetune_output directory for results." 