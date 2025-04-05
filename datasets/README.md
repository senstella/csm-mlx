# Sample Dataset for CSM-MLX Finetuning

This directory contains a sample dataset structure for finetuning CSM-MLX models.

## Dataset Structure

The sample dataset consists of:
- `sample_dataset.json`: JSON file containing text-audio pairs
- `audio/`: Directory where audio files should be placed

Each entry in the JSON file includes:
- `text`: The transcript of the audio
- `audio_path`: Path to the audio file
- `speaker_id`: Numeric ID to identify different speakers (important for voice cloning)

## How to Use This Dataset

1. **Replace with Real Audio**: 
   - Replace the placeholder files in the `audio/` directory with real audio recordings
   - Each audio file should be a WAV file sampled at 24kHz (mono)
   - The recordings should be clear and of good quality

2. **Update the JSON**:
   - Make sure the text accurately transcribes the audio content
   - Update paths if necessary to match your actual audio files
   - Assign appropriate speaker IDs (0, 1, 2, etc.) for different speakers

## Creating Your Own Dataset

### Recording Audio

For best results:
- Use a good quality microphone
- Record in a quiet environment with minimal background noise
- Speak clearly and at a natural pace
- Keep a consistent distance from the microphone
- Aim for 5-10 minutes of audio per speaker for basic adaptation, more for better results

### Processing Audio

All audio files should be:
- WAV format
- 24kHz sampling rate
- Mono channel
- Properly trimmed (minimal silence at start/end)

You can use tools like Audacity or FFmpeg to convert and process your audio:

```bash
# Example FFmpeg command to convert audio to the right format
ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav
```

### Using the Dataset Creation Script

Use the included script to create a dataset from your audio files with matching text files:

```bash
python -m csm_mlx.finetune.create_dataset \
  --audio-dir /path/to/your/audio/files \
  --output-json your_dataset.json
```

This script will automatically look for matching `.normalized.txt` or `.original.txt` files with the same basename as your audio files.

## Manual Dataset Creation

You can also create your dataset manually as a JSON file:
```json
[
  {
    "text": "This is an example.",
    "audio_path": "/path/to/audio1.wav",
    "speaker_id": 0
  },
  {
    "text": "Another example with a different speaker.",
    "audio_path": "/path/to/audio2.wav",
    "speaker_id": 1
  }
]
```

## Finetuning with this Dataset

### LoRA Finetuning (Recommended)

LoRA finetuning is memory-efficient and works well even with limited data:

```bash
# Basic LoRA finetuning
python -m csm_mlx.finetune.finetune_lora \
  --data-path datasets/sample_dataset.json \
  --output-dir ./lora_finetune_output \
  --batch-size 4 \
  --epochs 5 \
  --learning-rate 5e-4 \
  --target-modules attn codebook0_head projection
```

For better results but higher memory usage, include direct embedding training:

```bash
# LoRA with direct embedding training
python -m csm_mlx.finetune.finetune_lora \
  --data-path datasets/sample_dataset.json \
  --output-dir ./lora_finetune_output \
  --batch-size 4 \
  --epochs 5 \
  --learning-rate 5e-4 \
  --target-modules attn codebook0_head projection \
  --train-embeddings
```

### Full Finetuning

Full finetuning updates all model parameters but requires more data and memory:

```bash
python -m csm_mlx.finetune.finetune \
  --data-path datasets/sample_dataset.json \
  --output-dir ./finetune_output \
  --batch-size 4 \
  --epochs 5 \
  --learning-rate 1e-5
```

## Memory Usage Considerations

If you face memory issues, try these approaches (from lowest to highest memory usage):

1. **LoRA for everything without embedding training**:
   ```bash
   --target-modules attn codebook0_head projection text_embeddings audio_embeddings
   ```

2. **LoRA with direct embedding training**:
   ```bash
   --target-modules attn codebook0_head projection --train-embeddings
   ```

3. **Full finetuning** (highest memory usage):
   ```bash
   python -m csm_mlx.finetune.finetune
   ```

For most use cases, option 2 provides the best balance of quality and memory efficiency.

## Testing Your Finetuned Model

After training, you can test your model using the provided example scripts:

```bash
# For LoRA finetuned models
python examples/example_finetuned_lora.py lora_finetune_output/lora_ckpt_epoch_5.safetensors "Your custom text here" 0

# For fully finetuned models
python examples/example_finetuned_full.py finetune_output/ckpt_epoch_5.safetensors "Your custom text here" 0
```

The last parameter is the speaker ID, which should match the ID used during training. 