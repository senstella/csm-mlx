# CSM-MLX Finetuning

This module provides functionality for finetuning CSM (Conversation Speech Model) on custom text-audio datasets. Finetuning allows you to adapt the model to specific speakers, domains, or styles.

## Prerequisites

You'll need the following additional dependencies:
```bash
pip install tqdm audiofile audresample
```

## Preparing Your Dataset

To finetune the model, you need to prepare a dataset of text and audio pairs. Each sample should have:
- Text transcript
- Audio file (24kHz WAV recommended)
- Speaker ID (optional, but highly recommended for voice cloning)

### Using the Dataset Creation Script

We provide a helper script to create a dataset from your audio files with matching text files:

```bash
python -m csm_mlx.finetune.create_dataset \
  --audio-dir /path/to/audio/files \
  --output-json dataset.json
```

This script will automatically look for matching `.normalized.txt` or `.original.txt` files with the same basename as your audio files.


### Manual Dataset Creation

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

## Finetuning Methods

### LoRA Finetuning (Recommended)

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that significantly reduces memory requirements and training time while maintaining quality. Instead of updating all weights, LoRA injects trainable rank decomposition matrices into specific layers.

```bash
python -m csm_mlx.finetune.finetune_lora \
  --data-path dataset.json \
  --output-dir ./lora_finetune_output \
  --batch-size 4 \
  --epochs 5 \
  --learning-rate 5e-4 \
  --lora-rank 8 \
  --optimizer adamw \
  --target-modules attn codebook0_head projection \
  --train-embeddings
```

#### LoRA Parameters

- `--lora-rank`: Rank of the low-rank matrices (typically 4-16; higher rank = more capacity)
- `--lora-alpha`: Scaling factor (default is 16.0, often set to 2× rank)
- `--target-modules`: Which modules to apply LoRA to (defaults to attention, codebook head, and projection)
- `--train-embeddings`: Trains the embedding layers directly (not via LoRA adapters)

> **Important**: Do not include embedding layers in `--target-modules` when using `--train-embeddings`. These are mutually exclusive approaches for embedding layers. The `--train-embeddings` flag trains the full embedding matrices directly, while including them in `target_modules` would apply LoRA to them instead.

#### Memory Usage Considerations

There are three approaches for training embeddings, with different memory requirements:

1. **LoRA for everything (lowest memory)**: Don't use `--train-embeddings`
   ```
   --target-modules attn codebook0_head projection
   ```

2. **LoRA + direct embedding training (medium memory)**: Use `--train-embeddings` and don't include embeddings in `--target-modules`
   ```
   --target-modules attn codebook0_head projection --train-embeddings
   ```

3. **Full finetuning (highest memory)**: Use `csm_mlx.finetune.finetune` instead of LoRA

For best results with reasonable memory usage, option 2 is recommended.

#### Advanced LoRA Options

```bash
python -m csm_mlx.finetune.finetune_lora \
  --data-path dataset.json \
  --output-dir ./lora_finetune_output \
  --batch-size 4 \
  --epochs 5 \
  --learning-rate 5e-4 \
  --lora-rank 16 \
  --lora-alpha 32.0 \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --save-every 100 \
  --log-every 10
  --target-modules attn codebook0_head projection \
  --train-embeddings
```

### Full Finetuning

Full finetuning updates all model parameters. This gives maximum flexibility but requires more data and computational resources.

```bash
python -m csm_mlx.finetune.finetune \
  --data-path dataset.json \
  --output-dir ./finetune_output \
  --batch-size 4 \
  --epochs 5 \
  --learning-rate 1e-5
```

## Using Your Finetuned Models

### Using a LoRA-Finetuned Model

```python
from csm_mlx import CSM, csm_1b, generate
from csm_mlx.finetune.lora import apply_lora_to_model, load_lora_weights
import audiofile
import numpy as np
from huggingface_hub import hf_hub_download

# Initialize the base model
model = CSM(csm_1b())
weight_path = hf_hub_download(repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors")
model.load_weights(weight_path)

# Apply LoRA structure (with same parameters as training)
# IMPORTANT: Don't include embedding layers here if you used --train-embeddings during training
model = apply_lora_to_model(
    model,
    rank=8,
    alpha=16.0,
    target_modules=["attn", "codebook0_head", "projection"]  # No embedding layers
)

# Load LoRA weights (and trained embeddings if used --train-embeddings)
model = load_lora_weights(model, "lora_finetune_output/lora_ckpt_epoch_5.safetensors")

# Generate audio
audio = generate(
    model,
    text="This is generated with my LoRA-finetuned model.",
    speaker=0,  # Make sure to use the same speaker ID from training
    context=[],
    max_audio_length_ms=5000
)

# Save audio
audiofile.write("finetuned_output.wav", np.asarray(audio), 24000)
```

### Example Script

For convenience, you can use the included example script:

```bash
# Basic usage with LoRA weights
python examples/example_finetuned_lora.py lora_finetune_output/lora_ckpt_epoch_5.safetensors "Your custom text here" 0
```

### Merging LoRA Weights for Deployment

For inference efficiency, you can merge LoRA weights back into the base model:

```python
from csm_mlx.finetune.lora import merge_lora_weights

# After loading LoRA weights as above
model = merge_lora_weights(model)

# Now the model has LoRA adaptations merged and runs at original speed
# Directly trained embeddings (if used --train-embeddings) are already applied
audio = generate(
    model,
    text="This is generated with merged weights.",
    speaker=0,
    context=[]
)
```

### Using a Fully Finetuned Model

```python
from csm_mlx import CSM, csm_1b, generate

# Initialize the model with finetuned weights
csm = CSM(csm_1b())
csm.load_weights("./finetune_output/ckpt_epoch_5.safetensors")

# Generate audio
audio = generate(
    csm,
    text="This is generated with my finetuned model.",
    speaker=0,
    context=[],
    max_audio_length_ms=5000
)
```

## Tips for Successful Finetuning

1. **Data Quality**: Use high-quality audio recordings with clear pronunciation.
2. **Data Quantity**: More data usually leads to better results. Aim for at least 10-20 minutes of audio per speaker.
3. **Speaker ID Consistency**: Make sure to use the same speaker IDs during both training and inference.
4. **Learning Rate**: 
   - For full finetuning: Use a small learning rate (1e-5 to 5e-5) to avoid catastrophic forgetting.
   - For LoRA: Use a higher learning rate (1e-4 to 1e-3) as fewer parameters are updated.
5. **Batch Size**: Use the largest batch size that fits in your memory.
6. **LoRA vs. Full Finetuning**:
   - For voice cloning with limited data: Use LoRA with rank 8-16
   - For comprehensive fine-tuning with lots of data: Use full fine-tuning
7. **Embedding Training**:
   - Using `--train-embeddings` often gives better results than applying LoRA to embeddings
   - If you run into memory issues, reduce batch size or don't use `--train-embeddings`
8. **Optimizer**: AdamW with weight decay works well for most cases. 