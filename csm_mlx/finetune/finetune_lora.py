#!/usr/bin/env python3
"""Finetune CSM models using LoRA (Low-Rank Adaptation)."""

import argparse
import os

import mlx.optimizers as optim
from huggingface_hub import hf_hub_download

from csm_mlx import CSM, csm_1b
from csm_mlx.finetune.dataset import CSMDataset
from csm_mlx.finetune.trainer import LoRATrainer
from csm_mlx.finetune.utils import linear_to_lora_layers


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA finetuning for CSM models")

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="1b",
        help="Model size to use (currently only 1b is supported)",
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        help="Path to pretrained model weights. If not provided, will download from HF.",
    )
    parser.add_argument(
        "--resume-from", type=str, help="Path to checkpoint to resume training from"
    )

    # LoRA parameters
    parser.add_argument(
        "--lora-rank", type=int, default=8, help="Rank of LoRA matrices"
    )
    parser.add_argument(
        "--lora-alpha", type=float, default=16.0, help="Alpha scaling factor for LoRA"
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=["attn", "codebook0_head", "projection"],
        help="Module names to apply LoRA to",
    )
    parser.add_argument(
        "--train-embeddings",
        action="store_true",
        help="Train embedding layers directly (not via LoRA). Note: If you enable this, "
        "do not include embedding layers in --target-modules as this will cause conflicts.",
    )

    # Dataset parameters
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to JSON dataset file with text/audio pairs",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use from dataset",
    )

    # Training parameters
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--save-every", type=int, default=100, help="Save checkpoints every N steps"
    )
    parser.add_argument(
        "--log-every", type=int, default=10, help="Log metrics every N steps"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lora_finetune_output",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "sgd", "adamw"],
        help="Optimizer to use for training",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check for conflicting settings
    embedding_targets = [t for t in args.target_modules if "embeddings" in t]
    if args.train_embeddings and embedding_targets:
        print(
            "Warning: Both --train-embeddings and embedding modules in --target-modules detected"
        )
        print(f"Embedding modules in target_modules: {embedding_targets}")
        print(
            "This may cause conflicts. Removing embedding modules from target_modules"
        )
        args.target_modules = [t for t in args.target_modules if "embeddings" not in t]
        print(f"Updated target_modules: {args.target_modules}")

    # Initialize model
    print("Initializing model...")
    if args.model == "1b":
        model_args = csm_1b()
    else:
        raise ValueError(f"Unsupported model size: {args.model}")

    model = CSM(model_args)

    # Load pretrained weights
    if args.pretrained_path:
        print(f"Loading pretrained weights from {args.pretrained_path}")
        model.load_weights(args.pretrained_path)
    else:
        print("Downloading pretrained weights from Hugging Face...")
        weight_path = hf_hub_download(
            repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors"
        )
        model.load_weights(weight_path)

    # Apply LoRA to the model
    print(f"Applying LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"Target modules: {args.target_modules}")
    model = linear_to_lora_layers(
        model,
        config={
            "rank": args.lora_rank,
            "scale": args.lora_alpha / args.lora_rank,
            "dropout": 0.0,  # TODO: Add dropout rate
            "keys": args.target_modules,
        },
    )

    # Initialize optimizer
    print(f"Initializing {args.optimizer} optimizer with lr={args.learning_rate}")
    if args.weight_decay > 0:
        print(f"Using weight decay of {args.weight_decay}")

    if args.optimizer == "adam":
        optimizer = optim.Adam(learning_rate=args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(learning_rate=args.learning_rate)
    elif args.optimizer == "adamw":
        # AdamW is Adam with weight decay built in
        optimizer = optim.AdamW(
            learning_rate=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Initialize trainer
    trainer = LoRATrainer(
        model=model,  # type: ignore
        optimizer=optimizer,
        checkpoint_dir=args.output_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        train_embeddings=args.train_embeddings,
    )

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = CSMDataset.from_json(
        args.data_path,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(dataset)} samples")

    # Start training
    print(
        f"Starting LoRA training for {args.epochs} epochs, batch size {args.batch_size}"
    )
    _training_history = trainer.train(
        dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    print("Training complete!")
    print(f"Final checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
