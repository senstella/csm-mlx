#!/usr/bin/env python3
"""Finetune CSM models on custom text-audio datasets."""

import argparse
import os

import mlx.optimizers as optim
from huggingface_hub import hf_hub_download

from csm_mlx import CSM, csm_1b
from csm_mlx.finetune.dataset import CSMDataset
from csm_mlx.finetune.trainer import CSMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune CSM models")

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
        "--learning-rate", type=float, default=1e-5, help="Learning rate for optimizer"
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
        default="./finetune_output",
        help="Directory to save checkpoints and logs",
    )

    # Optimization parameters
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone parameters during finetuning",
    )
    parser.add_argument(
        "--freeze-decoder",
        action="store_true",
        help="Freeze decoder parameters during finetuning",
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

    # Optionally freeze parts of the model
    if args.freeze_backbone:
        print("Freezing backbone parameters...")
        model.backbone.freeze()

    if args.freeze_decoder:
        print("Freezing decoder parameters...")
        model.decoder.freeze()

    # Initialize optimizer
    print(
        f"Initializing {args.optimizer} optimizer with lr={args.learning_rate}, weight_decay={args.weight_decay}"
    )
    if args.optimizer == "adam":
        optimizer = optim.Adam(
            learning_rate=args.learning_rate,  # Adam doesn't have weight_decay
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            learning_rate=args.learning_rate, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            learning_rate=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Initialize trainer
    trainer = CSMTrainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=args.output_dir,
        save_every=args.save_every,
        log_every=args.log_every,
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
    print(f"Starting training for {args.epochs} epochs, batch size {args.batch_size}")
    _training_history = trainer.train(
        dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    print("Training complete!")
    print(f"Final checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
