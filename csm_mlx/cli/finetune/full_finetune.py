import os
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.optimizers as optim
import typer
from huggingface_hub import hf_hub_download
from typing_extensions import Annotated

from csm_mlx import CSM
from csm_mlx.cli.config import MODEL, Models, OptimizerChoice
from csm_mlx.finetune.dataset import CSMDataset
from csm_mlx.finetune.trainer import CSMTrainer, TrainArgs

app = typer.Typer(no_args_is_help=True)


@app.command("full")
def finetune_command(
    data_path: Annotated[
        Path,
        typer.Option(
            "--data-path",
            help="Path to JSON dataset file with text/audio pairs",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    model: Annotated[
        Models,
        typer.Option(
            "--model",
            "-m",
            help="Model size to use (currently only 1b is supported)",
        ),
    ] = Models._1b,
    pretrained_path: Annotated[
        Optional[Path],
        typer.Option(
            "--pretrained-path",
            help="Path to pretrained model weights. If not provided, will download from HF.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    max_audio_length_ms: Annotated[
        Optional[int],
        typer.Option(
            "--max-audio-length-ms",
            help="Maximum length of audio in milliseconds",
        ),
    ] = None,
    mask_speaker_ids: Annotated[
        Optional[List[int]],
        typer.Option(
            "--mask-speaker-ids",
            help="List of speaker IDs to mask in the output",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Batch size for training",
            min=1,
        ),
    ] = 4,
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            help="Number of epochs to train for",
            min=1,
        ),
    ] = 5,
    learning_rate: Annotated[
        float,
        typer.Option(
            "--learning-rate",
            "--lr",
            help="Learning rate for optimizer",
            min=0.0,
        ),
    ] = 1e-5,
    weight_decay: Annotated[
        float,
        typer.Option(
            "--weight-decay",
            "--wd",
            help="Weight decay for optimizer",
            min=0.0,
        ),
    ] = 1e-4,
    max_norm: Annotated[
        float,
        typer.Option(
            "--max-norm",
            help="Max norm for gradient clipping (0.0 to disable)",
            min=0.0,
        ),
    ] = 0.0,
    first_codebook_weight_multiplier: Annotated[
        float,
        typer.Option(
            "--first_codebook_weight_multiplier",
            help="Loss weight for zero-th codebook",
            min=0.0,
        ),
    ] = 1.0,
    ckpt_freq: Annotated[
        int,
        typer.Option(
            "--ckpt_freq",
            help="Save checkpoints every N steps",
            min=1,
        ),
    ] = 100,
    log_freq: Annotated[
        int,
        typer.Option(
            "--log-freq",
            help="Log metrics every N steps",
            min=1,
        ),
    ] = 10,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to save checkpoints and logs",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ] = Path("./finetune_output"),
    freeze_backbone: Annotated[
        bool,
        typer.Option(
            "--freeze-backbone/--no-freeze-backbone",
            help="Freeze backbone parameters during finetuning",
        ),
    ] = False,
    freeze_decoder: Annotated[
        bool,
        typer.Option(
            "--freeze-decoder/--no-freeze-decoder",
            help="Freeze decoder parameters during finetuning",
        ),
    ] = False,
    gradient_checkpointing: Annotated[
        bool,
        typer.Option(
            "--gradient-ckpt/--no-gradient-ckpt",
            help="Enable gradient checkpointing",
        ),
    ] = False,
    optimizer: Annotated[
        OptimizerChoice,
        typer.Option(
            "--optimizer",
            help="Optimizer to use for training",
            case_sensitive=False,
        ),
    ] = OptimizerChoice.ADAMW,
):
    """Full finetuning for CSM models."""
    print("Starting finetuning script...")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print("Initializing model...")
    model_config = MODEL[model.value]

    csm_model = CSM(model_config.get("config"))  # type: ignore
    mx.eval(csm_model.parameters())

    if pretrained_path:
        print(f"Loading pretrained weights from {pretrained_path}")
        csm_model.load_weights(str(pretrained_path))
    else:
        print("Using pretrained weights from Hugging Face...")
        weight = hf_hub_download(**model_config.get("loader"))  # type: ignore

        csm_model.load_weights(weight)

    if freeze_backbone:
        print("Freezing backbone parameters...")
        csm_model.backbone.freeze()

    if freeze_decoder:
        print("Freezing decoder parameters...")
        csm_model.decoder.freeze()

    mx.eval(csm_model.parameters())

    csm_model.train()

    print(
        f"Initializing {optimizer.value} optimizer with lr={learning_rate}, weight_decay={weight_decay}"
    )
    if optimizer == OptimizerChoice.ADAM:
        if weight_decay > 0:
            print(
                f"Warning: Weight decay {weight_decay} requested for Adam optimizer, but MLX Adam does not support it. Ignoring weight decay."
            )
        opt = optim.Adam(learning_rate=learning_rate)
    elif optimizer == OptimizerChoice.SGD:
        opt = optim.SGD(learning_rate=learning_rate, weight_decay=weight_decay)
    elif optimizer == OptimizerChoice.ADAMW:
        opt = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    trainer = CSMTrainer(
        TrainArgs(
            model=csm_model,
            optimizer=opt,
            output_dir=output_dir,
            max_norm=max_norm,
            first_codebook_weight_multiplier=first_codebook_weight_multiplier,
            gradient_checkpointing=gradient_checkpointing,
            ckpt_freq=ckpt_freq,
            log_freq=log_freq,
        )
    )

    print(f"Loading dataset from {data_path}")
    dataset = CSMDataset.from_json(
        str(data_path),
        n_audio_codebooks=csm_model.n_audio_codebooks,
        max_audio_length_ms=max_audio_length_ms,
        mask_speaker_ids=mask_speaker_ids,
    )
    print(f"Loaded {len(dataset)} samples")

    if len(dataset) == 0:
        print("Error: Dataset is empty. Please check the data path and format.")
        raise typer.Exit(code=1)
    if len(dataset) < batch_size:
        print(
            f"Warning: Dataset size ({len(dataset)}) is smaller than batch size ({batch_size}). Consider reducing batch size."
        )

    print(f"Starting training for {epochs} epochs, batch size {batch_size}")
    print(f"Optimizer: {optimizer.value}, LR: {learning_rate}, WD: {weight_decay}")
    print(f"Saving checkpoints every {ckpt_freq} steps to {output_dir}")
    print(f"Logging metrics every {log_freq} steps")
    print(f"Backbone frozen: {freeze_backbone}, Decoder frozen: {freeze_decoder}")

    try:
        _training_history = trainer.train(
            dataset=dataset,
            batch_size=batch_size,
            epochs=epochs,
        )
        print("\nTraining complete!")
        print(f"Checkpoints and logs saved in {output_dir}")
        final_save_path = output_dir / "final_model.safetensors"
        print(f"Saving final model weights to {final_save_path}...")
        csm_model.save_weights(str(final_save_path))
        print("Final model saved.")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)
