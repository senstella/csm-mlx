"""Finetuning module for CSM-MLX models."""

# Path handling isn't needed anymore since we're inside the csm_mlx package
from csm_mlx.finetune.dataset import AudioTextSample, CSMDataset
from csm_mlx.finetune.trainer import CSMTrainer

__all__ = ["CSMTrainer", "AudioTextSample", "CSMDataset"]
