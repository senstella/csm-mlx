"""Trainer for finetuning CSM models."""

import json
import os
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.losses import cross_entropy
from mlx.utils import tree_flatten
from mlx_lm.utils import save_weights
from tqdm import tqdm

from csm_mlx.finetune.dataset import CSMDataset
from csm_mlx.models import CSM


class BaseTrainer:
    """Base trainer class with common functionality for CSM trainers."""

    def __init__(
        self,
        model: CSM,
        optimizer: optim.Optimizer,
        checkpoint_dir: str,
        save_every: int = 100,
        log_every: int = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.log_every = log_every

        # Create checkpoint dir if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize training state
        self.step = 0
        self.best_loss = float("inf")
        self.training_history = {
            "loss": [],
            "learning_rate": [],
            "step": [],
        }

        # Compile the loss and train functions
        self.loss_and_grad_fn = nn.value_and_grad(self.model, self.compute_loss)  # type: ignore

    def compute_loss(self, tokens: mx.array, masks: mx.array) -> mx.array:
        """Compute loss for a batch of samples."""
        # Text target is the text tokens shifted by 1
        batch_size, seq_len, n_codebooks = tokens.shape

        # Extract text tokens (last codebook) and audio tokens (all other codebooks)
        audio_tokens = tokens[:, :, :-1]  # (batch, seq, codebook)
        shifted_audio_tokens = audio_tokens[..., 1:]  # (batch, seq - 1, codebook)
        audio_masks = masks[:, :, :-1]  # (batch, seq, codebook)

        # Forward pass through the model
        backbone_embeds = self.model.embed_tokens(tokens)
        backbone_embeds = backbone_embeds * mx.expand_dims(masks, axis=-1)
        backbone_input = backbone_embeds.sum(-2)

        backbone_hidden = self.model.backbone(backbone_input)
        shifted_backbone_hidden = backbone_hidden[
            ..., 1:, :
        ]  # (batch, seq - 1, embed_dim)

        c0_logits = self.model.codebook0_head(backbone_hidden)
        c0_shifted_logits = c0_logits[..., :-1, :]

        ci_stacked = mx.concat(
            [
                self.model.embed_audio(i, audio_tokens[:, :, i])
                for i in range(self.model.n_audio_codebooks)
            ],
            axis=-1,
        )  # (batch, seq, codebook + backbone activation, embed_dim)
        decoder_inputs = mx.concat(
            [mx.expand_dims(shifted_backbone_hidden, axis=-2), ci_stacked], axis=-2
        )  # (batch, seq, codebook + 1(backbone activation), embed_dim)

        # TODO: Apply compute amortization since those consumes VERY HIGH memory as mentioned in Sesame's blog
        # https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice
        decoder_hidden = self.model.decoder(self.model.projection(decoder_inputs))[
            ..., 1:, :
        ]  # (batch, seq, codebook, vocab_size) - we don't need c0 predictions

        # Calculate total losses at once.
        c0_loss = cross_entropy(
            c0_shifted_logits.reshape(-1, c0_shifted_logits.shape[-1]),
            shifted_audio_tokens[..., 0].reshape(-1),
            reduction="none",
        )
        c0_loss = (c0_loss * audio_masks[:, :, 0].reshape(-1)).sum() / audio_masks[
            :, :, 0
        ].sum()

        total_loss = c0_loss / self.model.n_audio_codebooks

        for index in range(1, self.model.n_audio_codebooks):
            ci_logits = mx.matmul(
                decoder_hidden[..., index - 1, :], self.model.audio_head[index - 1]
            )
            ci_loss = cross_entropy(
                ci_logits.reshape(-1, ci_logits.shape[-1]),
                shifted_audio_tokens[:, :, index].reshape(-1),
                reduction="none",
            )
            ci_loss = (
                ci_loss * audio_masks[:, :, index].reshape(-1)
            ).sum() / audio_masks[:, :, index].sum()
            total_loss += ci_loss / self.model.n_audio_codebooks

        return total_loss

    def train_step(self, batch_tokens: mx.array, batch_masks: mx.array) -> float:
        """Perform a single training step."""
        loss, grads = self.loss_and_grad_fn(batch_tokens, batch_masks)
        self.optimizer.update(self.model, grads)
        self.step += 1

        # Log and save
        if self.step % self.log_every == 0:
            self.training_history["loss"].append(float(loss))
            self.training_history["step"].append(self.step)
            self.training_history["learning_rate"].append(
                float(self.optimizer.learning_rate)
            )

            # Save history
            with open(os.path.join(self.checkpoint_dir, "history.json"), "w") as f:
                json.dump(self.training_history, f, indent=2)

        if self.step % self.save_every == 0:
            self.save_checkpoint()

        return float(loss)

    def train(
        self, dataset: CSMDataset, batch_size: int, epochs: int, shuffle: bool = True
    ) -> Dict:
        """Train the model on the dataset."""
        num_samples = len(dataset)
        _steps_per_epoch = (num_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Create batch indices
            indices = mx.arange(num_samples)
            if shuffle:
                indices = mx.random.permutation(indices)

            batch_indices = [
                indices[i : i + batch_size].tolist()
                for i in range(0, num_samples, batch_size)
            ]

            # Train on batches
            epoch_losses = []
            for batch_idx in tqdm(batch_indices, desc="Training"):
                batch_tokens, batch_masks = dataset.get_batch(batch_idx)  # type: ignore
                loss = self.train_step(batch_tokens, batch_masks)
                epoch_losses.append(loss)

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")

        return self.training_history

    def get_params_to_save(self) -> Dict[str, Any]:
        """Get parameters to save in checkpoint.
        This method can be overridden by subclasses to save different parameters.
        """
        return self.model.parameters()

    def get_checkpoint_prefix(self) -> str:
        """Get prefix for checkpoint files."""
        return "ckpt"

    def save_checkpoint(self, suffix: Optional[str] = None) -> None:
        """Save model checkpoint."""
        prefix = self.get_checkpoint_prefix()
        if suffix:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"{prefix}_{suffix}.safetensors"
            )
        else:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"{prefix}_step_{self.step}.safetensors"
            )

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

        # Get parameters to save
        params_to_save = self.get_params_to_save()

        # Get flattened parameters
        flat_params = self._flatten_params(params_to_save)

        # Filter out any parameters without the nbytes attribute
        valid_params = {k: v for k, v in flat_params.items() if hasattr(v, "nbytes")}

        # Save weights with flattened parameters
        save_weights(checkpoint_path, valid_params)

        # Save optimizer state
        optimizer_state = {
            "step": self.step,
            "learning_rate": float(self.optimizer.learning_rate),
        }

        with open(os.path.join(self.checkpoint_dir, "optimizer_state.json"), "w") as f:
            json.dump(optimizer_state, f, indent=2)

        print(f"Saved checkpoint to {checkpoint_path}")

    def _flatten_params(self, params_dict, prefix=""):
        """Recursively flatten a nested parameter dictionary.

        Args:
            params_dict: A possibly nested dictionary of parameters
            prefix: Prefix for parameter names (used in recursion)

        Returns:
            A flat dictionary suitable for save_weights
        """
        flat_params = {}

        for name, param in params_dict.items():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(param, dict):
                # Recursively flatten nested dictionaries
                nested_params = self._flatten_params(param, full_name)
                flat_params.update(nested_params)
            elif isinstance(param, list):
                # Handle lists of parameters (like audio_head)
                for i, item in enumerate(param):
                    if hasattr(item, "nbytes"):  # It's an MLX array
                        flat_params[f"{full_name}.{i}"] = item
                    elif isinstance(item, (list, dict)):
                        # Recursively handle nested structures
                        nested_params = self._flatten_params({str(i): item}, full_name)
                        flat_params.update(nested_params)
            elif hasattr(param, "nbytes"):
                # This is an actual parameter (mx.array)
                flat_params[full_name] = param
            else:
                # Skip parameters that aren't MLX arrays
                print(f"Warning: Skipping parameter {full_name} of type {type(param)}")

        return flat_params

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model and optimizer state from checkpoint."""
        # Load model weights
        self.model.load_weights(checkpoint_path)

        # Load optimizer state if exists
        optimizer_state_path = os.path.join(
            os.path.dirname(checkpoint_path), "optimizer_state.json"
        )
        if os.path.exists(optimizer_state_path):
            with open(optimizer_state_path, "r") as f:
                optimizer_state = json.load(f)

            self.step = optimizer_state["step"]
            self.optimizer.learning_rate = optimizer_state["learning_rate"]

            print(f"Loaded optimizer state from {optimizer_state_path}")

        print(f"Loaded checkpoint from {checkpoint_path}")


class CSMTrainer(BaseTrainer):
    """Trainer for full finetuning of CSM models."""

    def __init__(
        self,
        model: CSM,
        optimizer: optim.Optimizer,
        checkpoint_dir: str,
        save_every: int = 100,
        log_every: int = 10,
    ):
        super().__init__(model, optimizer, checkpoint_dir, save_every, log_every)


class LoRATrainer(BaseTrainer):
    """Trainer for finetuning CSM models with LoRA."""

    def __init__(
        self,
        model: CSM,
        optimizer: optim.Optimizer,
        checkpoint_dir: str,
        save_every: int = 100,
        log_every: int = 10,
        train_embeddings: bool = False,
    ):
        super().__init__(model, optimizer, checkpoint_dir, save_every, log_every)
        self.train_embeddings = train_embeddings

        # Extract LoRA parameters that need to be updated
        self.lora_params = tree_flatten(model.trainable_parameters())

        # Extract embedding parameters separately if requested
        self.embedding_params = {}
        if self.train_embeddings:
            all_params = self.model.parameters()
            for name, param in all_params.items():
                if "text_embeddings" in name or "audio_embeddings" in name:
                    self.embedding_params[name] = param

        # Combine parameters to train: both LoRA and embeddings (if requested)
        self.trainable_params = {**self.lora_params, **self.embedding_params}

        print("\nTrainable parameters:")
        print(f"LoRA parameters: {len(self.lora_params)} parameters")
        print(f"Embedding parameters: {len(self.embedding_params)} parameters")
        print(f"Total trainable parameters: {len(self.trainable_params)} parameters")
        if self.train_embeddings:
            print(
                "Embedding layers included in training: text_embeddings, audio_embeddings"
            )
        else:
            print("Embedding layers not directly trained")

    def get_params_to_save(self) -> Dict[str, Any]:
        """Override to save only trainable parameters (LoRA and embeddings if enabled)."""
        return self.trainable_params

    def get_checkpoint_prefix(self) -> str:
        """Override to use lora_ckpt prefix for checkpoint files."""
        return "lora_ckpt"
