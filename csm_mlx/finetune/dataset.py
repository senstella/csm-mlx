import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import audiofile
import audresample
import mlx.core as mx

from csm_mlx.segment import Segment
from csm_mlx.tokenizers import tokenize_segment


@dataclass
class AudioTextSample:
    """A text-audio sample for finetuning."""

    text: str
    audio_path: str
    speaker_id: int = 0

    @property
    def audio(self) -> mx.array:
        """Load the audio file if it exists."""
        if not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

        signal, original_sampling_rate = audiofile.read(self.audio_path, always_2d=True)
        signal = audresample.resample(signal, original_sampling_rate, 24000)

        # Convert to MLX array
        signal = mx.array(signal)

        # Handle stereo to mono conversion if needed
        if signal.shape[0] >= 1:
            signal = signal.mean(axis=0)
        else:
            signal = signal.squeeze(0)

        return signal


class CSMDataset:
    """Dataset for finetuning CSM models."""

    def __init__(
        self,
        samples: List[List[AudioTextSample]],
        n_audio_codebooks: int = 32,
        max_audio_length_ms: int | None = None,
        mask_speaker_ids: int | List[int] | None = None,
    ):
        self.samples = samples
        self.n_audio_codebooks = n_audio_codebooks
        self.max_audio_length_ms = max_audio_length_ms
        self.mask_speaker_ids = (
            mask_speaker_ids
            if isinstance(mask_speaker_ids, list)
            else [mask_speaker_ids]
            if mask_speaker_ids is not None
            else []
        )

    @classmethod
    def from_json(
        cls,
        json_path: str,
        n_audio_codebooks: int = 32,
        max_audio_length_ms: int | None = None,
        mask_speaker_ids: int | List[int] | None = None,
    ) -> "CSMDataset":
        """Load dataset from a JSON file with format:

        [
            [
                {"text": "Sample text", "audio_path": "/path/to/audio.wav", "speaker_id": 0},
                ...
            ]
        ]
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        samples = [
            [
                AudioTextSample(
                    text=item["text"],
                    audio_path=item["audio_path"],
                    speaker_id=item.get("speaker_id", 0),
                )
                for item in conversation
            ]
            for conversation in data
        ]

        return cls(
            samples,
            n_audio_codebooks=n_audio_codebooks,
            max_audio_length_ms=max_audio_length_ms,
            mask_speaker_ids=mask_speaker_ids,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[mx.array, mx.array, mx.array]:
        """Get tokenized representation of the sample."""
        sample = self.samples[idx]
        tokens_list, masks_list = map(
            list,
            zip(
                *[
                    tokenize_segment(
                        Segment(
                            speaker=segment.speaker_id,
                            audio=segment.audio,
                            text=segment.text,
                        ),
                        n_audio_codebooks=self.n_audio_codebooks,
                    )
                    for segment in sample
                ]
            ),
        )
        tokens = mx.concatenate(tokens_list, axis=0)
        masks = mx.concatenate(masks_list, axis=0)
        loss_masks = mx.ones_like(tokens)

        token_position = 0
        for sep_token, segment in zip(tokens_list, sample):
            segment_length = sep_token.shape[0]

            if segment.speaker_id in self.mask_speaker_ids:
                loss_masks[token_position : token_position + segment_length] = 0

            token_position += segment_length

        tokens = (
            tokens[: int(self.max_audio_length_ms / 80)]
            if self.max_audio_length_ms is not None
            else tokens
        )
        masks = (
            masks[: int(self.max_audio_length_ms / 80)]
            if self.max_audio_length_ms is not None
            else masks
        )
        loss_masks = (
            loss_masks[: int(self.max_audio_length_ms / 80)]
            if self.max_audio_length_ms is not None
            else loss_masks
        )

        return tokens, masks, loss_masks

    def get_batch(self, indices: List[int]) -> Tuple[mx.array, mx.array, mx.array]:
        """Get a batch of samples."""
        batch_tokens, batch_masks, batch_loss_masks = [], [], []

        for idx in indices:
            tokens, masks, loss_masks = self[idx]
            batch_tokens.append(tokens)
            batch_masks.append(masks)
            batch_loss_masks.append(loss_masks)

        # Need to handle variable sequence lengths here
        # For simplicity, we'll pad to the longest sequence in the batch
        max_len = max(tokens.shape[0] for tokens in batch_tokens)

        # Pad all sequences to max_len
        padded_tokens = []
        padded_masks = []
        padded_loss_masks = []

        for tokens, masks, loss_masks in zip(
            batch_tokens, batch_masks, batch_loss_masks
        ):
            pad_len = max_len - tokens.shape[0]

            if pad_len > 0:
                # Pad with zeros
                padded_token = mx.pad(tokens, [(0, pad_len), (0, 0)], constant_values=0)
                padded_mask = mx.pad(masks, [(0, pad_len), (0, 0)], constant_values=0)
                padded_loss_mask = mx.pad(
                    loss_masks, [(0, pad_len), (0, 0)], constant_values=0
                )

                padded_tokens.append(padded_token)
                padded_masks.append(padded_mask)
                padded_loss_masks.append(padded_loss_mask)
            else:
                padded_tokens.append(tokens)
                padded_masks.append(masks)
                padded_loss_masks.append(loss_masks)

        return (
            mx.stack(padded_tokens),
            mx.stack(padded_masks),
            mx.stack(padded_loss_masks),
        )
