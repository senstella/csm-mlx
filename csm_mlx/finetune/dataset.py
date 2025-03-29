"""Dataset utilities for finetuning CSM models."""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from csm_mlx.segment import Segment
from csm_mlx.tokenizers import tokenize_segment


@dataclass
class AudioTextSample:
    """A text-audio sample for finetuning."""
    text: str
    audio_path: str
    speaker_id: int = 0
    
    @property
    def audio(self) -> Optional[mx.array]:
        """Load the audio file if it exists."""
        if not os.path.exists(self.audio_path):
            return None
        
        try:
            import audiofile
            import audresample
            
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
        except Exception as e:
            print(f"Error loading audio file {self.audio_path}: {e}")
            return None


class CSMDataset:
    """Dataset for finetuning CSM models."""
    
    def __init__(
        self,
        samples: List[AudioTextSample],
        n_audio_codebooks: int = 32,
        max_samples: Optional[int] = None,
    ):
        self.samples = samples[:max_samples] if max_samples else samples
        self.n_audio_codebooks = n_audio_codebooks
        
    @classmethod
    def from_json(cls, json_path: str, max_samples: Optional[int] = None) -> "CSMDataset":
        """Load dataset from a JSON file with format:
        [
            {"text": "Sample text", "audio_path": "/path/to/audio.wav", "speaker_id": 0},
            ...
        ]
        """
        with open(json_path, "r") as f:
            data = json.load(f)
            
        samples = [
            AudioTextSample(
                text=item["text"],
                audio_path=item["audio_path"],
                speaker_id=item.get("speaker_id", 0)
            )
            for item in data
        ]
        
        return cls(samples, max_samples=max_samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[mx.array, mx.array]:
        """Get tokenized representation of the sample."""
        sample = self.samples[idx]
        audio = sample.audio
        
        if audio is None:
            raise ValueError(f"Could not load audio for sample {idx}")
        
        segment = Segment(
            speaker=sample.speaker_id,
            text=sample.text,
            audio=audio
        )
        
        # Tokenize the segment
        tokens, masks = tokenize_segment(
            segment, n_audio_codebooks=self.n_audio_codebooks
        )
        
        return tokens, masks
    
    def get_batch(self, indices: List[int]) -> Tuple[mx.array, mx.array]:
        """Get a batch of samples."""
        batch_tokens, batch_masks = [], []
        
        for idx in indices:
            tokens, masks = self[idx]
            batch_tokens.append(tokens)
            batch_masks.append(masks)
            
        # Need to handle variable sequence lengths here
        # For simplicity, we'll pad to the longest sequence in the batch
        max_len = max(tokens.shape[0] for tokens in batch_tokens)
        
        # Pad all sequences to max_len
        padded_tokens = []
        padded_masks = []
        
        for tokens, masks in zip(batch_tokens, batch_masks):
            pad_len = max_len - tokens.shape[0]
            
            if pad_len > 0:
                # Pad with zeros
                padded_token = mx.pad(
                    tokens, 
                    [(0, pad_len), (0, 0)],
                    constant_values=0
                )
                padded_mask = mx.pad(
                    masks,
                    [(0, pad_len), (0, 0)],
                    constant_values=0
                )
                
                padded_tokens.append(padded_token)
                padded_masks.append(padded_mask)
            else:
                padded_tokens.append(tokens)
                padded_masks.append(masks)
        
        return mx.stack(padded_tokens), mx.stack(padded_masks) 