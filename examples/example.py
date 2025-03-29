from mlx_lm.sample_utils import make_sampler
from huggingface_hub import hf_hub_download
from csm_mlx import CSM, csm_1b, generate

import audiofile
import numpy as np

# Initialize the model
csm = CSM(csm_1b())  # csm_1b() is a configuration for the CSM model.
weight = hf_hub_download(repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors")
csm.load_weights(weight)

# Generate audio from text
audio = generate(
    csm,
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
    sampler=make_sampler(temp=0.8, top_k=50), # Put mlx_lm's sampler here! Supports: temp, top_p, min_p, min_tokens_to_keep, top_k.
    # Additionally, you can provide `stream` argument to specify what device to use for generation.
    # https://ml-explore.github.io/mlx/build/html/usage/using_streams.html
)

audiofile.write("./output/audio.wav", np.asarray(audio), 24000)

# Create previous conversation segments
context = [
    Segment(
        speaker=0,
        text="How are you doing today?",
        audio=mx.array(...)  # Previous audio for this segment
    ),
    Segment(
        speaker=1,
        text="I'm doing great, thank you!",
        audio=mx.array(...)  # Previous audio for this segment
    )
]

# Generate a response in the conversation
audio = generate(
    csm,
    text="That's wonderful to hear!",
    speaker=0,
    context=context,
    max_audio_length_ms=5_000
    # If you don't provide any sampler, greedy sampling will be used.
)

audiofile.write("./output/audio.wav", np.asarray(audio), 24000)