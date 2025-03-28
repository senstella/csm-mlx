# csm-mlx

An implementation of the CSM(Conversation Speech Model) for Apple Silicon using MLX.

### Installation of the CLI

```bash
# Recommendation: uv tools - works best!
uv tool install "git+https://github.com/senstella/csm-mlx[cli]"

# Or with pipx
pipx install "git+https://github.com/senstella/csm-mlx[cli]"
```

### Basic Text-to-Speech

```bash
csm-mlx "Hello from Sesame." -o output.wav
```

## Installation of the package

Recommendation: Give [`uv`](https://docs.astral.sh/uv/) a try. It's truly magical.
```bash
uv add git+https://github.com/senstella/csm-mlx
```

Or, you can install it via pip:
```bash
pip install git+https://github.com/senstella/csm-mlx
```

## Usage

### Basic generation

> Note: Please install `audiofile` packages to run this example - it is used to save audio file.

```python
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

audiofile.write("./audio.wav", np.asarray(audio), 24000)
```

### Providing Context
```python
from csm_mlx import CSM, csm_1b, generate, Segment
import mlx.core as mx

# Initialize the model
csm = CSM(csm_1b())
weight = hf_hub_download(repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors")
csm.load_weights(weight)

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
```

### Speed up via quantization

```python
from mlx import nn
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from huggingface_hub import hf_hub_download
from csm_mlx import CSM, csm_1b, generate

import audiofile
import numpy as np

# Initialize the model
csm = CSM(csm_1b())  # csm_1b() is a configuration for the CSM model.
weight = hf_hub_download(repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors")
csm.load_weights(weight)

# Quantize
# If you want, specify the `group_size`, `bits`.
# https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html
nn.quantize(csm)

# Generate audio from text
audio = generate(
    csm,
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
    sampler=make_sampler(temp=0.8, top_k=50), # Put mlx_lm's sampler here! Supports: temp, top_p, min_p, min_tokens_to_keep, top_k.
    # If the model doesn't end the sequence properly, you can try to adjust the sampler parameters
    # or set eos_token logit bias via logits_processors=make_logits_processors(logit_bias={0: 4})
    # Additionally, you can provide `stream` argument to specify what device to use for generation.
    # https://ml-explore.github.io/mlx/build/html/usage/using_streams.html
)

audiofile.write("./audio.wav", np.asarray(audio), 24000)
```

### Streaming generation

```python
from mlx import nn
from mlx_lm.sample_utils import make_sampler
from csm_mlx import CSM, csm_1b
from csm_mlx.generation import stream_generate

# Initialize model
csm = CSM(csm_1b())
csm.load_weights("ckpt.safetensors", strict=True)

# nn.quantize(csm) — Speed up nearly real-time on M2 Air, but loses quality.

# Stream generate audio chunks
for chunk in stream_generate(
    csm,
    text="This is an example of streaming audio generation.",
    speaker=0,
    context=[],
    max_audio_length_ms=5000,
    sampler=make_sampler(temp=0.8, top_k=50),
):
    # Process each chunk as it's generated
```

### Loading audio for a segment

If you want to load an audio for a segment, you need to resample it to 24000.

> Note: Please install `audiofile` and `audresample` packages to run this example.

```python
import mlx.core as mx
import audiofile
import audresample

def read_audio(audio_path, sampling_rate=24000) -> mx.array:
    signal, original_sampling_rate = audiofile.read(audio_path, always_2d=True)
    signal = audresample.resample(signal, original_sampling_rate, sampling_rate)

    signal = mx.array(signal)

    if signal.shape[0] >= 1:
        signal = signal.mean(axis=0)
    else:
        signal = signal.squeeze(0)

    return signal  # (audio_length, )
```

## CLI

### Installation

```bash
# Recommendation: uv tools - works best!
uv tool install "git+https://github.com/senstella/csm-mlx[cli]"

# Or with pipx
pipx install "git+https://github.com/senstella/csm-mlx[cli]"
```

### Usage

#### Basic Text-to-Speech

```bash
csm-mlx "Hello from Sesame." -o output.wav
```

#### With Options

```bash
csm-mlx "Hello from Sesame." \
  --output output.wav \
  --model 1b \
  --speaker 0 \
  --temperature 0.8 \
  --min-p 0.05 \
  --max-audio-length 10000
```

#### Feeding Contexts

You can provide conversation context to make the generated speech more natural — or clone a voice with it.

You must provide audio & text & speaker in the pair.

```bash
csm-mlx "Nice to meet you too!" \
  --output response.wav \
  --input-audios previous.wav \
  --input-texts "Hello, nice to meet you." \
  --input-speakers 1
```

#### Portable with uv

```bash
uv run --with 'git+https://github.com/senstella/csm-mlx[cli]' --python 3.12 python -m csm_mlx "Hello from Sesame." -o output.wav
```

### CLI Reference

```
csm-mlx [TEXT] [OPTIONS]
```

#### Arguments

- `TEXT`: The text to convert to speech

#### Options

- `-o, --output PATH`: Output audio file path [required]
- `-m, --model [1b]`: Model size (default: 1b)
- `-s, --speaker INT`: Speaker ID (default: 0)
- `-l, --max-audio-length INT`: Maximum audio length in milliseconds (default: 10000 — 10 seconds)
- `-t, --temperature, --temp FLOAT`: Sampling temperature (default: 0.8)
- `-p, --top-p FLOAT`: Top-p sampling parameter
- `-m, --min-p FLOAT`: Minimum probability for sampling (default: 0.05)
- `-k, --top-k INT`: Top-k sampling parameter
- `-kt, --min-tokens-to-keep INT`: Minimum tokens to keep during sampling (default: 1)
- `-is, --input-speakers LIST`: List of speaker IDs for context
- `-ia, --input-audios LIST`: List of audio files for context
- `-it, --input-texts LIST`: List of text transcripts for context

## Todo

- [X] Fix up RoPE
- [ ] Implement watermarking
- [X] Add streaming generation
- [ ] Optimize performance further for real-time inference

## Acknowledgments

- Thanks to [Sesame](https://sesame.com) for [original PyTorch implementation](https://github.com/SesameAILabs/csm) and [weights](https://huggingface.co/sesame/csm-1b)!
- Thanks to [Moshi](https://github.com/kyutai-labs/moshi) project for creating the mimi codec and `mimi_mlx` implementation.
- Thanks to [torchtune](https://github.com/pytorch/torchtune) project for providing LLaMA attention implementation.
- Thanks to [MLX](https://github.com/ml-explore/mlx) project for providing the framework that made this implementation possible.
- Thanks to [typer](https://typer.tiangolo.com) for powering the CLI interface.
- Thanks to [audiofile](https://github.com/audeering/audiofile) and [audresample](https://github.com/audeering/audresample) for audio processing.

## License

Apache 2.0
