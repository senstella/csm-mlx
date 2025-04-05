from pathlib import Path

import typer
from huggingface_hub import hf_hub_download
from mlx_lm.sample_utils import make_sampler
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing_extensions import Annotated

from csm_mlx import CSM, Segment, generate
from csm_mlx.cli.config import MODEL, Models
from csm_mlx.cli.utils import read_audio, write_audio

app = typer.Typer()


@app.command("generate")
def generate_command(
    text: str,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            prompt="Please specify the output path",
            help="Output audio file path",
        ),
    ],
    model: Annotated[
        Models, typer.Option("--model", "-m", help="Model size")
    ] = Models._1b,
    speaker: Annotated[
        int,
        typer.Option(
            "--speaker",
            "-s",
            help="Speaker ID to generate (relevant if you're feeding the model previous context)",
        ),
    ] = 0,
    max_audio_length: Annotated[
        int,
        typer.Option(
            "--max-audio-length", "-l", help="Maximum audio length in miliseconds"
        ),
    ] = 10000,
    temperature: Annotated[
        float,
        typer.Option("--temperature", "--temp", "-t", help="Sampling temperature"),
    ] = 0.8,
    top_p: Annotated[
        float | None, typer.Option("--top-p", "-p", help="Top-p sampling parameter")
    ] = None,
    min_p: Annotated[
        float | None, typer.Option("--min-p", "-m", help="Min-p sampling parameter")
    ] = None,
    top_k: Annotated[
        int | None, typer.Option("--top-k", "-k", help="Top-k sampling parameter")
    ] = 50,
    min_tokens_to_keep: Annotated[
        int,
        typer.Option(
            "--min-tokens-to-keep",
            "-kt",
            help="Minimum tokens to keep during sampling",
        ),
    ] = 1,
    input_speakers: Annotated[
        list[int] | None,
        typer.Option("--input-speakers", "-is", help="List of speaker IDs for context"),
    ] = None,
    input_audios: Annotated[
        list[Path] | None,
        typer.Option("--input-audios", "-ia", help="List of audio files for context"),
    ] = None,
    input_texts: Annotated[
        list[str] | None,
        typer.Option(
            "--input-texts", "-it", help="List of text transcripts for context"
        ),
    ] = None,
):
    """Generate speech from text using CSM(Conversational Speech Model)."""
    input_audios = input_audios or []
    input_texts = input_texts or []
    input_speakers = input_speakers or []

    if len(input_audios) != len(input_texts) or len(input_audios) != len(
        input_speakers
    ):
        print(
            "[bold red]Error![/bold red] All context inputs (input_audios, input_texts, and input_speakers) must have the same length."
        )
        raise typer.Exit(code=1)

    sampler = make_sampler(
        temp=temperature,
        top_p=top_p or 0.0,
        min_p=min_p or 0.0,
        top_k=top_k or -1,
        min_tokens_to_keep=min_tokens_to_keep,
    )

    # Download the weights!
    model_config = MODEL[model.value]
    sampling_rate = model_config.get("sampling_rate", 24000)

    weight = hf_hub_download(**model_config.get("loader"))  # type: ignore

    csm = CSM(model_config.get("config"))  # type: ignore
    csm.load_weights(weight)

    context = [
        Segment(speaker, text, read_audio(audio, sampling_rate))
        for audio, text, speaker in zip(input_audios, input_texts, input_speakers)
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Inferencing...", total=None)
        result = generate(
            csm, text, speaker, context, max_audio_length, sampler=sampler
        )

    write_audio(result, output, sampling_rate)
    print(f"[bold green]Success![/bold green] Audio saved to: {output}")


if __name__ == "__main__":
    app()
