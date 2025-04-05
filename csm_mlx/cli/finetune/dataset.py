import json
from pathlib import Path
from typing import Any, Dict, List

import typer
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn

from csm_mlx.cli.finetune.utils import find_speaker_id, natural_sort_key

app = typer.Typer(no_args_is_help=True)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"}


@app.command()
def convert(
    input_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory containing conversation subdirectories.",
    ),
    output_json: Path = typer.Argument(
        ...,
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
        help="Path to save the output JSON file.",
    ),
):
    """
    Converts a directory structured with conversation subfolders into
    the JSON format expected by --data-path.
    """

    all_conversations_data: List[List[Dict[str, Any]]] = []
    processed_dirs = 0
    total_samples_added = 0
    total_skipped_files = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        sorted_dirs = sorted(input_dir.iterdir())
        scan_task = progress.add_task(
            "[cyan]Scanning directories...", total=len(sorted_dirs)
        )

        for item in sorted_dirs:
            progress.update(scan_task, description=f"[cyan]Processing {item.name}...")
            if item.is_dir():
                processed_dirs += 1
                potential_files = list(item.iterdir())
                audio_files: Dict[str, Path] = {}
                text_files: Dict[str, Path] = {}

                for file_path in potential_files:
                    if file_path.is_file():
                        base_name = file_path.stem
                        suffix = file_path.suffix.lower()
                        if suffix in AUDIO_EXTENSIONS:
                            audio_files[base_name] = file_path
                        elif suffix == ".txt":
                            text_files[base_name] = file_path

                current_conversation_samples: List[Dict[str, Any]] = []
                skipped_in_conv = 0

                sorted_audio_basenames = sorted(
                    audio_files.keys(), key=natural_sort_key
                )

                for base_name in sorted_audio_basenames:
                    audio_path = audio_files[base_name]
                    if base_name in text_files:
                        text_path = text_files[base_name]
                        speaker_id = find_speaker_id(audio_path.name)
                        if speaker_id is None:
                            print(
                                "[bold red]Error[/bold red] Could not detect speaker ID for file:"
                            )
                            print(f"  '{audio_path}'")
                            print(
                                "Filename must start with '<digits>_' or 'speaker<digits>_' (case-insensitive)."
                            )
                            raise typer.Exit(code=1)

                        try:
                            text_content = text_path.read_text(encoding="utf-8").strip()
                            if not text_content:
                                print(
                                    f"[yellow]Warning[/yellow] Empty text file skipped: '{text_path.name}' in '{item.name}'"
                                )
                                skipped_in_conv += 1
                                continue

                            sample_data = {
                                "text": text_content,
                                "audio_path": str(audio_path.resolve()),
                                "speaker_id": speaker_id,
                            }
                            current_conversation_samples.append(sample_data)

                        except Exception as e:
                            print(
                                f"[bold red]Error[/bold red] Failed to read text file '{text_path.name}': {e}"
                            )
                            skipped_in_conv += 1
                    else:
                        skipped_in_conv += 1

                if current_conversation_samples:
                    all_conversations_data.append(current_conversation_samples)
                    total_samples_added += len(current_conversation_samples)
                    if skipped_in_conv > 0:
                        print(
                            f"[yellow]Info:[/yellow] Skipped {skipped_in_conv} file(s) in '{item.name}' due to missing text or read errors."
                        )

                total_skipped_files += skipped_in_conv
            else:
                pass

            progress.advance(scan_task)

    print("\n--- Conversion Summary ---")
    print(f"Processed {processed_dirs} potential conversation directories.")
    if total_skipped_files > 0:
        print(
            f"[yellow]Skipped {total_skipped_files} audio files total[/yellow] (missing text, read errors)."
        )

    if not all_conversations_data:
        print(
            "[bold red]Error[/bold red] No valid conversations found. Output JSON will be empty."
        )
        all_conversations_data = []
    else:
        print(
            f"Found [bold cyan]{len(all_conversations_data)}[/bold cyan] valid conversations."
        )
        print(f"Total samples added: [bold cyan]{total_samples_added}[/bold cyan]")

    print(f"\nWriting JSON output to: '{output_json}'")
    try:
        output_json.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(
            f"[bold red]Error[/bold red] Could not create output directory '{output_json.parent}': {e}"
        )
        raise typer.Exit(code=1)

    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_conversations_data, f, indent=4, ensure_ascii=False)
        print(f"[bold green]Successfully wrote JSON to: {output_json}[/bold green]")
    except Exception as e:
        print(f"[bold red]Error[/bold red] Failed writing JSON file: {e}")
        raise typer.Exit(code=1)
