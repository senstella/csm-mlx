import typer

from csm_mlx.cli.finetune.dataset import app as dataset_app
from csm_mlx.cli.finetune.full_finetune import app as finetune_app
from csm_mlx.cli.finetune.lora_finetune import app as lora_finetune_app

app = typer.Typer(
    no_args_is_help=True, help="Finetuning module for CSM(Conversation Speech Model)."
)

app.add_typer(finetune_app)
app.add_typer(lora_finetune_app)
app.add_typer(dataset_app)
