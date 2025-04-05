import typer

from csm_mlx.cli.finetune import app as finetune_app
from csm_mlx.cli.generate import app as generate_app

app = typer.Typer(no_args_is_help=True)


app.add_typer(generate_app)
app.add_typer(finetune_app, name="finetune")
