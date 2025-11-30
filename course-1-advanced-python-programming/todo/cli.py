import typer
from .manager import TaskManager

app = typer.Typer(help="📒 Personal To-Do Manager")
manager = TaskManager()


@app.command()
def add(description: str):
    """Add a new task."""
    manager.add_task(description)
    typer.echo(f"Added task: {description}")


@app.command()
def list():
    """List all tasks."""
    manager.list_tasks()
    typer.echo("Listing tasks...")


@app.command()
def done(index: int):
    """Mark a task as done by its index (1-based)."""
    manager.mark_done(index)
    typer.echo(f"Task {index} marked as done")
