from rich.console import Console
from rich.table import Table
from .models import Task
from .storage import load_tasks, save_tasks

console = Console()


class TaskManager:
    def __init__(self):
        self.tasks = load_tasks()

    def add_task(self, description: str):
        self.tasks.append(Task(description=description))
        save_tasks(self.tasks)
        console.print(f"[green]✔ Added:[/] {description}")

    def list_tasks(self):
        if not self.tasks:
            console.print("[yellow]No tasks yet![/]")
            return
        table = Table(title="To-Do List")
        table.add_column("ID", style="cyan")
        table.add_column("Task", style="magenta")
        table.add_column("Status", style="green")
        for i, task in enumerate(self.tasks, 1):
            status = "✅ Done" if task.done else "❌ Pending"
            table.add_row(str(i), task.description, status)
        console.print(table)

    def mark_done(self, index: int):
        if 0 < index <= len(self.tasks):
            self.tasks[index - 1].mark_done()
            save_tasks(self.tasks)
            console.print(f"[blue]Task {index} marked as done[/]")
        else:
            console.print(f"[red]Invalid task index: {index}[/]")
