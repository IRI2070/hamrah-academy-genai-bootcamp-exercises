import json
from pathlib import Path

from .models import Task

FILE = Path("tasks.json")


def save_tasks(tasks: list[Task]) -> None:
    with FILE.open("w") as f:
        json.dump([task.__dict__ for task in tasks], f, indent=2)


def load_tasks() -> list[Task]:
    if not FILE.exists():
        return []
    with FILE.open() as f:
        data = json.load(f)
    return [Task(**item) for item in data]
