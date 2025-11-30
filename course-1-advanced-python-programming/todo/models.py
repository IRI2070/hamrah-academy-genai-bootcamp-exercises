from dataclasses import dataclass


@dataclass
class Task:
    description: str
    done: bool = False

    def mark_done(self) -> None:
        """Mark the task as completed."""
        self.done = True
