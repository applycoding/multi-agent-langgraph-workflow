import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


class CheckpointStore:
    """Own either an in-memory saver or a SQLite file connection."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        self._connection: sqlite3.Connection | None = None
        if path is None:
            self.saver: BaseCheckpointSaver = InMemorySaver()
            return
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(destination, check_same_thread=False)
        saver = SqliteSaver(self._connection)
        saver.setup()
        self.saver = saver

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            connection.close()
