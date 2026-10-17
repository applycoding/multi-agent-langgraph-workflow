from pathlib import Path

import pytest

from workflow.checkpoint import CheckpointStore
from workflow.graph import build_graph
from workflow.llm import FakeLLM
from workflow.nodes import Dependencies
from workflow.service import WorkflowService
from workflow.tools import WEB_CORPUS


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "checkpointing.md").write_text(
        "# Checkpointing\n\nLangGraph checkpoints store thread state for human approval.\n",
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def dependencies(docs_dir: Path) -> Dependencies:
    return Dependencies(
        llm=FakeLLM(),
        docs_dir=docs_dir,
        web_corpus=WEB_CORPUS,
        attempts=3,
    )


@pytest.fixture
def service(dependencies: Dependencies):
    store = CheckpointStore(None)
    try:
        yield WorkflowService(build_graph(dependencies, store.saver))
    finally:
        store.close()
