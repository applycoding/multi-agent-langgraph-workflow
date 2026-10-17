from pathlib import Path

from workflow.tools import search_local_docs, web_search


def test_web_search_ranks_the_sqlite_note_first() -> None:
    hits = web_search("sqlite checkpoint")
    assert hits[0].title == "SQLite checkpointing"
    assert hits[0].kind == "web"
    assert hits[0].locator.startswith("stub://")
    assert hits[0].score >= 1


def test_web_search_ignores_blank_and_unknown_queries() -> None:
    assert web_search("   ") == []
    assert web_search("zzzz-not-a-term") == []


def test_web_search_respects_the_limit() -> None:
    hits = web_search("checkpoint", limit=1)
    assert len(hits) == 1


def test_local_docs_search_reads_markdown(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text(
        "LangGraph checkpoints support human approval.\n",
        encoding="utf-8",
    )
    (tmp_path / "ignore.txt").write_text("LangGraph checkpoint\n", encoding="utf-8")
    hits = search_local_docs("langgraph checkpoint", tmp_path)
    assert len(hits) == 1
    assert hits[0].kind == "local_doc"
    assert hits[0].title == "notes"
    assert "LangGraph" in hits[0].excerpt


def test_local_docs_match_a_checkpoint_prefix(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("The system keeps checkpoints.\n", encoding="utf-8")
    hits = search_local_docs("checkpoint", tmp_path)
    assert len(hits) == 1


def test_local_docs_missing_directory_is_empty(tmp_path: Path) -> None:
    assert search_local_docs("langgraph", tmp_path / "missing") == []
