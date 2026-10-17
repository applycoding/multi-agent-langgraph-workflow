import json
from pathlib import Path

from workflow.cli import main


def test_run_status_and_approve(tmp_path: Path, capsys) -> None:
    docs, database = _layout(tmp_path)
    code = main(
        [
            "run",
            "LangGraph human approval",
            "--provider",
            "fake",
            "--checkpoint",
            str(database),
            "--docs-dir",
            str(docs),
            "--thread-id",
            "cli-demo",
        ]
    )
    assert code == 0
    started = json.loads(capsys.readouterr().out)
    assert started["status"] == "awaiting_approval"

    code = main(["status", "cli-demo", *_common(database, docs)])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["thread_id"] == "cli-demo"

    code = main(
        ["approve", "cli-demo", "--notes", "looks good", *_common(database, docs)]
    )
    assert code == 0
    done = json.loads(capsys.readouterr().out)
    assert done["report"]["approved"] is True

    code = main(["approve", "cli-demo", *_common(database, docs)])
    captured = capsys.readouterr()
    assert code == 3
    assert "not awaiting approval" in captured.err


def test_resume_rejection_and_auto_approve(tmp_path: Path, capsys) -> None:
    docs, database = _layout(tmp_path)
    code = main(
        [
            "run",
            "LangGraph human approval",
            "--thread-id",
            "resume-me",
            "--max-revisions",
            "1",
            *_common(database, docs),
        ]
    )
    assert code == 0
    capsys.readouterr()

    code = main(
        [
            "resume",
            "resume-me",
            "--approved",
            "false",
            "--notes",
            "one more pass",
            *_common(database, docs),
        ]
    )
    assert code == 0
    paused = json.loads(capsys.readouterr().out)
    assert paused["status"] == "awaiting_approval"
    assert paused["revision_count"] == 1

    code = main(
        [
            "run",
            "LangGraph human approval",
            "--yes",
            "--thread-id",
            "auto",
            *_common(database, docs),
        ]
    )
    assert code == 0
    finished = json.loads(capsys.readouterr().out)
    assert finished["status"] == "complete"
    assert finished["report"]["approved"] is True


def test_missing_run_exits_with_code_2(tmp_path: Path, capsys) -> None:
    docs, database = _layout(tmp_path)
    code = main(["status", "missing", *_common(database, docs)])
    captured = capsys.readouterr()
    assert code == 2
    assert "not found" in captured.err


def _layout(tmp_path: Path) -> tuple[Path, Path]:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text(
        "LangGraph checkpoint human approval demo.\n",
        encoding="utf-8",
    )
    return docs, tmp_path / "state.sqlite"


def _common(database: Path, docs: Path) -> list[str]:
    return [
        "--provider",
        "fake",
        "--checkpoint",
        str(database),
        "--docs-dir",
        str(docs),
    ]
