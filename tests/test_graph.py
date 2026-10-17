from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.fakes import FlakyFake
from workflow.checkpoint import CheckpointStore
from workflow.errors import InvalidRequest, RunExists, RunNotFound, RunNotWaiting
from workflow.graph import FAIL, REPORT, WRITE, build_graph, route_after_approval, route_after_step
from workflow.llm import FakeLLM
from workflow.nodes import Dependencies
from workflow.reporting import DEMO_CAVEAT
from workflow.service import WorkflowService
from workflow.state import WorkflowState
from workflow.tools import WEB_CORPUS


def test_routes() -> None:
    assert route_after_step({"status": "failed"}, ready=WRITE) == FAIL
    assert route_after_step({"status": "researching"}, ready=WRITE) == WRITE
    assert (
        route_after_approval(
            {"approval": {"approved": True}, "revision_count": 1, "max_revisions": 2}
        )
        == REPORT
    )
    assert (
        route_after_approval(
            {"approval": {"approved": False}, "revision_count": 2, "max_revisions": 2}
        )
        == WRITE
    )
    assert (
        route_after_approval(
            {"approval": {"approved": False}, "revision_count": 3, "max_revisions": 2}
        )
        == REPORT
    )
    failed: WorkflowState = {"status": "failed", "approval": {"approved": False}}
    assert route_after_approval(failed) == FAIL


def test_start_pauses_for_human_approval(service: WorkflowService) -> None:
    view = service.start("LangGraph human approval", thread_id="thread-1")
    assert view.status == "awaiting_approval"
    assert view.report is None
    assert view.interrupt is not None
    assert view.interrupt.value["critique"]["recommendation"] == "revise"
    sources = view.interrupt.value["sources"]
    kinds = {source["kind"] for source in sources}
    assert kinds == {"web", "local_doc"}
    web_sources = [source for source in sources if source["kind"] == "web"]
    assert all(source["locator"].startswith("stub://") for source in web_sources)

    done = service.approve(view.thread_id, True, "ship the demo")
    assert done.status == "complete"
    assert done.interrupt is None
    assert done.report is not None
    assert done.report.approved is True
    assert done.report.topic == "LangGraph human approval"
    assert DEMO_CAVEAT in done.report.caveats
    assert {source.kind for source in done.report.sources} == {"web", "local_doc"}


def test_rejection_rewrites_until_the_limit(service: WorkflowService) -> None:
    view = service.start("LangGraph human approval", max_revisions=1, thread_id="rewrite")
    mid = service.approve(view.thread_id, False, "tighten it")
    assert mid.status == "awaiting_approval"
    assert mid.revision_count == 1
    done = service.approve(view.thread_id, False, "stop")
    assert done.status == "complete"
    assert done.report is not None
    assert done.report.approved is False
    assert done.revision_count == 2


def test_zero_revisions_finishes_on_the_first_rejection(service: WorkflowService) -> None:
    view = service.start("LangGraph human approval", max_revisions=0, thread_id="once")
    done = service.approve(view.thread_id, False, "no")
    assert done.status == "complete"
    assert done.report is not None
    assert done.report.approved is False
    assert done.revision_count == 1


def test_invalid_resume_keeps_the_run_paused(service: WorkflowService) -> None:
    view = service.start("LangGraph human approval", thread_id="paused")
    with pytest.raises(ValidationError):
        service.resume(view.thread_id, {"notes": "missing the decision"})
    assert service.get(view.thread_id).status == "awaiting_approval"


def test_duplicate_and_missing_runs(service: WorkflowService) -> None:
    service.start("LangGraph human approval", thread_id="same")
    with pytest.raises(RunExists):
        service.start("LangGraph human approval", thread_id="same")
    with pytest.raises(RunNotFound):
        service.approve("missing-thread", True)
    done = service.approve("same", True)
    assert done.status == "complete"
    with pytest.raises(RunNotWaiting):
        service.approve("same", True)
    with pytest.raises(InvalidRequest):
        service.get("bad id")


def test_planner_retries_then_reaches_approval(docs_dir: Path) -> None:
    llm = FlakyFake(failures=2)
    service = _service(docs_dir, llm, attempts=3)
    view = service.start("LangGraph human approval", thread_id="flaky")
    assert view.status == "awaiting_approval"
    assert llm.calls >= 3


def test_exhausted_retries_produce_a_failure_report(docs_dir: Path) -> None:
    llm = FlakyFake(failures=3)
    service = _service(docs_dir, llm, attempts=3)
    view = service.start("LangGraph human approval", thread_id="failed")
    assert view.status == "failed"
    assert llm.calls == 3
    assert view.report is not None
    assert view.report.approved is False
    assert view.report.errors
    assert any("temporary outage" in error for error in view.report.errors)


def test_sqlite_checkpoint_can_be_reopened(docs_dir: Path, tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.sqlite"
    deps = Dependencies(llm=FakeLLM(), docs_dir=docs_dir, web_corpus=WEB_CORPUS)
    store = CheckpointStore(str(path))
    try:
        view = WorkflowService(build_graph(deps, store.saver)).start(
            "LangGraph human approval",
            thread_id="persist-me",
        )
    finally:
        store.close()
    assert view.status == "awaiting_approval"

    restored = CheckpointStore(str(path))
    try:
        service = WorkflowService(build_graph(deps, restored.saver))
        assert service.get("persist-me").status == "awaiting_approval"
        done = service.approve("persist-me", True, "approved after reload")
    finally:
        restored.close()
    assert done.status == "complete"
    assert done.report is not None
    assert done.report.approved is True


def _service(docs_dir: Path, llm: FlakyFake, *, attempts: int) -> WorkflowService:
    deps = Dependencies(llm=llm, docs_dir=docs_dir, web_corpus=WEB_CORPUS, attempts=attempts)
    return WorkflowService(build_graph(deps, CheckpointStore(None).saver))
