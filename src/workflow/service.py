import re
import uuid
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from workflow.checkpoint import CheckpointStore
from workflow.config import Settings
from workflow.errors import InvalidRequest, RunExists, RunNotFound, RunNotWaiting
from workflow.graph import build_graph
from workflow.llm import build_llm
from workflow.models import Approval, InterruptView, Report, RunView
from workflow.nodes import Dependencies
from workflow.tools import WEB_CORPUS

_THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class WorkflowService:
    def __init__(self, graph: CompiledStateGraph) -> None:
        self.graph = graph

    def start(
        self,
        topic: str,
        *,
        audience: str = "technical reader",
        thread_id: str | None = None,
        max_revisions: int = 2,
    ) -> RunView:
        cleaned = topic.strip()
        if not cleaned:
            raise InvalidRequest("topic is required")
        if max_revisions < 0 or max_revisions > 5:
            raise InvalidRequest("max_revisions must be between 0 and 5")
        chosen = validate_thread_id(thread_id or uuid.uuid4().hex)
        if self.get(chosen).status != "not_found":
            raise RunExists(f"Run {chosen} already exists.")
        self.graph.invoke(
            {
                "topic": cleaned,
                "audience": audience.strip() or "technical reader",
                "max_revisions": max_revisions,
                "revision_count": 0,
                "errors": [],
                "sources": [],
                "status": "running",
            },
            _config(chosen),
        )
        return self.get(chosen)

    def approve(self, thread_id: str, approved: bool, notes: str = "") -> RunView:
        return self.resume(thread_id, {"approved": approved, "notes": notes})

    def resume(self, thread_id: str, payload: dict[str, Any]) -> RunView:
        chosen = validate_thread_id(thread_id)
        self._require_waiting(chosen)
        Approval.model_validate(payload)
        self.graph.invoke(Command(resume=payload), _config(chosen))
        return self.get(chosen)

    def get(self, thread_id: str) -> RunView:
        chosen = validate_thread_id(thread_id)
        snapshot = self.graph.get_state(_config(chosen))
        if snapshot.created_at is None:
            return RunView(thread_id=chosen, status="not_found")
        values = snapshot.values or {}
        interrupts = tuple(snapshot.interrupts or ())
        task_errors = [
            {"agent": task.name, "type": "TaskError", "message": str(task.error)}
            for task in snapshot.tasks
            if task.error
        ]
        errors = list(values.get("errors") or [])
        for item in task_errors:
            if item not in errors:
                errors.append(item)
        if interrupts:
            status = "awaiting_approval"
        elif task_errors or values.get("status") == "failed":
            status = "failed"
        elif not snapshot.next:
            status = "complete"
        else:
            status = "running"
        interrupt = None
        if interrupts:
            first = interrupts[0]
            value = first.value if isinstance(first.value, dict) else {"value": first.value}
            interrupt = InterruptView(id=first.id, value=value)
        raw_report = values.get("report")
        report = Report.model_validate(raw_report) if raw_report else None
        return RunView(
            thread_id=chosen,
            status=status,
            topic=values.get("topic"),
            next_nodes=list(snapshot.next),
            revision_count=int(values.get("revision_count") or 0),
            interrupt=interrupt,
            report=report,
            errors=errors,
        )

    def _require_waiting(self, thread_id: str) -> RunView:
        current = self.get(thread_id)
        if current.status == "not_found":
            raise RunNotFound(f"Run {thread_id} was not found.")
        if current.status != "awaiting_approval":
            raise RunNotWaiting(f"Run {thread_id} is {current.status}, not awaiting approval.")
        return current


def build_service(settings: Settings, store: CheckpointStore) -> WorkflowService:
    dependencies = Dependencies(
        llm=build_llm(settings),
        docs_dir=settings.docs_dir,
        web_corpus=WEB_CORPUS,
        attempts=settings.llm_attempts,
    )
    return WorkflowService(build_graph(dependencies, store.saver))


def validate_thread_id(thread_id: str) -> str:
    if not _THREAD_ID.fullmatch(thread_id):
        raise InvalidRequest(
            "thread_id must be 1-80 characters of letters, numbers, underscores, or hyphens."
        )
    return thread_id


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}
