from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field, ValidationError

from workflow import __version__
from workflow.checkpoint import CheckpointStore
from workflow.config import settings_from_env
from workflow.errors import InvalidRequest, RunExists, RunNotFound, RunNotWaiting
from workflow.models import RunView
from workflow.service import WorkflowService, build_service

ThreadId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,80}$")]


class StartRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=500)
    audience: str = Field(default="technical reader", max_length=200)
    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,80}$")
    max_revisions: int = Field(default=2, ge=0, le=5)


class ApprovalRequest(BaseModel):
    approved: bool
    notes: str = Field(default="", max_length=2000)


def create_app(service: WorkflowService | None = None) -> FastAPI:
    store: CheckpointStore | None = None
    if service is None:
        settings = settings_from_env()
        store = CheckpointStore(settings.checkpoint_path)
        service = build_service(settings, store)
    app_service = service

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        try:
            yield
        finally:
            if store is not None:
                store.close()

    app = FastAPI(
        title="Multi-agent workflow demo",
        version=__version__,
        summary=(
            "Portfolio demo. The default model is fake, and web search reads a fixed corpus."
        ),
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "demo": True}

    @app.post("/runs", response_model=RunView, status_code=201)
    def start_run(body: StartRequest) -> RunView:
        return _call(
            lambda: app_service.start(
                body.topic,
                audience=body.audience,
                thread_id=body.thread_id,
                max_revisions=body.max_revisions,
            )
        )

    @app.get("/runs/{thread_id}", response_model=RunView)
    def get_run(thread_id: ThreadId) -> RunView:
        view = app_service.get(thread_id)
        if view.status == "not_found":
            raise HTTPException(status_code=404, detail=f"Run {thread_id} was not found.")
        return view

    @app.post("/runs/{thread_id}/approve", response_model=RunView)
    def approve_run(thread_id: ThreadId, body: ApprovalRequest) -> RunView:
        return _call(lambda: app_service.approve(thread_id, body.approved, body.notes))

    @app.post("/runs/{thread_id}/resume", response_model=RunView)
    def resume_run(thread_id: ThreadId, body: ApprovalRequest) -> RunView:
        return _call(
            lambda: app_service.resume(
                thread_id,
                {"approved": body.approved, "notes": body.notes},
            )
        )

    return app


def _call(action):
    try:
        return action()
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except (RunExists, RunNotWaiting) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except InvalidRequest as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from None
