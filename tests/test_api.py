import pytest
from fastapi.testclient import TestClient

from workflow.api import create_app
from workflow.service import WorkflowService


@pytest.fixture
def client(service: WorkflowService):
    with TestClient(create_app(service)) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "demo": True}


def test_start_approve_and_read_the_report(client: TestClient) -> None:
    started = client.post("/runs", json={"topic": "LangGraph human approval", "thread_id": "api-1"})
    assert started.status_code == 201
    body = started.json()
    assert body["status"] == "awaiting_approval"
    assert body["report"] is None
    assert body["interrupt"]["value"]["question"] == "Approve this draft?"

    fetched = client.get("/runs/api-1")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "awaiting_approval"

    rejected = client.post(
        "/runs/api-1/approve",
        json={"approved": False, "notes": "tighten the summary"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "awaiting_approval"
    assert rejected.json()["revision_count"] == 1

    approved = client.post(
        "/runs/api-1/resume",
        json={"approved": True, "notes": "ok"},
    )
    assert approved.status_code == 200
    payload = approved.json()
    assert payload["status"] == "complete"
    assert payload["report"]["approved"] is True
    assert payload["report"]["revision_count"] == 1
    kinds = {source["kind"] for source in payload["report"]["sources"]}
    assert kinds == {"web", "local_doc"}
    assert "stub" in " ".join(payload["report"]["caveats"]).lower()


def test_conflicts_and_missing_runs(client: TestClient) -> None:
    body = {"topic": "LangGraph human approval", "thread_id": "demo-thread"}
    assert client.post("/runs", json=body).status_code == 201
    conflict = client.post("/runs", json=body)
    assert conflict.status_code == 409

    missing = client.get("/runs/missing-thread")
    assert missing.status_code == 404

    assert client.post("/runs", json={"topic": ""}).status_code == 422
    assert client.get("/runs/has.dot").status_code == 422

    approved = client.post("/runs/demo-thread/approve", json={"approved": True})
    assert approved.status_code == 200
    again = client.post("/runs/demo-thread/approve", json={"approved": True})
    assert again.status_code == 409
