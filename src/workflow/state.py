import operator
from typing import Annotated, Any, TypedDict


class WorkflowState(TypedDict, total=False):
    topic: str
    audience: str
    max_revisions: int
    revision_count: int
    plan: dict[str, Any]
    research: dict[str, Any]
    sources: list[dict[str, Any]]
    draft: dict[str, Any]
    critique: dict[str, Any]
    approval: dict[str, Any]
    report: dict[str, Any]
    errors: Annotated[list[dict[str, Any]], operator.add]
    status: str
    failed_agent: str
