from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Source(BaseModel):
    title: str
    locator: str
    excerpt: str
    kind: Literal["web", "local_doc"]


class PlanStep(BaseModel):
    title: str
    objective: str
    queries: list[str] = Field(min_length=1)


class Plan(BaseModel):
    thesis: str
    steps: list[PlanStep] = Field(min_length=1)
    success_criteria: list[str] = Field(min_length=1)


class ResearchNotes(BaseModel):
    summary: str
    findings: list[str] = Field(min_length=1)
    gaps: list[str]


class Section(BaseModel):
    heading: str
    body: str


class Draft(BaseModel):
    title: str
    summary: str
    sections: list[Section] = Field(min_length=1)
    caveats: list[str]


class Critique(BaseModel):
    strengths: list[str]
    concerns: list[str]
    recommendation: Literal["approve", "revise"]
    notes: str


class Approval(BaseModel):
    approved: bool
    notes: str = ""


class Report(BaseModel):
    title: str
    topic: str
    audience: str
    thesis: str
    summary: str
    sections: list[Section]
    sources: list[Source]
    caveats: list[str]
    critique_notes: str
    approved: bool
    revision_count: int
    errors: list[str]


class InterruptView(BaseModel):
    id: str
    value: dict


class RunView(BaseModel):
    thread_id: str
    status: Literal["not_found", "awaiting_approval", "complete", "failed", "running"]
    topic: str | None = None
    next_nodes: list[str] = Field(default_factory=list)
    revision_count: int = 0
    interrupt: InterruptView | None = None
    report: Report | None = None
    errors: list[dict] = Field(default_factory=list)
