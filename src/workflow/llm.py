import json
import re
from collections.abc import Callable
from typing import Protocol, TypeVar, assert_never

import httpx
from pydantic import BaseModel, ValidationError

from workflow.config import Provider, Settings
from workflow.errors import ConfigurationError, StructuredOutputError, TransientLLMError
from workflow.models import ChatMessage, Critique, Draft, Plan, PlanStep, ResearchNotes, Section
from workflow.providers import AnthropicProvider, GeminiProvider, OllamaProvider, OpenAIProvider

ModelT = TypeVar("ModelT", bound=BaseModel)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMClient(Protocol):
    name: str

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel: ...


class FakeLLM:
    """Deterministic stand-in used by the demo default and by unit tests.

    It never calls a network and never claims to score model quality.
    """

    name = "fake"

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        topic = _labeled_value(messages, "Topic:") or "untitled topic"
        if schema is None:
            return f"Fake response for {topic}."
        if schema is Plan:
            return Plan(
                thesis=f"Explain {topic} with a planner, researcher, and writer.",
                steps=[
                    PlanStep(
                        title="Gather sources",
                        objective="Collect local notes and stub-web hits.",
                        queries=[topic, "langgraph checkpoint"],
                    )
                ],
                success_criteria=[
                    "Cite tool sources",
                    "Pause for human approval",
                ],
            )
        if schema is ResearchNotes:
            return ResearchNotes(
                summary=f"Notes for {topic} based on the supplied tool hits.",
                findings=[
                    "The demo reads a fixed web corpus and local markdown files.",
                ],
                gaps=["Live web search is not connected."],
            )
        if schema is Draft:
            return Draft(
                title=topic,
                summary=f"Draft briefing on {topic}.",
                sections=[
                    Section(
                        heading="Overview",
                        body=f"This briefing covers {topic}.",
                    )
                ],
                caveats=["Demo draft produced by the fake provider."],
            )
        if schema is Critique:
            return Critique(
                strengths=["The draft stays inside the demo scope."],
                concerns=["A person should confirm the draft before treating it as final."],
                recommendation="revise",
                notes="Confirm that the cited sources came from the tool results.",
            )
        raise TypeError(f"FakeLLM has no response for schema {schema.__name__}")


def build_llm(
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> LLMClient:
    provider = settings.provider
    match provider:
        case Provider.FAKE:
            return FakeLLM()
        case Provider.OPENAI:
            return OpenAIProvider(
                model=settings.openai_model,
                api_key=_require(settings.openai_api_key, "OPENAI_API_KEY"),
                base_url=settings.openai_base_url,
                timeout=settings.timeout_seconds,
                transport=transport,
            )
        case Provider.ANTHROPIC:
            return AnthropicProvider(
                model=settings.anthropic_model,
                api_key=_require(settings.anthropic_api_key, "ANTHROPIC_API_KEY"),
                base_url=settings.anthropic_base_url,
                timeout=settings.timeout_seconds,
                transport=transport,
            )
        case Provider.GEMINI:
            return GeminiProvider(
                model=settings.gemini_model,
                api_key=_require(settings.gemini_api_key, "GEMINI_API_KEY"),
                base_url=settings.gemini_base_url,
                timeout=settings.timeout_seconds,
                transport=transport,
            )
        case Provider.OLLAMA:
            return OllamaProvider(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                timeout=settings.timeout_seconds,
                transport=transport,
            )
        case _ as unreachable:
            assert_never(unreachable)


def complete_structured(
    llm: LLMClient,
    messages: list[ChatMessage],
    schema: type[ModelT],
    *,
    attempts: int = 3,
    sleep: Callable[[int], None] | None = None,
) -> ModelT:
    """Ask for a schema, then retry transient failures and invalid JSON."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    pause = sleep or (lambda _attempt: None)
    conversation = [
        ChatMessage(
            role="system",
            content=(
                "Return only a JSON object that validates against this JSON Schema. "
                "Do not wrap it in markdown.\n"
                + json.dumps(schema.model_json_schema())
            ),
        ),
        *messages,
    ]
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = llm.complete(conversation, schema=schema)
        except TransientLLMError as exc:
            last_error = exc
            if attempt == attempts:
                raise
            pause(attempt)
            continue
        try:
            return coerce_schema(raw, schema)
        except StructuredOutputError as exc:
            last_error = exc
            if attempt == attempts:
                raise
            conversation.append(
                ChatMessage(
                    role="user",
                    content=(
                        f"The previous response was not valid {schema.__name__} JSON. "
                        f"Error: {exc}. Return only the JSON object."
                    ),
                )
            )
            pause(attempt)
    raise StructuredOutputError(str(last_error or "structured output failed"))


def coerce_schema(raw: str | BaseModel, schema: type[ModelT]) -> ModelT:
    if isinstance(raw, schema):
        return raw
    if isinstance(raw, BaseModel):
        raw = raw.model_dump_json()
    if not isinstance(raw, str):
        raise StructuredOutputError(
            f"Expected text or {schema.__name__}, got {type(raw).__name__}"
        )
    try:
        return schema.model_validate_json(extract_json(raw))
    except (ValidationError, ValueError) as exc:
        raise StructuredOutputError(str(exc)) from exc


def extract_json(text: str) -> str:
    stripped = _FENCE.sub("", text.strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in model output")
    return stripped[start : end + 1]


def _require(value: str | None, name: str) -> str:
    if value:
        return value
    raise ConfigurationError(f"{name} is not set.")


def _labeled_value(messages: list[ChatMessage], label: str) -> str | None:
    for message in messages:
        for line in message.content.splitlines():
            if line.startswith(label):
                return line.split(":", 1)[1].strip()
    return None


__all__ = [
    "FakeLLM",
    "LLMClient",
    "build_llm",
    "coerce_schema",
    "complete_structured",
    "extract_json",
]
