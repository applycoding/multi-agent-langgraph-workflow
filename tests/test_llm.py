import json

import httpx
import pytest
from pydantic import BaseModel

from tests.fakes import ScriptedLLM
from workflow.config import Provider, Settings, parse_provider, settings_from_env
from workflow.errors import (
    AuthenticationError,
    ConfigurationError,
    StructuredOutputError,
    TransientLLMError,
)
from workflow.llm import (
    FakeLLM,
    build_llm,
    coerce_schema,
    complete_structured,
    extract_json,
)
from workflow.models import ChatMessage, Plan, PlanStep
from workflow.providers import AnthropicProvider, GeminiProvider, OllamaProvider, OpenAIProvider

PLAN = Plan(
    thesis="Explain the topic.",
    steps=[PlanStep(title="Gather", objective="Find notes", queries=["langgraph"])],
    success_criteria=["Cite sources"],
)
PLAN_JSON = PLAN.model_dump_json()
USER = [ChatMessage(role="user", content="Topic: LangGraph human approval")]


def test_fake_plan_uses_the_topic() -> None:
    plan = FakeLLM().complete(USER, schema=Plan)
    assert isinstance(plan, Plan)
    assert "LangGraph human approval" in plan.thesis


def test_fake_rejects_an_unknown_schema() -> None:
    class Other(BaseModel):
        name: str

    with pytest.raises(TypeError):
        FakeLLM().complete(USER, schema=Other)


def test_parse_provider_and_settings() -> None:
    assert parse_provider("OpenAI") is Provider.OPENAI
    settings = settings_from_env({"CHECKPOINT_PATH": "memory", "LLM_PROVIDER": "fake"})
    assert settings.checkpoint_path is None
    assert settings.provider is Provider.FAKE
    with pytest.raises(ConfigurationError):
        parse_provider("not-a-provider")
    with pytest.raises(ConfigurationError):
        settings_from_env({"LLM_ATTEMPTS": "0"})


def test_build_llm_requires_hosted_credentials() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_llm(Settings(provider=Provider.OPENAI))
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        build_llm(Settings(provider=Provider.ANTHROPIC))
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        build_llm(Settings(provider=Provider.GEMINI))
    assert isinstance(build_llm(Settings(provider=Provider.FAKE)), FakeLLM)


def test_extract_json_strips_a_fence() -> None:
    parsed = Plan.model_validate_json(extract_json(f"```json\n{PLAN_JSON}\n```"))
    assert parsed == PLAN
    with pytest.raises(StructuredOutputError):
        coerce_schema("nope", Plan)


def test_structured_output_retries_invalid_json() -> None:
    seen: dict[str, list[ChatMessage]] = {}

    def record(messages, schema):
        del schema
        seen["messages"] = messages
        return PLAN_JSON

    llm = ScriptedLLM(["not json", record])
    result = complete_structured(llm, USER, Plan, attempts=2)
    assert result == PLAN
    assert llm.calls == 2
    assert any("JSON Schema" in message.content for message in seen["messages"])


def test_openai_request_and_parse() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"choices": [{"message": {"content": PLAN_JSON}}]})

    client = OpenAIProvider(
        model="gpt-4o-mini",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert client.complete(USER, schema=Plan) == PLAN_JSON
    finally:
        client.close()
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gpt-4o-mini"
    assert body["response_format"] == {"type": "json_object"}
    assert seen["auth"] == "Bearer test-key"


def test_openai_retries_503_and_stops_on_401() -> None:
    calls = {"n": 0}

    def busy(request: httpx.Request) -> httpx.Response:
        del request
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json={"choices": [{"message": {"content": PLAN_JSON}}]})

    client = OpenAIProvider(
        model="gpt-4o-mini",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(busy),
    )
    try:
        result = complete_structured(client, USER, Plan, attempts=2)
    finally:
        client.close()
    assert result == PLAN
    assert calls["n"] == 2

    def denied(request: httpx.Request) -> httpx.Response:
        del request
        calls["n"] += 1
        return httpx.Response(401, json={"error": "nope"})

    calls["n"] = 0
    denied_client = OpenAIProvider(
        model="gpt-4o-mini",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(denied),
    )
    try:
        with pytest.raises(AuthenticationError):
            complete_structured(denied_client, USER, Plan, attempts=3)
    finally:
        denied_client.close()
    assert calls["n"] == 1


def test_anthropic_gemini_and_ollama_parse_text() -> None:
    def anthropic(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["model"] == "claude-3-5-sonnet-latest"
        assert request.headers["x-api-key"] == "test-key"
        return httpx.Response(200, json={"content": [{"type": "text", "text": PLAN_JSON}]})

    def gemini(request: httpx.Request) -> httpx.Response:
        assert "key=test-key" in str(request.url)
        body = json.loads(request.content.decode())
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": PLAN_JSON}]}}]},
        )

    def ollama(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["model"] == "llama3.2"
        assert body["format"] == "json"
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"message": {"role": "assistant", "content": PLAN_JSON}})

    clients = [
        AnthropicProvider(
            model="claude-3-5-sonnet-latest",
            api_key="test-key",
            base_url="https://api.anthropic.com",
            transport=httpx.MockTransport(anthropic),
        ),
        GeminiProvider(
            model="gemini-2.0-flash",
            api_key="test-key",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            transport=httpx.MockTransport(gemini),
        ),
        OllamaProvider(
            model="llama3.2",
            base_url="http://127.0.0.1:11434",
            transport=httpx.MockTransport(ollama),
        ),
    ]
    try:
        for client in clients:
            assert client.complete(USER, schema=Plan) == PLAN_JSON
    finally:
        for client in clients:
            client.close()


def test_transient_error_is_distinct_from_schema_errors() -> None:
    llm = ScriptedLLM([TransientLLMError("timeout"), PLAN])
    assert complete_structured(llm, USER, Plan, attempts=2) == PLAN
