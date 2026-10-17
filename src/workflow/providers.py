import json
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel

from workflow.errors import AuthenticationError, LLMError, TransientLLMError
from workflow.models import ChatMessage

_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
_AUTH_STATUS = {401, 403}
_USER_AGENT = "multi-agent-langgraph-workflow/0.1.0"


class HTTPChatProvider:
    """Shared HTTP plumbing for hosted and local chat providers."""

    name = "http"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 45.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        schema: type[BaseModel] | None = None,
    ) -> str:
        try:
            response = self._client.post(
                self._url(),
                headers=self._headers(),
                json=self._body(messages, schema),
            )
        except httpx.TimeoutException as exc:
            raise TransientLLMError(f"{self.name} request timed out") from exc
        except httpx.TransportError as exc:
            raise TransientLLMError(f"{self.name} request failed: {exc}") from exc
        raise_for_status(response, provider=self.name)
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise LLMError(f"{self.name} returned a non-JSON body") from exc
        return self._parse(payload)

    def _url(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        return {"user-agent": _USER_AGENT, "content-type": "application/json"}

    def _body(self, messages: list[ChatMessage], schema: type[BaseModel] | None) -> dict[str, Any]:
        raise NotImplementedError

    def _parse(self, payload: dict[str, Any]) -> str:
        raise NotImplementedError


class OpenAIProvider(HTTPChatProvider):
    name = "openai"

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _body(self, messages: list[ChatMessage], schema: type[BaseModel] | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_payload(message) for message in messages],
        }
        if schema is not None:
            body["response_format"] = {"type": "json_object"}
        return body

    def _parse(self, payload: dict[str, Any]) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("OpenAI response did not include message content") from exc
        return text_content(content)


class AnthropicProvider(HTTPChatProvider):
    name = "anthropic"

    def _url(self) -> str:
        return f"{self.base_url}/v1/messages"

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        headers["x-api-key"] = self.api_key or ""
        headers["anthropic-version"] = "2023-06-01"
        return headers

    def _body(self, messages: list[ChatMessage], schema: type[BaseModel] | None) -> dict[str, Any]:
        del schema
        system, chat = split_system(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": chat,
        }
        if system:
            body["system"] = system
        return body

    def _parse(self, payload: dict[str, Any]) -> str:
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise LLMError("Anthropic response did not include content blocks")
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text:
            raise LLMError("Anthropic response did not include text")
        return text


class GeminiProvider(HTTPChatProvider):
    name = "gemini"

    def _url(self) -> str:
        key = quote(self.api_key or "", safe="")
        return f"{self.base_url}/models/{self.model}:generateContent?key={key}"

    def _body(self, messages: list[ChatMessage], schema: type[BaseModel] | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "contents": gemini_contents(messages),
        }
        if schema is not None:
            body["generationConfig"] = {"responseMimeType": "application/json"}
        return body

    def _parse(self, payload: dict[str, Any]) -> str:
        try:
            parts = payload["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Gemini response did not include content parts") from exc
        text = "".join(
            part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")
        )
        if not text:
            raise LLMError("Gemini response did not include text")
        return text


class OllamaProvider(HTTPChatProvider):
    name = "ollama"

    def _url(self) -> str:
        return f"{self.base_url}/api/chat"

    def _body(self, messages: list[ChatMessage], schema: type[BaseModel] | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [_message_payload(message) for message in messages],
        }
        if schema is not None:
            body["format"] = "json"
        return body

    def _parse(self, payload: dict[str, Any]) -> str:
        try:
            content = payload["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError("Ollama response did not include message content") from exc
        return text_content(content)


def _message_payload(message: ChatMessage) -> dict[str, str]:
    return {"role": message.role, "content": message.content}


def raise_for_status(response: httpx.Response, *, provider: str) -> None:
    if response.is_success:
        return
    detail = response.text[:200]
    message = f"{provider} request failed with HTTP {response.status_code}: {detail}"
    if response.status_code in _TRANSIENT_STATUS:
        raise TransientLLMError(message)
    if response.status_code in _AUTH_STATUS:
        raise AuthenticationError(message)
    raise LLMError(message)


def text_content(content: Any) -> str:
    if isinstance(content, str):
        if content.strip():
            return content
        raise LLMError("Model response content was empty")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        text = "".join(parts)
        if text.strip():
            return text
    raise LLMError("Model response did not include text content")


def split_system(messages: list[ChatMessage]) -> tuple[str, list[dict[str, str]]]:
    system_parts = [message.content for message in messages if message.role == "system"]
    chat: list[dict[str, str]] = []
    for message in messages:
        if message.role == "system":
            continue
        role = "assistant" if message.role == "assistant" else "user"
        chat.append({"role": role, "content": message.content})
    if not chat:
        chat.append({"role": "user", "content": "Respond with the requested JSON."})
    return "\n\n".join(system_parts), chat


def gemini_contents(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = "model" if message.role == "assistant" else "user"
        text = message.content if message.role != "system" else f"Instructions:\n{message.content}"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += "\n\n" + text
            continue
        contents.append({"role": role, "parts": [{"text": text}]})
    if not contents:
        contents.append({"role": "user", "parts": [{"text": "Respond with the requested JSON."}]})
    return contents
