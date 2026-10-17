from workflow.errors import TransientLLMError
from workflow.llm import FakeLLM
from workflow.models import ChatMessage


class ScriptedLLM:
    """Return queued results, or raise queued exceptions, from ``complete``."""

    name = "scripted"

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        schema: type | None = None,
    ) -> object:
        self.calls += 1
        if not self.outcomes:
            raise RuntimeError("script exhausted")
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(messages, schema)
        return item


class FlakyFake(FakeLLM):
    """Fail a fixed number of times, then behave like ``FakeLLM``."""

    def __init__(self, failures: int) -> None:
        self._remaining = failures
        self.calls = 0

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        schema: type | None = None,
    ):
        self.calls += 1
        if self._remaining:
            self._remaining -= 1
            raise TransientLLMError("temporary outage")
        return super().complete(messages, schema=schema)
