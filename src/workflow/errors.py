class WorkflowError(Exception):
    """Base error for workflow control flow."""


class InvalidRequest(WorkflowError):
    """The caller sent a topic or thread id this demo does not accept."""


class RunNotFound(WorkflowError):
    """No checkpoint exists for the requested thread."""


class RunExists(WorkflowError):
    """A thread id is already in use."""


class RunNotWaiting(WorkflowError):
    """Resume was requested, but the run is not paused for approval."""


class ConfigurationError(WorkflowError):
    """Provider settings are missing or inconsistent."""


class LLMError(WorkflowError):
    """A model provider failed."""


class TransientLLMError(LLMError):
    """A model call failed in a way that a bounded retry may fix."""


class AuthenticationError(LLMError):
    """The provider rejected the configured credentials."""


class StructuredOutputError(LLMError):
    """A model response did not match the expected schema."""
