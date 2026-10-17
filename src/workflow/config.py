import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from workflow.errors import ConfigurationError


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    FAKE = "fake"


def parse_provider(value: str) -> Provider:
    try:
        return Provider(value.strip().lower())
    except ValueError as exc:
        known = ", ".join(provider.value for provider in Provider)
        raise ConfigurationError(
            f"Unknown LLM provider {value!r}. Expected one of: {known}."
        ) from exc


def default_docs_dir() -> Path:
    candidates = [Path.cwd() / "docs", Path(__file__).resolve().parents[2] / "docs"]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.md")):
            return candidate
    return candidates[0]


@dataclass(frozen=True)
class Settings:
    provider: Provider = Provider.FAKE
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-latest"
    anthropic_base_url: str = "https://api.anthropic.com"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://127.0.0.1:11434"
    checkpoint_path: str | None = "data/checkpoints.sqlite"
    docs_dir: Path = Path("docs")
    max_revisions: int = 2
    llm_attempts: int = 3
    timeout_seconds: float = 45.0


def settings_from_env(environ: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if environ is None else environ
    docs_raw = _optional(env, "DOCS_DIR")
    checkpoint_raw = _optional(env, "CHECKPOINT_PATH")
    if checkpoint_raw is None:
        checkpoint_path: str | None = "data/checkpoints.sqlite"
    elif checkpoint_raw.lower() == "memory":
        checkpoint_path = None
    else:
        checkpoint_path = checkpoint_raw
    return Settings(
        provider=parse_provider(_text(env, "LLM_PROVIDER", Provider.FAKE.value)),
        openai_api_key=_optional(env, "OPENAI_API_KEY"),
        openai_model=_text(env, "OPENAI_MODEL", "gpt-4o-mini"),
        openai_base_url=_text(env, "OPENAI_BASE_URL", "https://api.openai.com/v1"),
        anthropic_api_key=_optional(env, "ANTHROPIC_API_KEY"),
        anthropic_model=_text(env, "ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        anthropic_base_url=_text(env, "ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        gemini_api_key=_optional(env, "GEMINI_API_KEY"),
        gemini_model=_text(env, "GEMINI_MODEL", "gemini-2.0-flash"),
        gemini_base_url=_text(
            env,
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
        ollama_model=_text(env, "OLLAMA_MODEL", "llama3.2"),
        ollama_base_url=_text(env, "OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        checkpoint_path=checkpoint_path,
        docs_dir=Path(docs_raw) if docs_raw else default_docs_dir(),
        max_revisions=_int_env(env, "MAX_REVISIONS", 2, minimum=0, maximum=5),
        llm_attempts=_int_env(env, "LLM_ATTEMPTS", 3, minimum=1, maximum=6),
        timeout_seconds=_float_env(env, "LLM_TIMEOUT_SECONDS", 45.0),
    )


def _optional(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    return _optional(env, name) or default


def _float_env(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _optional(env, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number.") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than 0.")
    return value


def _int_env(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = _optional(env, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if value < minimum or value > maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}.")
    return value
