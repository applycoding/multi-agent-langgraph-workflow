import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_TOKEN = re.compile(r"[a-z0-9]+")
_MAX_FILES = 50
_EXCERPT_CHARS = 280


@dataclass(frozen=True)
class WebDocument:
    title: str
    url: str
    body: str


@dataclass(frozen=True)
class Hit:
    title: str
    locator: str
    excerpt: str
    kind: Literal["web", "local_doc"]
    score: int

    def as_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "locator": self.locator,
            "excerpt": self.excerpt,
            "kind": self.kind,
        }


# Fixed corpus used by the web-search stub. These are short original notes,
# not live pages and not a copy of the upstream documentation.
WEB_CORPUS: tuple[WebDocument, ...] = (
    WebDocument(
        title="LangGraph overview",
        url="stub://langgraph/overview",
        body=(
            "LangGraph models a workflow as a graph of nodes and edges. "
            "Each node reads shared state and returns a partial update. "
            "A checkpoint records that state under a thread id."
        ),
    ),
    WebDocument(
        title="Human approval interrupt",
        url="stub://langgraph/interrupts",
        body=(
            "An interrupt pauses a run so a person can approve or reject a draft. "
            "The human approval decision is supplied later, and the same thread resumes "
            "from the checkpoint."
        ),
    ),
    WebDocument(
        title="SQLite checkpointing",
        url="stub://langgraph/checkpoints",
        body=(
            "A checkpointer stores graph state for each thread. "
            "SQLite checkpointing is a local option for demos. "
            "Production systems typically use a server database instead."
        ),
    ),
)


def web_search(
    query: str,
    *,
    corpus: tuple[WebDocument, ...] = WEB_CORPUS,
    limit: int = 3,
) -> list[Hit]:
    """Rank the built-in corpus by token overlap. This does not call the public web."""
    ranked: list[Hit] = []
    for document in corpus:
        score = overlap_score(query, f"{document.title} {document.body}")
        if score <= 0:
            continue
        ranked.append(
            Hit(
                title=document.title,
                locator=document.url,
                excerpt=excerpt(document.body),
                kind="web",
                score=score,
            )
        )
    return _top(ranked, limit)


def search_local_docs(query: str, docs_dir: Path, *, limit: int = 3) -> list[Hit]:
    """Rank markdown files under ``docs_dir`` by token overlap."""
    if not docs_dir.is_dir():
        return []
    ranked: list[Hit] = []
    files = sorted(path for path in docs_dir.rglob("*.md") if path.is_file())[:_MAX_FILES]
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        score = overlap_score(query, f"{path.stem} {text}")
        if score <= 0:
            continue
        ranked.append(
            Hit(
                title=path.stem.replace("-", " "),
                locator=str(path),
                excerpt=excerpt(text),
                kind="local_doc",
                score=score,
            )
        )
    return _top(ranked, limit)


def overlap_score(query: str, text: str) -> int:
    query_terms = _tokens(query)
    if not query_terms:
        return 0
    document_terms = _tokens(text)
    return sum(1 for term in query_terms if _matches(term, document_terms))


def excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(text.lower()) if len(token) > 2}


def _matches(query_term: str, document_terms: set[str]) -> bool:
    for term in document_terms:
        if query_term == term:
            return True
        shorter, longer = (
            (query_term, term) if len(query_term) <= len(term) else (term, query_term)
        )
        if len(shorter) >= 4 and longer.startswith(shorter):
            return True
    return False


def _top(hits: list[Hit], limit: int) -> list[Hit]:
    ordered = sorted(hits, key=lambda hit: (-hit.score, hit.title))
    deduped: list[Hit] = []
    seen: set[tuple[str, str]] = set()
    for hit in ordered:
        key = (hit.kind, hit.locator)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
        if len(deduped) >= limit:
            break
    return deduped
