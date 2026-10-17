from workflow.models import Report, Section, Source
from workflow.state import WorkflowState

DEMO_CAVEAT = (
    "This report was produced by a portfolio demo. "
    "Web hits come from a fixed stub corpus, not a live search."
)


def build_report(state: WorkflowState) -> Report:
    topic = str(state.get("topic") or "unknown topic")
    audience = str(state.get("audience") or "")
    plan = state.get("plan") or {}
    draft = state.get("draft") or {}
    critique = state.get("critique") or {}
    approval = state.get("approval") or {}
    sections = draft.get("sections") or [
        {
            "heading": "Status",
            "body": "The workflow stopped before a draft was written.",
        }
    ]
    caveats = list(draft.get("caveats") or [])
    if DEMO_CAVEAT not in caveats:
        caveats.append(DEMO_CAVEAT)
    errors = [str(item.get("message", item)) for item in state.get("errors") or []]
    return Report(
        title=str(draft.get("title") or topic),
        topic=topic,
        audience=audience,
        thesis=str(plan.get("thesis") or ""),
        summary=str(draft.get("summary") or ""),
        sections=[Section.model_validate(section) for section in sections],
        sources=[Source.model_validate(source) for source in state.get("sources") or []],
        caveats=caveats,
        critique_notes=str(critique.get("notes") or ""),
        approved=bool(approval.get("approved", False)),
        revision_count=int(state.get("revision_count") or 0),
        errors=errors,
    )
