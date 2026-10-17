from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from workflow.nodes import (
    Dependencies,
    run_approve,
    run_fail,
    run_planner,
    run_report,
    run_researcher,
    run_reviewer,
    run_writer,
)
from workflow.state import WorkflowState

PLAN = "plan"
RESEARCH = "research"
WRITE = "write"
REVIEW = "review"
APPROVE = "approve"
REPORT = "report"
FAIL = "fail"


def route_after_step(state: WorkflowState, *, ready: str) -> str:
    if state.get("status") == "failed":
        return FAIL
    return ready


def route_after_approval(state: WorkflowState) -> str:
    if state.get("status") == "failed":
        return FAIL
    approval = state.get("approval") or {}
    if approval.get("approved") is True:
        return REPORT
    revisions = int(state.get("revision_count") or 0)
    limit = int(state.get("max_revisions") or 0)
    if revisions <= limit:
        return WRITE
    return REPORT


def route_plan(state: WorkflowState) -> str:
    return route_after_step(state, ready=RESEARCH)


def route_research(state: WorkflowState) -> str:
    return route_after_step(state, ready=WRITE)


def route_write(state: WorkflowState) -> str:
    return route_after_step(state, ready=REVIEW)


def route_review(state: WorkflowState) -> str:
    return route_after_step(state, ready=APPROVE)


def build_graph(
    deps: Dependencies,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    def plan(state: WorkflowState) -> dict:
        return run_planner(state, deps)

    def research(state: WorkflowState) -> dict:
        return run_researcher(state, deps)

    def write(state: WorkflowState) -> dict:
        return run_writer(state, deps)

    def review(state: WorkflowState) -> dict:
        return run_reviewer(state, deps)

    graph = StateGraph(WorkflowState)
    graph.add_node(PLAN, plan)
    graph.add_node(RESEARCH, research)
    graph.add_node(WRITE, write)
    graph.add_node(REVIEW, review)
    graph.add_node(APPROVE, run_approve)
    graph.add_node(REPORT, run_report)
    graph.add_node(FAIL, run_fail)

    graph.add_edge(START, PLAN)
    graph.add_conditional_edges(PLAN, route_plan, {RESEARCH: RESEARCH, FAIL: FAIL})
    graph.add_conditional_edges(RESEARCH, route_research, {WRITE: WRITE, FAIL: FAIL})
    graph.add_conditional_edges(WRITE, route_write, {REVIEW: REVIEW, FAIL: FAIL})
    graph.add_conditional_edges(REVIEW, route_review, {APPROVE: APPROVE, FAIL: FAIL})
    graph.add_conditional_edges(
        APPROVE,
        route_after_approval,
        {REPORT: REPORT, WRITE: WRITE, FAIL: FAIL},
    )
    graph.add_edge(REPORT, END)
    graph.add_edge(FAIL, END)
    return graph.compile(checkpointer=checkpointer)
