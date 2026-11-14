"""The LangGraph graph: typed state, 14 nodes, conditional edges, chat() and resume().

Read build_graph() top to bottom: it is the whole flow from docs/DESIGN.md section 2.1.
"""

from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing_extensions import TypedDict

from . import agents
from .config import USERS


class CodeForgeState(TypedDict, total=False):
    """Everything the graph carries. Identity is NOT here: it lives in config only."""
    # kept between messages in the same chat (thread)
    messages: Annotated[list, add_messages]
    summary: str
    unhappy_streak: int
    strikes: int
    # reset at the start of every message by guard_in
    request: str
    pii_found: list
    quota_left: int
    blocked: str
    triage: dict
    escalate_reason: str
    route: str
    requirements: dict
    code: str
    tests: str
    served_by: dict
    cache_hit: str
    policy: dict
    approved: bool
    run: dict
    attempts: int
    repair: bool
    refusal: dict
    ticket: dict
    sql: dict
    answer: str


# ── edge functions: small, readable decisions ────────────────────────────────

def after_guard(state) -> str:
    return "refuse" if state.get("blocked") else "triage"


def after_triage(state) -> str:
    return "ticket" if state.get("escalate_reason") else "cache_lookup"


def after_cache(state) -> str:
    return "policy_gate" if state.get("cache_hit") else "router"


def after_router(state) -> str:
    return {"python_function": "requirements", "sql_query": "sql_agent"}.get(state.get("route"), "refuse")


def after_code_gen(state) -> str:
    return "refuse" if state.get("refusal") else "test_gen"


def after_test_gen(state) -> str:
    return "refuse" if state.get("refusal") else "policy_gate"


def after_policy(state) -> str:
    verdict = state["policy"]["verdict"]
    return {"allow": "run_tests", "needs_approval": "human_approval"}.get(verdict, "refuse")


def after_approval(state) -> str:
    return "run_tests" if state.get("approved") else "refuse"


def after_run(state) -> str:
    return "code_gen" if state.get("repair") else "report"


def needs_ticket(state) -> str:
    return "ticket" if state.get("escalate_reason") else END


def build_graph(checkpointer=None):
    """Assemble the CodeForge graph (MemorySaver keeps paused and multi-turn chats)."""
    g = StateGraph(CodeForgeState)
    for name in ("guard_in", "triage", "cache_lookup", "router", "requirements", "code_gen", "test_gen",
                 "policy_gate", "human_approval", "run_tests", "report", "refuse", "ticket", "sql_agent"):
        g.add_node(name, getattr(agents, name))

    g.add_edge(START, "guard_in")
    g.add_conditional_edges("guard_in", after_guard, ["triage", "refuse"])
    g.add_conditional_edges("triage", after_triage, ["ticket", "cache_lookup"])
    g.add_conditional_edges("cache_lookup", after_cache, ["policy_gate", "router"])
    g.add_conditional_edges("router", after_router, ["requirements", "sql_agent", "refuse"])
    g.add_edge("requirements", "code_gen")
    g.add_conditional_edges("code_gen", after_code_gen, ["test_gen", "refuse"])
    g.add_conditional_edges("test_gen", after_test_gen, ["policy_gate", "refuse"])
    g.add_conditional_edges("policy_gate", after_policy, ["run_tests", "human_approval", "refuse"])
    g.add_conditional_edges("human_approval", after_approval, ["run_tests", "refuse"])
    g.add_conditional_edges("run_tests", after_run, ["code_gen", "report"])
    g.add_conditional_edges("report", needs_ticket, ["ticket", END])
    g.add_conditional_edges("refuse", needs_ticket, ["ticket", END])
    g.add_edge("ticket", END)
    g.add_edge("sql_agent", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


_APP = None


def get_app():
    """The shared graph used by the console scripts and the UI."""
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def make_config(user_id: str, thread: str, role: str | None = None, callbacks=None, **extra) -> dict:
    """Identity goes into config["configurable"]. The chat text can never change it."""
    role = role or USERS.get(user_id, "student")
    cfg = {"configurable": {"thread_id": thread, "user_id": user_id, "role": role, **extra}}
    if callbacks:
        cfg["callbacks"] = callbacks
    return cfg


def chat(app, text: str, user_id: str = "u_alice", thread: str = "main", role: str | None = None, callbacks=None):
    """Send one message and return the final state (with "__interrupt__" if it paused for approval)."""
    cfg = make_config(user_id, thread, role, callbacks)
    return app.invoke({"messages": [HumanMessage(content=text)]}, cfg)


def resume(app, approved: bool, user_id: str, thread: str, approver_id: str = "u_carol",
           approver_role: str = "instructor", role: str | None = None, callbacks=None):
    """Resume a paused graph with the approver's decision. The approver's role comes from config."""
    cfg = make_config(user_id, thread, role, callbacks, approver_id=approver_id, approver_role=approver_role)
    return app.invoke(Command(resume={"approved": approved}), cfg)


def is_paused(state: dict) -> bool:
    """True when the graph stopped at human_approval and waits for a decision."""
    return bool(state.get("__interrupt__"))
