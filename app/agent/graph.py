import logging

from langgraph.graph import END, StateGraph

from app.agent.state import AgentState
from app.agent.nodes.fetch_diff import fetch_diff
from app.agent.nodes.rag_lookup import rag_lookup
from app.agent.nodes.analyze import analyze
from app.agent.nodes.post_review import post_review

logger = logging.getLogger(__name__)


# ── Router ────────────────────────────────────────────────────────

def _route(state: AgentState) -> str:
    """
    Shared conditional edge used after every node.

    If any node writes to state["error"], the graph routes to END
    immediately.

    Returns "continue" or "end"
    """
    if state.get("error"):
        logger.warning(
            f"[graph:{state['full_name']}#{state['pr_number']}] "
            f"routing to END — {state['error']}"
        )
        return "end"
    return "continue"


# --- Graph ----------------
def build_graph() -> StateGraph:
    """
    Constructs and compiles the PR review LangGraph state machine.

    Flow:
        fetch_diff → rag_lookup → analyze → post_review → END
    """
    graph = StateGraph(AgentState)

    # --- Nodes -------------------------------
    graph.add_node("fetch_diff",  fetch_diff)
    graph.add_node("rag_lookup",  rag_lookup)
    graph.add_node("analyze",     analyze)
    graph.add_node("post_review", post_review)

    # --- Entry point -------------------------------
    graph.set_entry_point("fetch_diff")

    # --- Conditional edges -------------------------------
    # Same router function reused after every node.
    # "continue" maps to the next node, "end" maps to END.
    graph.add_conditional_edges(
        "fetch_diff",
        _route,
        {"continue": "rag_lookup", "end": END},
    )
    graph.add_conditional_edges(
        "rag_lookup",
        _route,
        {"continue": "analyze", "end": END},
    )
    graph.add_conditional_edges(
        "analyze",
        _route,
        {"continue": "post_review", "end": END},
    )
    graph.add_conditional_edges(
        "post_review",
        _route,
        {"continue": END, "end": END},
    )

    return graph.compile()

review_graph = build_graph()