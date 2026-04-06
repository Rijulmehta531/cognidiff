import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.config import get_llm
from app.exceptions import TransientError
from app.github.models import PullRequestReview
from app.agent.prompts import (
    format_diff_for_prompt,
    review_human_prompt,
    review_system_prompt,
)

logger = logging.getLogger(__name__)


async def analyze(state: AgentState) -> AgentState:
    """
    Node 3: Runs LLM analysis over the diff and retrieved context.
    Produces a structured PullRequestReview.

    Reads:  pr_diff, pr_title, retrieved_context, retrieval_skipped
    Writes: review   — on success
            error    — on permanent failure (empty body)

    Raises: TransientError — if the LLM call itself throws.
            ARQ catches this and schedules a retry for the whole job.
            We do not write to state["error"] in this case because
            the graph never gets to route — the exception propagates up.
    """
    full_name  = state["full_name"]
    pr_number  = state["pr_number"]
    prefix     = f"[analyze:{full_name}#{pr_number}]"

    logger.info(f"{prefix} starting LLM analysis")

    formatted_diff = format_diff_for_prompt(state["pr_diff"])

    messages = [
        SystemMessage(content=review_system_prompt()),
        HumanMessage(content=review_human_prompt(
            pr_title          = state["pr_title"],
            pr_diff           = formatted_diff,
            retrieved_context = state["retrieved_context"] or "",
            retrieval_skipped = state["retrieval_skipped"],
        )),
    ]

    try:
        llm    = get_llm()
        chain  = llm.with_structured_output(PullRequestReview)
        review = await chain.ainvoke(messages)

    except Exception as e:
        # LLM call failed — network, timeout, model unavailable.
        # Re-raise as TransientError so ARQ retries the whole job.
        logger.error(f"{prefix} LLM call failed — {e}", exc_info=True)
        raise TransientError(f"LLM call failed: {e}") from e

    # Valid structure but empty review body — treat as permanent failure (bad output).
    if not review.body.strip():
        logger.error(f"{prefix} LLM returned empty review body")
        return {
            **state,
            "error": "analyze failed: LLM returned an empty review body",
        }

    logger.info(
        f"{prefix} analysis complete — "
        f"event: {review.event}, "
        f"inline comments: {len(review.comments)}"
    )

    return {**state, "review": review}