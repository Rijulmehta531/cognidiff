import logging

from app.agent.state import AgentState
from app.github.client import GitHubClient

logger = logging.getLogger(__name__)

async def post_review(state: AgentState) -> AgentState:
    """
    Node 4: Posts the structured review back to GitHub.

    Reads:  review, full_name, pr_number, commit_sha
    Writes: error — on failure
    """
    review    = state["review"]
    full_name = state["full_name"]
    pr_number = state["pr_number"]
    prefix    = f"[post_review:{full_name}#{pr_number}]"

    logger.info(
        f"{prefix} posting review — "
        f"event: {review.event}, "
        f"inline comments: {len(review.comments)}"
    )

    client = GitHubClient()

    try:
        await client.post_review(
            full_name  = full_name,
            pr_number  = pr_number,
            commit_sha = state["commit_sha"],
            review     = review,
        )
        logger.info(f"{prefix} review posted successfully")
        return state

    except Exception as e:
        if not _is_422(e):
            logger.error(f"{prefix} failed to post review — {e}", exc_info=True)
            return {**state, "error": f"post_review failed: {e}"}

        # 422 — at least one inline comment has an invalid line number.
        # Strip comments and retry so the top-level review still lands.
        logger.warning(
            f"{prefix} GitHub returned 422 — likely invalid diff line numbers. "
            f"retrying without inline comments. "
            f"dropped comments: {[c.model_dump() for c in review.comments]}"
        )

        stripped_review = review.model_copy(update={"comments": []})

        try:
            await client.post_review(
                full_name  = full_name,
                pr_number  = pr_number,
                commit_sha = state["commit_sha"],
                review     = stripped_review,
            )
            logger.info(
                f"{prefix} review posted without inline comments after 422 retry"
            )
            return state

        except Exception as retry_e:
            logger.error(
                f"{prefix} stripped retry also failed — {retry_e}",
                exc_info=True,
            )
            return {
                **state,
                "error": f"post_review failed after 422 retry: {retry_e}",
            }


def _is_422(exc: Exception) -> bool:
    return "422" in str(exc)