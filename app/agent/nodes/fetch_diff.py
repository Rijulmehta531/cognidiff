import logging

from app.agent.state import AgentState
from app.github.client import GitHubClient

logger = logging.getLogger(__name__)


async def fetch_diff(state: AgentState) -> AgentState:
    """
    Node 1: Fetches the PR diff from GitHub and stores it in state.

    Reads:  full_name, pr_number, commit_sha
    Writes: pr_diff  — on success
            error    — on failure
    """
    full_name  = state["full_name"]
    pr_number  = state["pr_number"]
    commit_sha = state["commit_sha"]
    prefix     = f"[fetch_diff:{full_name}#{pr_number}]"

    logger.info(f"{prefix} fetching diff — commit {commit_sha[:8]}")

    try:
        client   = GitHubClient()
        pr_diff  = await client.get_pr_diff(
            full_name  = full_name,
            pr_number  = pr_number,
            commit_sha = commit_sha,
        )

        logger.info(
            f"{prefix} diff fetched — "
            f"{len(pr_diff.files)} files, "
            f"+{pr_diff.total_additions} -{pr_diff.total_deletions}"
        )

        return {**state, "pr_diff": pr_diff}

    except Exception as e:
        logger.error(f"{prefix} failed to fetch diff — {e}", exc_info=True)
        return {**state, "error": f"fetch_diff failed: {e}"}