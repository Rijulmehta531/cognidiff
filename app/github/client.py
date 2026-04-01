import logging

import httpx

from app.config import get_settings
from app.exceptions import (
    AuthenticationError,
    GitHubRateLimitError,
    GitHubUnavailableError,
    RepoNotFoundError,
)
from app.github.diff_parser import parse_diff
from app.github.models import PullRequestDiff, PullRequestReview

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


class GitHubClient:
    """
    Responsibilities:
      - Fetch PR diffs from GitHub
      - Post completed reviews back to GitHub
      - Map HTTP errors to typed exceptions
    """

    def __init__(self):
        self.settings = get_settings()
        self._headers = {
            "Authorization":        f"Bearer {self.settings.GITHUB_TOKEN}",
            "X-GitHub-API-Version": self.settings.GITHUB_API_VERSION,
        }

    # ── Public interface ──────────────────────────────────────────

    async def get_pr_diff(
        self,
        full_name: str,
        pr_number: int,
        commit_sha: str,
    ) -> PullRequestDiff:
        """
        Fetches the diff for a PR and returns it as a structured PullRequestDiff

        Args:
            full_name:  Repository full name e.g. "owner/repo".
            pr_number:  PR number.
            commit_sha: Head commit SHA

        Returns:
            PullRequestDiff with one DiffFile per changed file.

        Raises:
            AuthenticationError:    GitHub returned 401.
            RepoNotFoundError:      GitHub returned 404.
            GitHubRateLimitError:   GitHub returned 429.
            GitHubUnavailableError: GitHub returned 503/504.
        """
        url = f"{GITHUB_API}/repos/{full_name}/pulls/{pr_number}"

        logger.info(
            f"[github_client] fetching diff — "
            f"{full_name} #{pr_number}"
        )

        headers = {
            **self._headers,
            "Accept": "application/vnd.github.diff",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            _raise_for_github_status(response)
            raw_diff = response.text

        logger.info(
            f"[github_client] diff received — "
            f"{full_name} #{pr_number} "
            f"({len(raw_diff):,} bytes)"
        )

        return parse_diff(
            raw_diff   = raw_diff,
            full_name  = full_name,
            pr_number  = pr_number,
            commit_sha = commit_sha,
        )

    async def post_review(
        self,
        full_name:  str,
        pr_number:  int,
        commit_sha: str,
        review:     PullRequestReview,
    ) -> None:
        """
        Posts a completed review to GitHub.

        Args:
            full_name:  Repository full name e.g. "owner/repo".
            pr_number:  PR number.
            commit_sha: Head commit SHA the review is anchored to.
            review:     Structured review from the agent.

        Raises:
            AuthenticationError:    GitHub returned 401.
            RepoNotFoundError:      GitHub returned 404.
            GitHubRateLimitError:   GitHub returned 429.
            GitHubUnavailableError: GitHub returned 503/504.
        """
        url = (
            f"{GITHUB_API}/repos/{full_name}"
            f"/pulls/{pr_number}/reviews"
        )

        payload = _build_review_payload(commit_sha, review)

        logger.info(
            f"[github_client] posting review — "
            f"{full_name} #{pr_number} "
            f"event={review.event} "
            f"comments={len(review.comments)}"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            headers  = {
                **self._headers,
                "Accept": "application/vnd.github+json",
            }
            response = await client.post(url, headers=headers, json=payload)
            _raise_for_github_status(response)

        logger.info(
            f"[github_client] review posted — "
            f"{full_name} #{pr_number}"
        )


# ─helpers ───────────────────────────────────────────────

def _raise_for_github_status(response: httpx.Response) -> None:
    """
    Maps GitHub HTTP error codes to typed exceptions.
    """
    if response.status_code == 401:
        raise AuthenticationError(
            f"invalid or expired GitHub token "
            f"(HTTP 401 for {response.url})"
        )
    if response.status_code == 404:
        raise RepoNotFoundError(
            f"repo or PR not found — check token permissions "
            f"(HTTP 404 for {response.url})"
        )
    if response.status_code == 429:
        raise GitHubRateLimitError(
            f"GitHub API rate limit exceeded "
            f"(HTTP 429 for {response.url})"
        )
    if response.status_code in (503, 504):
        raise GitHubUnavailableError(
            f"GitHub temporarily unavailable "
            f"(HTTP {response.status_code} for {response.url})"
        )
    response.raise_for_status()


def _build_review_payload(
    commit_sha: str,
    review:     PullRequestReview,
) -> dict:
    """
    Serialises a PullRequestReview into the shape GitHub's
    review API expects.

    """
    payload: dict = {
        "commit_id": commit_sha,
        "body":      review.body,
        "event":     review.event,
    }

    if review.comments:
        payload["comments"] = [
            {
                "path":     comment.path,
                "line":     comment.line,
                "body":     comment.body,
            }
            for comment in review.comments
        ]

    return payload