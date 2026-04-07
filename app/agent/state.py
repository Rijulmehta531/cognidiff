import uuid
from typing import Optional, TypedDict

from app.github.models import PullRequestDiff, PullRequestReview


class AgentState(TypedDict):
    #  Worker-supplied inputs
    full_name:  str
    pr_number:  int
    pr_title:   str
    commit_sha: str
    repo_id:    uuid.UUID

    #  Node outputs
    pr_diff:           Optional[PullRequestDiff]
    retrieved_context: Optional[str]
    retrieval_skipped: bool
    review:            Optional[PullRequestReview]
    error:             Optional[str]