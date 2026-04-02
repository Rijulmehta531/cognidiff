from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

# ── Diff models ────────────────────────────────
# These models are written as dataclasses as they are used internally
#  within the codebase and don't need the validation or serialization

@dataclass
class DiffHunk:
    """
    A single contiguous block of changes within a file.

    One file can have multiple hunks — e.g. a change at line 10
    and another at line 80 produce two separate hunks.
    Each hunk includes a few lines of surrounding context
    so the LLM can reason about what the change sits next to.
    """
    header:  str        # raw @@ -x,y +x,y @@ line
    content: str        # full hunk text including context lines


@dataclass
class DiffFile:
    """
    All changes to a single file within a PR.

    status follows GitHub's convention:
      added    — new file
      modified — existing file changed
      removed  — file deleted
      renamed  — file moved, possibly with content changes
    """
    filename:    str
    status:      str              # added | modified | removed | renamed
    additions:   int
    deletions:   int
    hunks:       list[DiffHunk]  = field(default_factory=list)
    old_filename: str            = ""   # populated for renamed files only


@dataclass
class PullRequestDiff:
    """
    Structured representation of a PR diff.

    Produced by - diff_parser.py from the raw GitHub API response.
    Consumed by - rag_lookup.py to construct retrieval queries,
    and by the analyze node to reason over what changed.
    """
    full_name:  str
    pr_number:  int
    commit_sha: str
    files:      list[DiffFile] = field(default_factory=list)

    @property
    def changed_filenames(self) -> set[str]:
        """
        Set of all filenames touched by the PR.
        Used by rag_lookup.py to filter out stale retrieved
        chunks from files the PR itself already changes.
        """
        return {f.filename for f in self.files}

    @property
    def total_additions(self) -> int:
        return sum(f.additions for f in self.files)

    @property
    def total_deletions(self) -> int:
        return sum(f.deletions for f in self.files)


# ── Review models ─────────────────────────────────────────────────
# These are LLM output models — passed directly to
# llm.with_structured_output() which requires a Pydantic schema
# to generate the JSON schema for constrained decoding.

class ReviewComment(BaseModel):
    """
    A single inline comment on a specific line of the diff.

    path and line together identify exactly where in the PR
    the comment should appear.
    """
    path: str = Field(
        description=(
            "File path relative to the repo root, "
            "e.g. 'src/auth/validator.py'"
        )
    )
    line: int = Field(
        description=(
            "Line number in the diff where the comment should appear. "
            "Must be a line visible in the diff, not an arbitrary file line."
        )
    )
    body: str = Field(
        description=(
            "The comment text. Be precise and actionable — "
            "explain what the issue is and how to fix it."
        )
    )


class PullRequestReview(BaseModel):
    """
    The complete review the agent posts back to GitHub.

    event controls the review outcome:
      APPROVE          — agent thinks the PR is good to merge
      REQUEST_CHANGES  — agent found issues that must be addressed
      COMMENT          — agent has observations but no strong verdict
    """
    body: str = Field(
        description=(
            "The overall review summary shown at the top of the GitHub "
            "review. Should cover: what the PR does, key findings, "
            "and the reasoning behind the review decision. "
            "If codebase context was unavailable during retrieval, "
            "note that the review is based on the diff alone."
        )
    )
    event: Literal["APPROVE", "REQUEST_CHANGES", "COMMENT"] = Field(
        description=(
            "The review decision. Use APPROVE if no significant issues "
            "were found. Use REQUEST_CHANGES if issues must be addressed "
            "before merge. Use COMMENT for observations without a "
            "blocking verdict."
        )
    )
    comments: list[ReviewComment] = Field(
        default_factory=list,
        description=(
            "Inline comments on specific diff lines. Only include comments "
            "where you have something precise and actionable to say. "
            "Prefer fewer high-quality comments over many shallow ones."
        )
    )