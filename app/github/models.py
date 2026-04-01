from dataclasses import dataclass, field


# ── Diff models ───────────────────────────────────────────────────

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

@dataclass
class ReviewComment:
    """
    A single inline comment on a specific line of the diff.

    path and line together identify exactly where in the PR
    the comment should appear.
    """
    path: str    # file path relative to repo root
    line: int    # line number in the diff (not the file)
    body: str    # comment text


@dataclass
class PullRequestReview:
    """
    The complete review the agent posts back to GitHub.

    event controls the review outcome:
      APPROVE          — agent thinks the PR is good to merge
      REQUEST_CHANGES  — agent found issues that must be addressed
      COMMENT          — agent has observations but no strong verdict
    """
    body:     str
    event:    str                        # APPROVE | REQUEST_CHANGES | COMMENT
    comments: list[ReviewComment] = field(default_factory=list)