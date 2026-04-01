import logging
import re

from app.github.models import DiffFile, DiffHunk, PullRequestDiff

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────

# start of a new file block
# e.g. "diff --git a/src/auth.py b/src/auth.py"
_DIFF_GIT_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$")

# hunk header
# e.g. "@@ -10,7 +10,8 @@ class AuthService:"
_HUNK_HEADER = re.compile(r"^@@.+@@")

# GitHub's rename header
# e.g. "rename from src/old.py" / "rename to src/new.py"
_RENAME_FROM = re.compile(r"^rename from (.+)$")
_RENAME_TO   = re.compile(r"^rename to (.+)$")

# addition/deletion count lines
# e.g. "--- a/src/auth.py" or "+++ b/src/auth.py"
_FILE_MARKER = re.compile(r"^(\+\+\+|---) [ab]/(.+)$")


# ── Public interface ──────────────────────────────────────────────

def parse_diff(
    raw_diff:   str,
    full_name:  str,
    pr_number:  int,
    commit_sha: str,
) -> PullRequestDiff:
    """
    Parses a raw GitHub unified diff string into a PullRequestDiff.

    Args:
        raw_diff:   Raw unified diff string from GitHub API.
        full_name:  Repository full name e.g. "owner/repo".
        pr_number:  PR number.
        commit_sha: Head commit SHA of the PR.

    Returns:
        PullRequestDiff with one DiffFile per changed file,
        each containing its hunks.
    """
    if not raw_diff or not raw_diff.strip():
        logger.warning(
            f"[diff_parser] empty diff received for "
            f"{full_name} #{pr_number}"
        )
        return PullRequestDiff(
            full_name  = full_name,
            pr_number  = pr_number,
            commit_sha = commit_sha,
        )

    file_blocks = _split_into_file_blocks(raw_diff)
    files       = []

    for block in file_blocks:
        diff_file = _parse_file_block(block)
        if diff_file:
            files.append(diff_file)

    logger.info(
        f"[diff_parser] parsed {len(files)} files "
        f"for {full_name} #{pr_number}"
    )

    return PullRequestDiff(
        full_name  = full_name,
        pr_number  = pr_number,
        commit_sha = commit_sha,
        files      = files,
    )


# ── Private helpers ───────────────────────────────────────────────

def _split_into_file_blocks(raw_diff: str) -> list[str]:
    """
    Splits a full diff string into per-file blocks.

    Each block starts with "diff --git a/... b/..." and ends
    just before the next such line (or at end of string).
    """
    lines  = raw_diff.splitlines(keepends=True)
    blocks = []
    current: list[str] = []

    for line in lines:
        if _DIFF_GIT_HEADER.match(line) and current:
            blocks.append("".join(current))
            current = []
        current.append(line)

    if current:
        blocks.append("".join(current))

    return blocks


def _parse_file_block(block: str) -> DiffFile | None:
    """
    Parses a single per-file diff block into a DiffFile.

    Extracts:
      - filename (new name for renames, otherwise the file path)
      - old_filename (only for renames)
      - status (added / modified / removed / renamed)
      - additions and deletions counts
      - hunks (split at @@ markers)

    Returns None if the block cannot be parsed — caller skips it.
    """
    lines       = block.splitlines()
    filename    = ""
    old_filename = ""
    status      = "modified"
    additions   = 0
    deletions   = 0
    hunks       = []

    # ── Extract filename and status from headers ──────────────────
    for line in lines:
        git_match = _DIFF_GIT_HEADER.match(line)
        if git_match:
            # b/ side is the new filename — correct for adds/modifies
            filename = git_match.group(2)
            continue

        if line.startswith("new file mode"):
            status = "added"
            continue

        if line.startswith("deleted file mode"):
            status = "removed"
            continue

        rename_from = _RENAME_FROM.match(line)
        if rename_from:
            old_filename = rename_from.group(1)
            status       = "renamed"
            continue

        rename_to = _RENAME_TO.match(line)
        if rename_to:
            filename = rename_to.group(1)

    if not filename:
        logger.warning("[diff_parser] could not extract filename from block")
        return None

    # ── Split block into hunks at @@ markers ─────────────────────
    hunks, additions, deletions = _extract_hunks(lines)

    return DiffFile(
        filename     = filename,
        old_filename = old_filename,
        status       = status,
        additions    = additions,
        deletions    = deletions,
        hunks        = hunks,
    )


def _extract_hunks(lines: list[str]) -> tuple[list[DiffHunk], int, int]:
    """
    Splits file block lines into DiffHunk objects at @@ boundaries.
    Also counts total additions and deletions across all hunks.

    Returns (hunks, additions, deletions).
    """
    hunks       : list[DiffHunk] = []
    additions   = 0
    deletions   = 0
    hunk_header = ""
    hunk_lines  : list[str] = []
    in_hunk     = False

    def _save_hunk() -> None:
        if in_hunk and hunk_lines:
            hunks.append(DiffHunk(
                header=hunk_header,
                content="".join(hunk_lines),
            ))

    for line in lines:
        if _HUNK_HEADER.match(line):
            # save previous hunk before starting a new one
            _save_hunk()
            hunk_header = line
            hunk_lines  = []
            in_hunk     = True
            continue

        if in_hunk:
            hunk_lines.append(line + "\n")
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1

    # save the final hunk
    _save_hunk()

    return hunks, additions, deletions