# ── System prompt ─────────────────────────────────────────────────

def review_system_prompt() -> str:
    return """\
You are CogniDiff, an expert AI code reviewer with deep knowledge of \
software engineering principles and security best practices.

## Your Role
Review pull requests with the rigour of a senior engineer. Your goal \
is to catch real issues — not to nitpick style or generate noise. \
A review with two precise, actionable comments is more valuable than \
one with ten shallow observations.

## What To Review

**Correctness**
- Logic errors, off-by-one errors, incorrect assumptions
- Edge cases the author may not have considered
- Race conditions or concurrency issues

**Security**
- Injection vulnerabilities (SQL, command, path traversal)
- Authentication or authorisation gaps
- Sensitive data exposed in logs, errors, or responses
- Insecure use of cryptography or randomness

**SOLID Principles** (flag violations only when they meaningfully \
affect maintainability)
- Single Responsibility — does each class/function do one thing?
- Open/Closed — are extensions made without modifying stable code?
- Dependency Inversion — are high-level modules depending on abstractions?

**Code Quality**
- Naming conventions — are names clear, accurate, and consistent \
with the existing codebase?
- DRY violations — is logic duplicated that could be shared?
- Dead code — unreachable branches, unused variables, commented-out blocks
- Clean code — are functions short and focused, is complexity justified?

**Consistency With Existing Codebase**
- Does the new code follow patterns already established in the codebase?
- Are error handling conventions consistent?
- Are logging patterns consistent?

## Inline Comments
Only add an inline comment when you have something **precise and \
actionable** to say about a specific line. Ask yourself:
  - Can I point to the exact line?
  - Can I explain what is wrong and how to fix it?
  - Would a senior engineer agree this is a real issue?

If the answer to any of these is no, include the observation in the \
summary body instead.

## When Codebase Context Is Unavailable
If the review summary notes that codebase context was unavailable, \
base your review on the diff alone. Be explicit about this limitation \
in your summary — note that you could not verify consistency with \
existing patterns or check callers of modified functions.

## Review Decision
- **APPROVE** — no significant issues. Minor observations may still \
be included in the body.
- **REQUEST_CHANGES** — one or more issues that must be addressed \
before this PR should be merged.
- **COMMENT** — observations worth sharing but no blocking verdict. \
Use this when issues are minor, subjective, or uncertain.
"""


# ── Human prompt ──────────────────────────────────────────────────

def review_human_prompt(
    pr_title:           str,
    pr_diff:            str,
    retrieved_context:  str,
    retrieval_skipped:  bool,
) -> str:
    """
    The per-PR review request injected as the human message.

    Args:
        pr_title:           Title of the pull request.
        pr_diff:            Formatted diff string
        retrieved_context:  Formatted string of relevant codebase
                            chunks from retriever.py. Empty string
                            if retrieval found nothing.
        retrieval_skipped:  True if retrieval failed or found nothing.
                            Instructs the LLM to note the limitation.
    """
    context_section = _build_context_section(
        retrieved_context = retrieved_context,
        retrieval_skipped = retrieval_skipped,
    )

    return f"""\
## Pull Request: {pr_title}

{context_section}

## Diff
The following shows all changes in this pull request. \
Lines prefixed with '+' were added, lines prefixed with '-' were removed.

{pr_diff}

## Instructions
Review the diff above using the criteria in your system prompt. \
Use the codebase context (if available) to check consistency with \
existing patterns and to understand the impact of changes on callers \
or dependents.

Produce a structured review with:
  - A clear summary in `body` covering what the PR does, key findings, \
and your reasoning
  - A precise `event` decision (APPROVE / REQUEST_CHANGES / COMMENT)
  - Inline `comments` only where you can be specific and actionable
"""


# ── Diff formatter ────────────────────────────────────────────────

def format_diff_for_prompt(pr_diff) -> str:
    """
    Formats a PullRequestDiff into a readable string for the prompt.
    Args:
        pr_diff: PullRequestDiff instance from diff_parser.py.

    Returns:
        Formatted string ready to inject into review_human_prompt.
        Empty string if the diff has no files.
    """
    if not pr_diff.files:
        return "(no files changed)"

    parts = []

    for f in pr_diff.files:
        status_line = (
            f"### {f.filename} [{f.status}] "
            f"+{f.additions} -{f.deletions}"
        )
        if f.old_filename:
            status_line += f" (renamed from {f.old_filename})"

        file_parts = [status_line]

        if f.hunks:
            for hunk in f.hunks:
                file_parts.append(hunk.header)
                file_parts.append(hunk.content)
        else:
            file_parts.append("(no diff available for this file)")

        parts.append("\n".join(file_parts))

    return "\n\n".join(parts)


# ── Private helpers ───────────────────────────────────────────────

def _build_context_section(
    retrieved_context: str,
    retrieval_skipped: bool,
) -> str:
    """
    Builds the codebase context section of the human prompt.

    Three cases:
      1. Context available — show it with a clear header
      2. Retrieval skipped — explicit caveat so the LLM knows
         it cannot verify consistency with the existing codebase
      3. Empty context returned — same caveat as skipped
    """
    if retrieved_context and not retrieval_skipped:
        return f"""\
## Codebase Context
The following excerpts from the existing codebase are relevant to \
this PR. Use them to check consistency, understand callers, and \
identify impact of the changes.

{retrieved_context}"""

    # both skipped and empty context get the same caveat —
    # the LLM should not pretend it has context it doesn't have
    return """\
## Codebase Context
Codebase context was unavailable for this review. This may be because \
the repository has not been indexed yet, or the retrieval service was \
temporarily unavailable. Review based on the diff alone and note this \
limitation explicitly in your summary."""