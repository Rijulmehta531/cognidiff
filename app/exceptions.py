class CogniDiffError(Exception):
    """
    Root base class for all CogniDiff application exceptions.
    Catch this to handle any application-level error in one place.
    """


# ── Pipeline exceptions ────────────────────────────────────────

class PipelineError(CogniDiffError):
    """Base class for all ingestion pipeline errors."""


class TransientError(PipelineError):
    """
    A failure that is worth retrying — network issues,
    rate limits, temporary service unavailability.
    """


class PermanentError(PipelineError):
    """
    A failure that is not worth retrying — bad credentials,
    resource not found, malformed data.
    """


# ── GitHub-specific exceptions ────────────────────────────────────

class RepoNotFoundError(PermanentError):
    """
    Raised when GitHub returns 404 for a repo.
    Either the repo doesn't exist or the token
    doesn't have access to it.
    """


class AuthenticationError(PermanentError):
    """
    Raised when GitHub returns 401.
    The token is invalid or has expired.
    """


class GitHubRateLimitError(TransientError):
    """
    Raised when GitHub returns 429.
    The app has exceeded its API rate limit — retry after backoff.
    """


class GitHubUnavailableError(TransientError):
    """
    Raised when GitHub returns 503 or 504.
    GitHub is temporarily unavailable — retry after backoff.
    """


# ── Embedding exceptions ──────────────────────────────────────────

class EmbeddingUnavailableError(TransientError):
    """
    Raised when the embedding service is unreachable
    or returns an error. Covers both Ollama and Azure.
    """