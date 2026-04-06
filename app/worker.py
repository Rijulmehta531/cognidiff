import logging

from arq import cron
from arq.connections import RedisSettings
from arq.jobs import Retry

from app.config import get_settings
from app.exceptions import TransientError, PermanentError
from app.ingestion.pipeline import run_ingestion_pipeline

logger = logging.getLogger(__name__)


# ── Job functions ─────────────────────────────────────────────────

async def ingest_repo(
    ctx:       dict,
    full_name: str,
    ref:       str,
) -> None:
    """
    ARQ job — runs the full ingestion pipeline for a repository.
    """
    logger.info(
        f"[worker] ingest_repo started — "
        f"{full_name}@{ref} "
        f"(attempt {ctx['job_try']})"
    )

    try:
        settings = get_settings()
        token = settings.GITHUB_TOKEN
        await run_ingestion_pipeline(
            full_name = full_name,
            ref       = ref,
            token     = token,
        )
        logger.info(f"[worker] ingest_repo complete — {full_name}@{ref}")

    except PermanentError as e:
        logger.error(
            f"[worker] ingest_repo permanent failure — "
            f"{full_name}@{ref}: {e}"
        )

    except TransientError as e:
        logger.warning(
            f"[worker] ingest_repo transient failure — "
            f"{full_name}@{ref}: {e} "
            f"(attempt {ctx['job_try']} of {ctx['job_try_count']})"
        )
        raise Retry(defer=30)  # retry after 30 seconds


# Worker settings
# entry point for ARQ to start the worker.
# command: arq app.worker.WorkerSettings

class WorkerSettings:
    """
    ARQ worker configuration.
    """
    settings = get_settings()

    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)

    # job functions this worker handles
    functions = [ingest_repo]

    # max parallel jobs — balances throughput against
    # GitHub API rate limits and embedding service capacity
    max_jobs = 5

    # per-job timeout in seconds — 10 minutes covers
    # large repos with many new chunks to embed
    job_timeout = 600

    # max attempts before giving up on a transient failure
    max_tries = 3

    # how long to keep job results in Redis (seconds)
    keep_result = 3600