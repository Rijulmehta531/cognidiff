import logging
import time
import uuid

from arq import Retry
from arq.connections import RedisSettings

from app.agent.graph import review_graph
from app.config import get_settings
from app.exceptions import TransientError, PermanentError
from app.ingestion.pipeline import run_ingestion_pipeline
from app.retrieval.store import CodeStore

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
        raise Retry(defer=30)

async def review_pr(
    ctx:        dict,
    full_name:  str,
    pr_number:  int,
    pr_title:   str,
    commit_sha: str,
) -> None:
    prefix = f"[worker:review_pr:{full_name}#{pr_number}]"
    logger.info(f"{prefix} started — commit {commit_sha[:8]}")

    store = CodeStore()

    repo = await store.get_repo_by_name(full_name)
    if not repo:
        raise PermanentError(
            f"repo {full_name!r} not found in database — "
            f"has it been indexed yet?"
        )

    repo_id = repo["id"]
    if isinstance(repo_id, str):
        repo_id = uuid.UUID(repo_id)
    
    existing = await store.get_review_by_commit(
        repo_id    = repo_id,
        pr_number  = pr_number,
        commit_sha = commit_sha,
    )

    if existing:
        logger.info(f"{prefix} already reviewed at commit {commit_sha[:8]} — skipping")
        return

    initial_state = {
        "full_name":         full_name,
        "pr_number":         pr_number,
        "pr_title":          pr_title,
        "commit_sha":        commit_sha,
        "repo_id":           repo_id,
        "pr_diff":           None,
        "retrieved_context": None,
        "retrieval_skipped": False,
        "review":            None,
        "error":             None,
    }

    started_at = time.monotonic()

    # TransientError raised inside the graph
    # propagates naturally — ARQ catches it and schedules a retry.
    final_state = await review_graph.ainvoke(initial_state)
 
    processing_ms = int((time.monotonic() - started_at) * 1000)
 
    if final_state.get("error"):
        logger.error(
            f"{prefix} graph ended with error — "
            f"{final_state['error']}"
        )
        return

    review = final_state["review"]

    await store.save_review(
        repo_id        = repo_id,
        pr_number      = pr_number,
        pr_title       = pr_title,
        commit_sha     = commit_sha,
        decision       = review.event,
        summary        = review.body,
        comments_count = len(review.comments),
        processing_ms  = processing_ms,
    )

    logger.info(
        f"{prefix} complete — "
        f"{review.event}, "
        f"{len(review.comments)} inline comments, "
        f"{processing_ms}ms"
    )



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
    functions = [ingest_repo, review_pr]

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