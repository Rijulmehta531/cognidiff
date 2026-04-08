import hashlib
import hmac
import logging
from typing import Annotated

from arq.connections import ArqRedis
from fastapi import APIRouter, Header, HTTPException, Request, Depends

from app.config import get_settings
from app.exceptions import PermanentError

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Signature validation ──────────────────────────────────────────

def _verify_signature(payload: bytes, signature_header: str) -> None:
    """
    Validates the GitHub webhook signature.
    Raises PermanentError if the signature is invalid or missing.
    """
    settings = get_settings()

    if not settings.GITHUB_WEBHOOK_SECRET:
        raise PermanentError(
            "GITHUB_WEBHOOK_SECRET is not configured"
        )

    if not signature_header or not signature_header.startswith("sha256="):
        raise PermanentError(
            "missing or malformed X-Hub-Signature-256 header"
        )

    expected = hmac.new(
        settings.GITHUB_WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(f"sha256={expected}", signature_header):
        raise PermanentError("webhook signature validation failed")


# ── Dependency ────────────────────────────────────────────────────

async def get_arq_redis(request: Request) -> ArqRedis:
    """
    Pulls the shared ARQ Redis connection from app state.
    Created once at startup in main.py and shared across requests —
    not created per request.
    """
    return request.app.state.arq_redis

ArqRedisDep = Annotated[ArqRedis, Depends(get_arq_redis)]

# ── Route ─────────────────────────────────────────────────────────

@router.post(
    "/webhooks/github",
    responses={
        401: {"description": "Invalid or missing webhook signature"},
        400: {"description": "Missing required GitHub event header"},
    },
)
async def github_webhook(
    request:              Request,
    arq_redis:            ArqRedisDep,
    x_hub_signature_256:  Annotated[str | None, Header()] = None,
    x_github_event:       Annotated[str | None, Header()] = None,
) -> dict:
    """
    Receives and processes GitHub webhook events.

    Handles:
    - push   → ingest_repo if push is to default branch
    - pull_request (opened, synchronize) → review_pr (stub)

    Responds with 200 immediately — actual work is done
    asynchronously by the ARQ worker.
    """
    payload_bytes = await request.body()

    try:
        _verify_signature(payload_bytes, x_hub_signature_256)
    except PermanentError as e:
        logger.warning(f"[webhook] signature validation failed — {e}")
        raise HTTPException(status_code=401, detail=str(e))

    if not x_github_event:
        raise HTTPException(status_code=400, detail="missing X-GitHub-Event header")

    payload = await request.json()

    logger.info(f"[webhook] received event: {x_github_event}")

    # ── push event ────────────────────────────────────────────────
    if x_github_event == "push":
        await _handle_push(payload, arq_redis)

    # ── pull_request event ────────────────────────────────────────
    elif x_github_event == "pull_request":
        await _handle_pull_request(payload, arq_redis)

    else:
        logger.info(f"[webhook] ignoring unhandled event: {x_github_event}")

    # always respond 200 to avoid unnecessary retries from GitHub — actual work is done asynchronously
    return {"status": "accepted"}


# ── Event handlers ────────────────────────────────────────────────

async def _handle_push(payload: dict, arq_redis: ArqRedis) -> None:
    """
    Handles push events.
    Only triggers ingestion for pushes to the default branch —
    the vector DB snapshot should reflect the stable, merged
    codebase, not feature branches.
    """
    full_name      = payload["repository"]["full_name"]
    default_branch = payload["repository"]["default_branch"]
    ref            = payload["ref"]  # e.g. "refs/heads/main"

    if ref != f"refs/heads/{default_branch}":
        logger.info(
            f"[webhook] ignoring push to non-default branch "
            f"{ref} for {full_name}"
        )
        return

    logger.info(
        f"[webhook] enqueueing ingest_repo — "
        f"{full_name}@{default_branch}"
    )

    await arq_redis.enqueue_job(
        "ingest_repo",
        full_name,
        default_branch,
        # deterministic job id — if the same repo receives another
        # push before this job runs, the new job replaces the old one
        _job_id=f"ingest:{full_name}:{default_branch}",
    )

    logger.info(
        f"[webhook] ingest_repo enqueued — "
        f"{full_name}@{default_branch}"
    )


async def _handle_pull_request(payload: dict, arq_redis: ArqRedis) -> None:
    """
    Handles pull_request events.
    Triggers a review for newly opened PRs and PRs updated
    with a new commit (synchronize action).

    review_pr job is not implemented yet — this is a stub
    that will be wired up when the agent layer is built.
    """
    action     = payload["action"]
    full_name  = payload["repository"]["full_name"]
    pr_number  = payload["pull_request"]["number"]
    pr_title   = payload["pull_request"]["title"]
    commit_sha = payload["pull_request"]["head"]["sha"]

    if action not in ("opened", "synchronize"):
        logger.info(
            f"[webhook] ignoring pull_request action: {action}"
        )
        return

    logger.info(
        f"[webhook] pull_request {action} — "
        f"{full_name} #{pr_number} ({commit_sha[:8]})"
    )

    await arq_redis.enqueue_job(
        "review_pr",
        full_name,
        pr_number,
        pr_title,
        commit_sha,
        _job_id=f"review:{full_name}:{pr_number}",
    )

    logger.info(
        f"[webhook] review_pr not yet implemented — "
        f"skipping {full_name} #{pr_number}"
    )