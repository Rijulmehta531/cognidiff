import logging
from contextlib import asynccontextmanager

from arq.connections import create_pool, RedisSettings
from fastapi import FastAPI

from app.config import get_settings
from app.github.webhook import router as webhook_router

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────
    settings = get_settings()

    logger.info("[main] starting up — connecting to Redis")
    app.state.arq_redis = await create_pool(
        RedisSettings.from_dsn(settings.REDIS_URL)
    )
    logger.info("[main] Redis connection established")

    yield
    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("[main] shutting down — closing Redis connection")
    await app.state.arq_redis.close()
    logger.info("[main] shutdown complete")


# ── App ───────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    settings = get_settings()

    app = FastAPI(
        title       = "CogniDiff",
        description = "AI-powered GitHub PR reviewer",
        lifespan    = lifespan,
    )

    # ── Routes ────────────────────────────────────────────────────
    app.include_router(webhook_router)

    return app


app = create_app()