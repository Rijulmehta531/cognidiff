import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, text, literal, select, cast
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from sqlalchemy.dialects.postgresql import insert

from app.config import get_settings
from app.ingestion.chunker import CodeChunk
from app.ingestion.embedder import EmbeddingResult

logger = logging.getLogger(__name__)


# ── Database engine ───────────────────────────────────────────────

def get_engine():
    settings = get_settings()
    return create_async_engine(
        settings.DATABASE_URL,
        echo=False,      # set True to log all SQL — useful for debugging
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,  # verify connection is alive before using
    )

def get_session_factory(engine=None) -> async_sessionmaker:
    if engine is None:
        engine = get_engine()
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        # expire_on_commit=False means ORM objects stay usable
        # after commit — important for async where we read
        # attributes after the session closes
    )


# ── ORM models ────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class RepositoryModel(Base):
    __tablename__ = "repositories"

    id:                  Mapped[uuid.UUID]          = mapped_column(primary_key=True, default=uuid.uuid4)
    full_name:           Mapped[str]
    clone_url:           Mapped[str]
    default_branch:      Mapped[str]                = mapped_column(default="main")
    last_indexed_at:     Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    status:              Mapped[str]                = mapped_column(default="pending")
    active_index_run_id: Mapped[Optional[uuid.UUID]]
    created_at:          Mapped[datetime]           = mapped_column(DateTime(timezone=True),default=lambda: datetime.now(timezone.utc))
    updated_at:          Mapped[datetime]           = mapped_column(DateTime(timezone=True),default=lambda: datetime.now(timezone.utc))


class IndexRunModel(Base):
    __tablename__ = "index_runs"

    id:               Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    repo_id:          Mapped[uuid.UUID]
    commit_sha:       Mapped[str]
    branch:           Mapped[str]       = mapped_column(default="main")
    status:           Mapped[str]       = mapped_column(default="running")
    embedding_model:  Mapped[str]
    chunking_version: Mapped[str]       = mapped_column(default="1.0")
    parser_version:   Mapped[str]
    files_processed:  Mapped[int]       = mapped_column(default=0)
    chunks_created:   Mapped[int]       = mapped_column(default=0)
    chunks_deleted:   Mapped[int]       = mapped_column(default=0)
    error_message:    Mapped[Optional[str]]
    started_at:       Mapped[datetime]  = mapped_column(DateTime(timezone=True),default=lambda: datetime.now(timezone.utc))
    completed_at:     Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, default=None)


class CodeChunkModel(Base):
    __tablename__ = "code_chunks"

    id:           Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    repo_id:      Mapped[uuid.UUID]
    index_run_id: Mapped[uuid.UUID]
    file_path:    Mapped[str]
    chunk_type:   Mapped[str]
    name:         Mapped[str]
    parent_class: Mapped[str]       = mapped_column(default="")
    language:     Mapped[str]
    content:      Mapped[str]
    content_hash: Mapped[str]
    line_start:   Mapped[Optional[int]]
    line_end:     Mapped[Optional[int]]
    embedding:    Mapped[Optional[list[float]]] = mapped_column(Vector(768))
    created_at:   Mapped[datetime]  = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ── Store ─────────────────────────────────────────────────────────

class CodeStore:

    def __init__(self):
        self.settings        = get_settings()
        self.engine          = get_engine()
        self.session_factory = get_session_factory(self.engine)

    # ── Repository operations ─────────────────────────────────────

    async def get_or_create_repository(
        self,
        full_name:      str,
        clone_url:      str,
        default_branch: str = "main",
    ) -> RepositoryModel:
        """
        Returns existing repo row or creates a new one.
        Idempotent — safe to call multiple times.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                text("SELECT * FROM repositories WHERE full_name = :name"),
                {"name": full_name},
            )
            row = result.mappings().first()

            if row:
                return RepositoryModel(**row)

            repo = RepositoryModel(
                full_name      = full_name,
                clone_url      = clone_url,
                default_branch = default_branch,
                status         = "pending",
            )
            session.add(repo)
            await session.commit()
            logger.info(f"created repository record: {full_name}")
            return repo

    async def update_repository_status(
        self,
        repo_id: uuid.UUID,
        status:  str,
    ) -> None:
        async with self.session_factory() as session:
            await session.execute(
                text("""
                    UPDATE repositories
                    SET status = :status, updated_at = NOW()
                    WHERE id = :id
                """),
                {"status": status, "id": repo_id},
            )
            await session.commit()

    async def set_active_index_run(
        self,
        repo_id:      uuid.UUID,
        index_run_id: uuid.UUID,
    ) -> None:
        """
        Points the repository at the newly completed index run.
        All RAG queries scope to this run from this point on.
        """
        async with self.session_factory() as session:
            await session.execute(
                text("""
                    UPDATE repositories
                    SET active_index_run_id = :run_id,
                        last_indexed_at     = NOW(),
                        status              = 'indexed',
                        updated_at          = NOW()
                    WHERE id = :repo_id
                """),
                {"run_id": index_run_id, "repo_id": repo_id},
            )
            await session.commit()
            logger.info(
                f"repo {repo_id} now pointing at "
                f"index run {index_run_id}"
            )

    # ── Index run operations ──────────────────────────────────────

    async def create_index_run(
        self,
        repo_id:    uuid.UUID,
        commit_sha: str,
        branch:     str,
    ) -> IndexRunModel:
        async with self.session_factory() as session:
            run = IndexRunModel(
                repo_id          = repo_id,
                commit_sha       = commit_sha,
                branch           = branch,
                status           = "running",
                embedding_model  = self.settings.active_embedding_model,
                chunking_version = self.settings.CHUNKING_VERSION,
                parser_version   = self.settings.parser_version,
            )
            session.add(run)
            await session.commit()
            logger.info(
                f"created index run {run.id} "
                f"for repo {repo_id} at {commit_sha[:8]}"
            )
            return run

    async def complete_index_run(
        self,
        run_id:          uuid.UUID,
        chunks_created:  int,
        files_processed: int,
    ) -> None:
        async with self.session_factory() as session:
            await session.execute(
                text("""
                    UPDATE index_runs
                    SET status          = 'completed',
                        completed_at    = NOW(),
                        chunks_created  = :chunks_created,
                        files_processed = :files_processed
                    WHERE id = :id
                """),
                {
                    "id":              run_id,
                    "chunks_created":  chunks_created,
                    "files_processed": files_processed,
                },
            )
            await session.commit()

    async def fail_index_run(
        self,
        run_id:  uuid.UUID,
        error:   str,
    ) -> None:
        """
        Marks a run as failed. The previous completed run
        remains active — no degradation to RAG quality.
        """
        async with self.session_factory() as session:
            await session.execute(
                text("""
                    UPDATE index_runs
                    SET status        = 'failed',
                        completed_at  = NOW(),
                        error_message = :error
                    WHERE id = :id
                """),
                {"id": run_id, "error": error},
            )
            await session.commit()
            logger.error(f"index run {run_id} failed: {error}")

    async def supersede_previous_runs(
        self,
        repo_id:         uuid.UUID,
        current_run_id:  uuid.UUID,
    ) -> None:
        """
        Marks all previously completed runs for this repo
        as superseded. Called after a new run completes.
        """
        async with self.session_factory() as session:
            await session.execute(
                text("""
                    UPDATE index_runs
                    SET status = 'superseded'
                    WHERE repo_id = :repo_id
                      AND id      != :current_id
                      AND status  = 'completed'
                """),
                {"repo_id": repo_id, "current_id": current_run_id},
            )
            await session.commit()

    # ── Chunk operations ──────────────────────────────────────────

    async def bulk_insert_chunks(
        self,
        results:      list[EmbeddingResult],
        repo_id:      uuid.UUID,
        index_run_id: uuid.UUID,
    ) -> tuple[int, int]:
        """
        Bulk inserts all successfully embedded chunks.
        Skips duplicates via ON CONFLICT DO NOTHING —
        safe for retries within the same run.

        Returns (inserted_count, skipped_count).
        """
        successful = [r for r in results if r.success]
        failed     = [r for r in results if not r.success]

        if failed:
            logger.warning(
                f"{len(failed)} chunks had no embedding "
                f"and will be stored without vector"
            )

        if not successful:
            return 0, len(results)

        rows = [
            {
                "id":           str(uuid.uuid4()),
                "repo_id":      str(repo_id),
                "index_run_id": str(index_run_id),
                "file_path":    r.chunk.file_path,
                "chunk_type":   r.chunk.chunk_type,
                "name":         r.chunk.name,
                "parent_class": r.chunk.parent_class,
                "language":     r.chunk.language,
                "content":      r.chunk.content,
                "content_hash": r.chunk.content_hash,
                "line_start":   r.chunk.line_start,
                "line_end":     r.chunk.line_end,
                "embedding":    r.embedding,
            }
            for r in successful
        ]

        # insert in batches of 500 to avoid hitting
        # PostgreSQL's parameter limit per statement
        inserted = 0
        for i in range(0, len(rows), 500):
            batch = rows[i : i + 500]
            result = await self._insert_chunk_batch(batch)
            inserted += result

        skipped = len(successful) - inserted
        logger.info(
            f"chunks: {inserted} inserted, "
            f"{skipped} skipped (duplicates), "
            f"{len(failed)} had no embedding"
        )
        return inserted, skipped

    async def _insert_chunk_batch(self, rows: list[dict]) -> int:
        """
        True bulk insert using pgvector's SQLAlchemy integration.
        One round trip per batch regardless of row count.
        """
        if not rows:
            return 0

        async with self.session_factory() as session:
            typed_rows = []
            for row in rows:
                typed_rows.append({
                    "id":           uuid.UUID(row["id"]),
                    "repo_id":      uuid.UUID(row["repo_id"]),
                    "index_run_id": uuid.UUID(row["index_run_id"]),
                    "file_path":    row["file_path"],
                    "chunk_type":   row["chunk_type"],
                    "name":         row["name"],
                    "parent_class": row["parent_class"],
                    "language":     row["language"],
                    "content":      row["content"],
                    "content_hash": row["content_hash"],
                    "line_start":   row["line_start"],
                    "line_end":     row["line_end"],
                    "embedding":    row["embedding"],
                })

            stmt = (
                insert(CodeChunkModel)
                .values(typed_rows)
                .on_conflict_do_nothing(
                    index_elements=["index_run_id", "file_path", "content_hash"]
                )
            )

            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount

    # ── Retrieval operations ──────────────────────────────────────

    async def similarity_search(
        self,
        query_embedding: list[float],
        repo_id:         uuid.UUID,
        top_k:           int   = 10,
        min_score:       float = 0.70,
        language:        Optional[str] = None,
        chunk_type:      Optional[str] = None,
    ) -> list[dict]:

        async with self.session_factory() as session:
            # get active run
            run_result = await session.execute(
                text("""
                    SELECT active_index_run_id
                    FROM repositories
                    WHERE id = CAST(:repo_id AS uuid)
                """),
                {"repo_id": str(repo_id)},
            )
            row = run_result.mappings().first()

            if not row or not row["active_index_run_id"]:
                logger.warning(f"repo {repo_id} has no active index run")
                return []

            active_run_id   = row["active_index_run_id"]
            query_vector    = cast(query_embedding, Vector(768))

            # cosine similarity expressed via SQLAlchemy operators
            similarity_expr = (
                literal(1.0) - CodeChunkModel.embedding.op("<=>")(query_vector)
            ).label("similarity")

            stmt = (
                select(
                    CodeChunkModel.id,
                    CodeChunkModel.file_path,
                    CodeChunkModel.chunk_type,
                    CodeChunkModel.name,
                    CodeChunkModel.parent_class,
                    CodeChunkModel.language,
                    CodeChunkModel.content,
                    CodeChunkModel.line_start,
                    CodeChunkModel.line_end,
                    similarity_expr,
                )
                .where(CodeChunkModel.index_run_id == active_run_id)
                .where(similarity_expr >= min_score)
                .order_by(
                    CodeChunkModel.embedding.op("<=>")(query_vector)
                )
                .limit(top_k)
            )

            if language:
                stmt = stmt.where(CodeChunkModel.language == language)
            if chunk_type:
                stmt = stmt.where(CodeChunkModel.chunk_type == chunk_type)

            result  = await session.execute(stmt)
            rows    = result.mappings().all()
            return [dict(r) for r in rows]

    async def get_existing_chunks(
        self,
        repo_id: uuid.UUID,
    ) -> dict[str, dict]:
        """
        Returns all chunks from the current active index run as a dict (key is content_hash).

        Used to determine which chunks are unchanged and can be copied forward instead of re-embedded.

        Returns empty dict if no active run exists (first-time index).
        """
        async with self.session_factory() as session:

            # get active run id
            result = await session.execute(
                text("""
                    SELECT active_index_run_id
                    FROM repositories
                    WHERE id = CAST(:repo_id AS uuid)
                """),
                {"repo_id": str(repo_id)},
            )
            row = result.mappings().first()

            if not row or not row["active_index_run_id"]:
                logger.info(
                    f"repo {repo_id} has no active index run — "
                    f"first time indexing, all chunks are new"
                )
                return {}

            active_run_id = row["active_index_run_id"]

            # fetch all chunks for the active run
            result = await session.execute(
                text("""
                    SELECT
                        id,
                        file_path,
                        chunk_type,
                        name,
                        parent_class,
                        language,
                        content,
                        content_hash,
                        line_start,
                        line_end,
                        embedding
                    FROM code_chunks
                    WHERE index_run_id = CAST(:run_id AS uuid)
                """),
                {"run_id": str(active_run_id)},
            )
            rows = result.mappings().all()

            chunks_by_hash = {}
            for row in rows:
                row_dict = dict(row)

                embedding = row_dict.get("embedding")
                if isinstance(embedding, str):
                    row_dict["embedding"] = [
                        float(x) for x in embedding.strip("[]").split(",") if x.strip()
                    ]
                elif embedding is not None:
                    row_dict["embedding"] = [float(x) for x in embedding]

                chunks_by_hash[row_dict["content_hash"]] = row_dict

            logger.info(
                f"found {len(chunks_by_hash)} existing chunks "
                f"in active run {active_run_id} for repo {repo_id}"
            )
            return chunks_by_hash

    async def copy_chunks_forward(
        self,
        chunk_rows:      list[dict],
        repo_id:         uuid.UUID,
        new_index_run_id: uuid.UUID,
    ) -> int:
        """
        Copies existing chunk rows into a new index run. Assigns fresh UUIDs and the new index_run_id —
        everything else is preserved.

        Returns count of rows copied.
        """
        if not chunk_rows:
            return 0

        typed_rows = [
            {
                "id":           uuid.uuid4(),
                "repo_id":      repo_id,
                "index_run_id": new_index_run_id,
                "file_path":    row["file_path"],
                "chunk_type":   row["chunk_type"],
                "name":         row["name"],
                "parent_class": row["parent_class"],
                "language":     row["language"],
                "content":      row["content"],
                "content_hash": row["content_hash"],
                "line_start":   row["line_start"],
                "line_end":     row["line_end"],
                "embedding":    row["embedding"],
            }
            for row in chunk_rows
        ]

        # insert in batches of 500 — same pattern as bulk_insert_chunks
        copied = 0
        for i in range(0, len(typed_rows), 500):
            batch = typed_rows[i : i + 500]
            async with self.session_factory() as session:
                stmt = (
                    insert(CodeChunkModel)
                    .values(batch)
                    .on_conflict_do_nothing(
                        index_elements=["index_run_id", "file_path", "content_hash"]
                    )
                )
                result = await session.execute(stmt)
                await session.commit()
                copied += result.rowcount

        logger.info(
            f"copied {copied} unchanged chunks forward "
            f"into index run {new_index_run_id}"
        )
        return copied

    async def get_repo_by_name(
        self,
        full_name: str,
    ) -> Optional[dict]:
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT * FROM repositories
                    WHERE full_name = :name
                """),
                {"name": full_name},
            )
            row = result.mappings().first()
            return dict(row) if row else None

    async def get_review_by_commit(
        self,
        repo_id:    uuid.UUID,
        pr_number:  int,
        commit_sha: str,
    ) -> Optional[dict]:
        """
        Returns the existing review row for a specific commit SHA,
        or None if this commit has not been reviewed yet.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT * FROM pr_reviews
                    WHERE repo_id    = CAST(:repo_id AS uuid)
                      AND pr_number  = :pr_number
                      AND commit_sha = :commit_sha
                    LIMIT 1
                """),
                {
                    "repo_id":    str(repo_id),
                    "pr_number":  pr_number,
                    "commit_sha": commit_sha,
                },
            )
            row = result.mappings().first()
            return dict(row) if row else None

    async def save_review(
        self,
        repo_id:       uuid.UUID,
        pr_number:     int,
        pr_title:      str,
        commit_sha:    str,
        decision:      str,
        summary:       str,
        comments_count: int,
        processing_ms: Optional[int] = None,
    ) -> None:
        """
        Appends a new row to pr_reviews.

        append-only — never updates existing rows. Each commit SHA
        gets its own row. review_run is computed as the count of
        previous reviews for this PR plus one.

        issues_found is set to comments_count when the decision is
        REQUEST_CHANGES — these are the actionable problems surfaced.
        issues_open mirrors issues_found on insert; it decreases as
        the author addresses comments in subsequent review runs.
        """
        async with self.session_factory() as session:
            # compute review_run as count of existing reviews for this PR plus one
            count_result = await session.execute(
                text("""
                    SELECT COUNT(*) AS cnt FROM pr_reviews
                    WHERE repo_id   = CAST(:repo_id AS uuid)
                      AND pr_number = :pr_number
                """),
                {"repo_id": str(repo_id), "pr_number": pr_number},
            )
            review_run = count_result.scalar() + 1

            issues_found = comments_count if decision == "REQUEST_CHANGES" else 0

            await session.execute(
                text("""
                    INSERT INTO pr_reviews (
                        repo_id,
                        pr_number,
                        pr_title,
                        commit_sha,
                        review_run,
                        review_decision,
                        review_summary,
                        comments_count,
                        issues_found,
                        issues_open,
                        processing_ms
                    ) VALUES (
                        CAST(:repo_id AS uuid),
                        :pr_number,
                        :pr_title,
                        :commit_sha,
                        :review_run,
                        :review_decision,
                        :review_summary,
                        :comments_count,
                        :issues_found,
                        :issues_open,
                        :processing_ms
                    )
                """),
                {
                    "repo_id":        str(repo_id),
                    "pr_number":      pr_number,
                    "pr_title":       pr_title,
                    "commit_sha":     commit_sha,
                    "review_run":     review_run,
                    "review_decision": decision,
                    "review_summary": summary,
                    "comments_count": comments_count,
                    "issues_found":   issues_found,
                    "issues_open":    issues_found,
                    "processing_ms":  processing_ms,
                },
            )
            await session.commit()
            logger.info(
                f"saved review run {review_run} for "
                f"{repo_id} PR#{pr_number} — {decision}"
            )