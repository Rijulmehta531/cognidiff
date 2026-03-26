import logging
from app.config import get_settings
from app.ingestion.chunker import ASTChunker
from app.ingestion.cloner import download_repo, get_commit_sha
from app.retrieval.store import CodeStore
from app.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)
async def run_ingestion_pipeline(
    full_name: str,
    ref:       str,
    token:     str,
) -> None:
    """ Run the full ingestion pipeline for a given repo and ref.
    clone -> chunk -> embed -> store
    """

    settings = get_settings()
    store = CodeStore()
    chunker = ASTChunker()
    embedder = Embedder()

    prefix = f"[pipeline:{full_name}]"

    repo = None
    run  = None

    try:
        # ── Step 1: Repository record ─────────────────────────────
        logger.info(f"{prefix} starting — ref: {ref}")

        repo = await store.get_or_create_repository(
            full_name      = full_name,
            clone_url      = f"https://github.com/{full_name}.git",
            default_branch = ref,
        )
        await store.update_repository_status(repo.id, "indexing")
        logger.info(f"{prefix} repository record ready — id: {repo.id}")

        # ── Step 2: Resolve commit SHA ────────────────────────────
        logger.info(f"{prefix} resolving commit sha — ref: {ref}")
        commit_sha = await get_commit_sha(
            full_name = full_name,
            ref       = ref,
            token     = token,
        )
        logger.info(f"{prefix} commit sha resolved — {commit_sha[:8]}")

        # ── Step 3: Create index run ──────────────────────────────
        run = await store.create_index_run(
            repo_id    = repo.id,
            commit_sha = commit_sha,
            branch     = ref,
        )
        logger.info(f"{prefix} index run created — id: {run.id}")
 
        # ── Step 4 & 5: Download and chunk ────────────────────────
        logger.info(f"{prefix} cloning — sha: {commit_sha[:8]}")
        async with download_repo(full_name, ref, token) as repo_path:
            logger.info(f"{prefix} cloning complete — chunking started")
            chunks = chunker.chunk_repo(repo_path)
 
        logger.info(
            f"{prefix} chunking complete — "
            f"{len(chunks)} chunks found across repo"
        )
 
        # ── Step 6: Fetch existing chunks from previous run ───────
        logger.info(f"{prefix} fetching existing chunks from previous run")
        existing_chunks = await store.get_existing_chunks(repo_id=repo.id)
        logger.info(
            f"{prefix} existing chunks fetched — "
            f"{len(existing_chunks)} chunks in previous run"
        )
 
        # ── Step 7: Split into new vs unchanged ───────────────────
        chunks_to_embed = []
        chunks_to_copy  = []
 
        for chunk in chunks:
            if chunk.content_hash in existing_chunks:
                chunks_to_copy.append(existing_chunks[chunk.content_hash])
            else:
                chunks_to_embed.append(chunk)
 
        logger.info(
            f"{prefix} split complete — "
            f"{len(chunks_to_embed)} new, "
            f"{len(chunks_to_copy)} unchanged"
        )
 
        # ── Step 8: Embed new chunks ──────────────────────────────
        logger.info(f"{prefix} embedding — {len(chunks_to_embed)} new chunks")
        embedding_results = await embedder.embed_chunks(chunks_to_embed)
 
        successful = sum(1 for r in embedding_results if r.success)
        failed     = sum(1 for r in embedding_results if not r.success)
        logger.info(
            f"{prefix} embedding complete — "
            f"{successful} succeeded, {failed} failed"
        )
 
        # ── Step 9: Copy unchanged chunks forward ─────────────────
        logger.info(
            f"{prefix} copying {len(chunks_to_copy)} "
            f"unchanged chunks forward"
        )
        copied = await store.copy_chunks_forward(
            chunk_rows       = chunks_to_copy,
            repo_id          = repo.id,
            new_index_run_id = run.id,
        )
        logger.info(f"{prefix} copy complete — {copied} chunks copied")
 
        # ── Step 10: Insert newly embedded chunks ─────────────────
        logger.info(f"{prefix} inserting {successful} newly embedded chunks")
        inserted, skipped = await store.bulk_insert_chunks(
            results      = embedding_results,
            repo_id      = repo.id,
            index_run_id = run.id,
        )
        logger.info(
            f"{prefix} insert complete — "
            f"{inserted} inserted, {skipped} skipped"
        )
 
        # ── Step 11: Finalise run ─────────────────────────────────
        total_chunks = copied + inserted
        logger.info(f"{prefix} finalising run — {total_chunks} total chunks")
 
        await store.complete_index_run(
            run_id          = run.id,
            chunks_created  = total_chunks,
            files_processed = len({c.file_path for c in chunks}),
        )
        await store.supersede_previous_runs(repo.id, run.id)
        await store.set_active_index_run(repo.id, run.id)
 
        logger.info(
            f"{prefix} pipeline complete — "
            f"run {run.id} is now active"
        )
 
    except Exception as e:
        logger.error(
            f"{prefix} pipeline failed — {e}",
            exc_info=True, # log stack trace for debugging
        )
        if run:
            await store.fail_index_run(run.id, str(e))
        if repo:
            await store.update_repository_status(repo.id, "failed")
        raise
