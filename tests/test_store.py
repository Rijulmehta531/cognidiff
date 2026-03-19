import pytest

from app.ingestion.embedder import EmbeddingResult
from app.ingestion.chunker import CodeChunk
from app.ingestion.cloner import compute_content_hash
from app.retrieval.store import CodeStore
from sqlalchemy import text


@pytest.fixture
def store():
    return CodeStore()

@pytest.fixture(autouse=True)
async def clean_db(store):
    async with store.session_factory() as session:
        await session.execute(
            text("TRUNCATE code_chunks, index_runs, repositories CASCADE")
        )
        await session.commit()


@pytest.fixture
def sample_repo_data():
    return {
        "full_name":      "test/sample-repo",
        "clone_url":      "https://github.com/test/sample-repo.git",
        "default_branch": "main",
    }


@pytest.fixture
def sample_chunks():
    """Minimal chunks with fake embeddings — no Ollama needed."""
    return [
        CodeChunk(
            file_path    = "src/auth.py",
            chunk_type   = "function",
            name         = "validate_token",
            parent_class = "",
            language     = "py",
            content      = "def validate_token(token):\n    return True",
            content_hash = compute_content_hash(
                "def validate_token(token):\n    return True"
            ),
            line_start   = 1,
            line_end     = 2,
        ),
        CodeChunk(
            file_path    = "src/payments.py",
            chunk_type   = "function",
            name         = "process_payment",
            parent_class = "",
            language     = "py",
            content      = "def process_payment(amount):\n    pass",
            content_hash = compute_content_hash(
                "def process_payment(amount):\n    pass"
            ),
            line_start   = 1,
            line_end     = 2,
        ),
    ]


@pytest.fixture
def sample_embedding_results(sample_chunks):
    """Fake embedding results — 768 zeros as placeholder vectors."""
    return [
        EmbeddingResult(
            chunk     = chunk,
            embedding = [0.0] * 768,
        )
        for chunk in sample_chunks
    ]


# ── Repository tests ──────────────────────────────────────────────

async def test_get_or_create_repository(store, sample_repo_data):
    repo = await store.get_or_create_repository(**sample_repo_data)
    assert repo.full_name == sample_repo_data["full_name"]
    assert repo.status    == "pending"
    assert repo.id        is not None


async def test_get_or_create_repository_is_idempotent(
    store, sample_repo_data
):
    """Calling twice should return the same repo, not create two."""
    repo1 = await store.get_or_create_repository(**sample_repo_data)
    repo2 = await store.get_or_create_repository(**sample_repo_data)
    assert repo1.id == repo2.id


async def test_update_repository_status(store, sample_repo_data):
    repo = await store.get_or_create_repository(**sample_repo_data)
    await store.update_repository_status(repo.id, "indexing")

    updated = await store.get_repo_by_name(sample_repo_data["full_name"])
    assert updated["status"] == "indexing"


# ── Index run tests ───────────────────────────────────────────────

async def test_create_index_run(store, sample_repo_data):
    repo = await store.get_or_create_repository(**sample_repo_data)
    run  = await store.create_index_run(
        repo_id    = repo.id,
        commit_sha = "abc123def456",
        branch     = "main",
    )
    assert run.id         is not None
    assert run.status     == "running"
    assert run.commit_sha == "abc123def456"


async def test_complete_index_run(store, sample_repo_data):
    repo = await store.get_or_create_repository(**sample_repo_data)
    run  = await store.create_index_run(
        repo_id    = repo.id,
        commit_sha = "abc123",
        branch     = "main",
    )
    await store.complete_index_run(
        run_id          = run.id,
        chunks_created  = 42,
        files_processed = 10,
    )
    # verify by checking repo can be set to active
    await store.set_active_index_run(repo.id, run.id)
    updated = await store.get_repo_by_name(sample_repo_data["full_name"])
    assert str(updated["active_index_run_id"]) == str(run.id)


async def test_fail_index_run(store, sample_repo_data):
    repo = await store.get_or_create_repository(**sample_repo_data)
    run  = await store.create_index_run(
        repo_id    = repo.id,
        commit_sha = "failsha",
        branch     = "main",
    )
    await store.fail_index_run(run.id, "network timeout")


# ── Chunk insert tests ────────────────────────────────────────────

async def test_bulk_insert_chunks(
    store, sample_repo_data, sample_embedding_results
):
    repo = await store.get_or_create_repository(**sample_repo_data)
    run  = await store.create_index_run(
        repo_id    = repo.id,
        commit_sha = "inserttest",
        branch     = "main",
    )
    inserted, skipped = await store.bulk_insert_chunks(
        results      = sample_embedding_results,
        repo_id      = repo.id,
        index_run_id = run.id,
    )
    assert inserted == 2
    assert skipped  == 0


async def test_bulk_insert_idempotent(
    store, sample_repo_data, sample_embedding_results
):
    """Inserting same chunks twice should skip on second insert."""
    repo = await store.get_or_create_repository(**sample_repo_data)
    run  = await store.create_index_run(
        repo_id    = repo.id,
        commit_sha = "idempotenttest",
        branch     = "main",
    )
    await store.bulk_insert_chunks(
        results      = sample_embedding_results,
        repo_id      = repo.id,
        index_run_id = run.id,
    )
    # second insert — should skip all
    inserted, skipped = await store.bulk_insert_chunks(
        results      = sample_embedding_results,
        repo_id      = repo.id,
        index_run_id = run.id,
    )
    assert inserted == 0
    assert skipped  == 2


# ── Similarity search tests ───────────────────────────────────────

async def test_similarity_search_no_active_run(store, sample_repo_data):
    """Repo with no active run returns empty results."""
    repo = await store.get_or_create_repository(**sample_repo_data)
    results = await store.similarity_search(
        query_embedding = [0.0] * 768,
        repo_id         = repo.id,
    )
    assert results == []


async def test_similarity_search_returns_results(
    store, sample_repo_data, sample_embedding_results
):
    repo = await store.get_or_create_repository(**sample_repo_data)
    run  = await store.create_index_run(
        repo_id    = repo.id,
        commit_sha = "searchtest",
        branch     = "main",
    )
    await store.bulk_insert_chunks(
        results      = sample_embedding_results,
        repo_id      = repo.id,
        index_run_id = run.id,
    )
    await store.complete_index_run(run.id, 2, 2)
    await store.set_active_index_run(repo.id, run.id)

    results = await store.similarity_search(
        query_embedding = [0.0] * 768,
        repo_id         = repo.id,
        min_score       = 0.0,   # accept all results for testing
    )
    assert len(results) > 0
    assert "file_path"  in results[0]
    assert "content"    in results[0]
    assert "similarity" in results[0]