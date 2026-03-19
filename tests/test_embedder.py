import pytest

from app.ingestion.chunker import ASTChunker, CodeChunk
from app.ingestion.embedder import Embedder, EmbeddingResult


@pytest.fixture
def embedder():
    return Embedder()


@pytest.fixture
def sample_chunks(tmp_path):
    """Creates real chunks from a small Python snippet."""
    (tmp_path / "sample.py").write_text(
        "def hello(name):\n"
        "    return f'Hello {name}'\n"
        "\n"
        "def goodbye(name):\n"
        "    return f'Goodbye {name}'\n"
    )
    chunker = ASTChunker()
    return chunker.chunk_repo(tmp_path)


async def test_embed_chunks_returns_results(embedder, sample_chunks):
    """Basic: embedding returns one result per chunk."""
    results = await embedder.embed_chunks(sample_chunks)
    assert len(results) == len(sample_chunks)


async def test_embed_chunks_all_succeed(embedder, sample_chunks):
    """With Ollama running, all chunks should embed successfully."""
    results = await embedder.embed_chunks(sample_chunks)
    failed  = [r for r in results if not r.success]
    assert len(failed) == 0, (
        f"{len(failed)} chunks failed: "
        f"{[r.error for r in failed]}"
    )


async def test_embedding_dimension_correct(embedder, sample_chunks):
    """Embeddings must match the dimension pgvector expects."""
    results = await embedder.embed_chunks(sample_chunks)
    for result in results:
        if result.success:
            valid = embedder.validate_embedding_dimension(
                result.embedding
            )
            assert valid, (
                f"wrong dimension for {result.chunk.name}: "
                f"got {len(result.embedding)}"
            )


async def test_embed_empty_list(embedder):
    """Empty input should return empty output without errors."""
    results = await embedder.embed_chunks([])
    assert results == []


async def test_embedding_result_structure(embedder, sample_chunks):
    """EmbeddingResult fields are correctly populated."""
    results = await embedder.embed_chunks(sample_chunks)
    for result in results:
        assert isinstance(result, EmbeddingResult)
        assert isinstance(result.chunk, CodeChunk)
        if result.success:
            assert isinstance(result.embedding, list)
            assert all(isinstance(v, float) for v in result.embedding)
            assert result.error is None