import logging
from dataclasses import dataclass
from typing import Optional

from app.config import get_embedder, get_settings
from app.ingestion.chunker import CodeChunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """
    Result of embedding a single chunk.
    Separates success from failure cleanly —
    caller decides what to do with failed chunks.
    """
    chunk:     CodeChunk
    embedding: Optional[list[float]]
    error:     Optional[str] = None

    @property
    def success(self) -> bool:
        return self.embedding is not None


class Embedder:

    def __init__(self):
        self.settings = get_settings()
        self.embedder = get_embedder()
        self.batch_size = 32

    async def embed_chunks(
        self,
        chunks: list[CodeChunk],
    ) -> list[EmbeddingResult]:
        """
        Embeds all chunks in batches.
        Returns one EmbeddingResult per chunk — never raises.
        Failed chunks have embedding=None and error populated.
        """
        if not chunks:
            return []

        results  = []
        total    = len(chunks)
        embedded = 0
        failed   = 0

        for i in range(0, total, self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_results = await self._embed_batch(batch)
            results.extend(batch_results)

            embedded += sum(1 for r in batch_results if r.success)
            failed   += sum(1 for r in batch_results if not r.success)

            logger.info(
                f"embedding progress: {min(i + self.batch_size, total)}"
                f"/{total} chunks"
            )

        logger.info(
            f"embedding complete: {embedded} succeeded, "
            f"{failed} failed"
        )
        return results

    async def _embed_batch(
        self,
        batch: list[CodeChunk],
    ) -> list[EmbeddingResult]:
        """
        Embeds one batch of chunks.
        If the whole batch fails, falls back to embedding
        one chunk at a time so partial failures don't lose
        the entire batch.
        """
        texts = [chunk.to_embedding_text() for chunk in batch]

        try:
            # embed_documents is synchronous in LangChain —
            # run in executor to avoid blocking the event loop
            embeddings = await self._run_embed(texts)

            return [
                EmbeddingResult(chunk=chunk, embedding=embedding)
                for chunk, embedding in zip(batch, embeddings)
            ]

        except Exception as e:
            logger.warning(
                f"batch of {len(batch)} failed: {e}. "
                f"falling back to one-by-one embedding"
            )
            return await self._embed_one_by_one(batch)

    async def _embed_one_by_one(
        self,
        chunks: list[CodeChunk],
    ) -> list[EmbeddingResult]:
        """
        Fallback: embed each chunk individually.
        Slower but isolates failures to individual chunks
        instead of losing the whole batch.
        """
        results = []
        for chunk in chunks:
            try:
                embeddings = await self._run_embed(
                    [chunk.to_embedding_text()]
                )
                results.append(
                    EmbeddingResult(
                        chunk=chunk,
                        embedding=embeddings[0],
                    )
                )
            except Exception as e:
                logger.error(
                    f"failed to embed {chunk.file_path}:"
                    f"{chunk.name} — {e}"
                )
                results.append(
                    EmbeddingResult(
                        chunk=chunk,
                        embedding=None,
                        error=str(e),
                    )
                )
        return results

    async def _run_embed(self, texts: list[str]) -> list[list[float]]:
        """
        Runs embed_documents in a thread pool executor.

        Why: LangChain's embed_documents is synchronous.
        Calling it directly in an async function blocks the
        entire event loop until embedding completes — meaning
        no other coroutines can run during that time.

        run_in_executor offloads it to a thread so the event
        loop stays free to handle other work.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.embedder.embed_documents,
            texts,
        )

    def validate_embedding_dimension(
        self,
        embedding: list[float],
    ) -> bool:
        """
        Verifies the embedding dimension matches what the
        database expects. A mismatch means the wrong model
        was used — would cause pgvector to reject the insert.
        """
        expected = self.settings.EMBEDDING_DIMENSION
        actual   = len(embedding)
        if actual != expected:
            logger.error(
                f"embedding dimension mismatch: "
                f"expected {expected}, got {actual}. "
                f"check OLLAMA_EMBED_MODEL or AZURE_EMBED_DEPLOYMENT"
            )
            return False
        return True