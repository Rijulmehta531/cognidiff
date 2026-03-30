import logging
import uuid
from typing import Optional

from app.config import get_settings
from app.ingestion.embedder import Embedder
from app.retrieval.store import CodeStore

logger = logging.getLogger(__name__)


# ── Formatting ────────────────────────────────────────────────────

def _format_chunks(chunks: list[dict]) -> str:
    """
    Formats a ranked list of deduplicated chunk dicts into a
    single prompt-ready string.

    Returns an empty string if no chunks are provided, so the
    caller can do a simple truthiness check without special casing.
    """
    if not chunks:
        return ""

    parts = []

    for i, chunk in enumerate(chunks, start=1):
        parent  = chunk.get("parent_class", "")
        name    = f"{parent}.{chunk['name']}" if parent else chunk["name"]
        header  = (
            f"[{i}] {name} ({chunk['chunk_type']}) "
            f"— {chunk['file_path']}:"
            f"{chunk['line_start']}-{chunk['line_end']}"
        )
        parts.append(f"{header}\n{chunk['content']}")

    return "\n\n".join(parts)


# ── Retriever ─────────────────────────────────────────────────────

class Retriever:
    """
    RAG lookup layer for PR review agent.

    Responsibilities:
      - Embed one or more natural language queries
      - Run per-query similarity search against the vector DB
      - Deduplicate results across queries by chunk id
      - Re-rank deduplicated results by best score seen
      - Apply global cap to control prompt budget
      - Format into a single prompt-ready string
    """

    def __init__(
        self,
        store:    Optional[CodeStore] = None,
        embedder: Optional[Embedder]  = None,
    ):
        self.settings = get_settings()
        self.store    = store    or CodeStore()
        self.embedder = embedder or Embedder()

    async def search(
        self,
        queries:    list[str],
        repo_id:    uuid.UUID,
        language:   Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> str:
        """
        Entry point for the agent's rag_lookup node.

        Accepts multiple queries, runs them against the vector DB,
        deduplicates and re-ranks the results, then returns a single
        formatted string ready to be inserted into an LLM prompt.

        Args:
            queries:    One or more natural language queries
                        extracted from the PR diff. Caller is
                        responsible for constructing these.
            repo_id:    The repository to search against.
            language:   Optional filter — e.g. "py" or "js".
            chunk_type: Optional filter — e.g. "function".

        Returns:
            Formatted string of top-k chunks, or empty string
            if nothing was found above the minimum score.
        """
        if not queries:
            logger.warning("[retriever] search called with no queries — returning empty")
            return ""

        prefix = f"[retriever:{repo_id}]"
        logger.info(f"{prefix} starting search — {len(queries)} queries")

        # ── Step 1: Embed all queries in one batch ────────────────
        query_embeddings = await self.embedder.embed_texts(queries)
        logger.info(f"{prefix} embedded {len(query_embeddings)} queries")

        # ── Step 2: Search per query, collect all raw results ─────
        raw_results = await self._search_all_queries(
            query_embeddings = query_embeddings,
            repo_id          = repo_id,
            language         = language,
            chunk_type       = chunk_type,
            prefix           = prefix,
        )
        logger.info(
            f"{prefix} raw results before dedup — "
            f"{len(raw_results)} chunks across all queries"
        )

        # ── Step 3: Deduplicate by chunk id ──────────────────────
        deduplicated = self._deduplicate(raw_results)
        logger.info(
            f"{prefix} after deduplication — "
            f"{len(deduplicated)} unique chunks"
        )

        # ── Step 4: Re-rank by best score, apply global cap ───────
        top_chunks = self._rank_and_cap(deduplicated)
        logger.info(
            f"{prefix} returning {len(top_chunks)} chunks "
            f"(global cap: {self.settings.RETRIEVAL_TOP_K})"
        )

        # ── Step 5: Format for LLM prompt ─────────────────────────
        return _format_chunks(top_chunks)

    # ── Private helpers ───────────────────────────────────────────

    async def _search_all_queries(
        self,
        query_embeddings: list[list[float]],
        repo_id:          uuid.UUID,
        language:         Optional[str],
        chunk_type:       Optional[str],
        prefix:           str,
    ) -> list[dict]:
        """
        Runs similarity_search for every query embedding and
        collects all results into a flat list.

        Each result dict gets a '_query_index' key injected so
        we can trace which query produced which result during
        deduplication — useful for debugging retrieval quality.
        """
        all_results = []

        for idx, embedding in enumerate(query_embeddings):
            results = await self.store.similarity_search(
                query_embedding = embedding,
                repo_id         = repo_id,
                top_k           = self.settings.RETRIEVAL_QUERY_TOP_K,
                min_score       = self.settings.RETRIEVAL_MIN_SCORE,
                language        = language,
                chunk_type      = chunk_type,
            )
            logger.info(
                f"{prefix} query {idx + 1} returned "
                f"{len(results)} results"
            )
            for result in results:
                result["_query_index"] = idx

            all_results.extend(results)

        return all_results

    def _deduplicate(self, results: list[dict]) -> dict[str, dict]:
        """
        Deduplicates results by chunk id, keeping the highest
        similarity score seen across all queries for each chunk.

        Returns a dict keyed by chunk id — the score is stored
        under '_best_score' for use in re-ranking.

        Why keep best score rather than average?
        A chunk that scores 0.95 against one query and 0.60
        against another is highly relevant to the first query.
        Averaging would dilute that signal. Best score preserves it.
        """
        seen: dict[str, dict] = {}

        for result in results:
            chunk_id = str(result["id"])
            score    = result["similarity"]

            if chunk_id not in seen:
                result["_best_score"] = score
                seen[chunk_id] = result
            elif score > seen[chunk_id]["_best_score"]:
                seen[chunk_id]["_best_score"] = score

        return seen

    def _rank_and_cap(self, deduplicated: dict[str, dict]) -> list[dict]:
        """
        Sorts unique chunks by best score descending, then applies
        the global RETRIEVAL_TOP_K cap for prompt budget control.

        """
        ranked = sorted(
            deduplicated.values(),
            key     = lambda c: c["_best_score"],
            reverse = True,
        )
        capped = ranked[: self.settings.RETRIEVAL_TOP_K]

        # strip internal tracking keys before handing off
        for chunk in capped:
            chunk.pop("_best_score",  None)
            chunk.pop("_query_index", None)
            chunk.pop("similarity",   None)   # score excluded from LLM output

        return capped