import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.retrieval.retriever import Retriever, _format_chunks


# ── Helpers ───────────────────────────────────────────────────────

def make_chunk(
    chunk_id:     str  = None,
    name:         str  = "my_function",
    chunk_type:   str  = "function",
    file_path:    str  = "src/auth.py",
    parent_class: str  = "",
    content:      str  = "def my_function(): pass",
    line_start:   int  = 1,
    line_end:     int  = 2,
    similarity:   float = 0.85,
) -> dict:
    """
    Builds a chunk dict matching the shape returned by
    store.similarity_search. All fields have sensible defaults
    so each test only specifies what it cares about.
    """
    return {
        "id":           chunk_id or str(uuid.uuid4()),
        "name":         name,
        "chunk_type":   chunk_type,
        "file_path":    file_path,
        "parent_class": parent_class,
        "content":      content,
        "line_start":   line_start,
        "line_end":     line_end,
        "similarity":   similarity,
    }


def make_retriever(top_k: int = 10, query_top_k: int = 5) -> Retriever:
    mock_store    = MagicMock()
    mock_embedder = MagicMock()

    retriever = Retriever(store=mock_store, embedder=mock_embedder)

    retriever.settings.RETRIEVAL_TOP_K       = top_k
    retriever.settings.RETRIEVAL_QUERY_TOP_K = query_top_k
    retriever.settings.RETRIEVAL_MIN_SCORE   = 0.70

    return retriever


# ── _format_chunks tests ──────────────────────────────────────────

def test_format_chunks_empty_returns_empty_string():
    """Empty input must return empty string — not None, not whitespace."""
    assert _format_chunks([]) == ""


def test_format_chunks_single_chunk_no_parent_class():
    """
    A standalone function (no parent class) should format as:
    [1] function_name (function) — file.py:1-10
    <content>
    """
    chunk = make_chunk(
        name       = "validate_token",
        chunk_type = "function",
        file_path  = "src/auth.py",
        line_start = 1,
        line_end   = 10,
        content    = "def validate_token(token): pass",
    )
    result = _format_chunks([chunk])

    assert "[1] validate_token (function) — src/auth.py:1-10" in result
    assert "def validate_token(token): pass" in result


def test_format_chunks_method_prepends_parent_class():
    """
    A method chunk should show parent class prepended:
    [1] AuthService.validate_token (method) — ...
    """
    chunk = make_chunk(
        name         = "validate_token",
        chunk_type   = "method",
        parent_class = "AuthService",
        file_path    = "src/auth.py",
        line_start   = 5,
        line_end     = 15,
        content      = "def validate_token(self, token): pass",
    )
    result = _format_chunks([chunk])

    assert "AuthService.validate_token (method)" in result


def test_format_chunks_multiple_chunks_joined_by_double_newline():
    """
    Multiple chunks must be separated by double newline so the
    LLM can clearly distinguish where one chunk ends and the next begins.
    """
    chunks = [
        make_chunk(name="func_one", content="def func_one(): pass"),
        make_chunk(name="func_two", content="def func_two(): pass"),
    ]
    result = _format_chunks(chunks)

    assert "\n\n" in result
    assert "[1]" in result
    assert "[2]" in result


def test_format_chunks_similarity_score_not_in_output():
    """
    Similarity scores must never appear in the formatted output.
    Exposing scores to the LLM introduces anchoring bias.
    """
    chunk  = make_chunk(similarity=0.97)
    result = _format_chunks([chunk])

    assert "0.97"      not in result
    assert "similarity" not in result


def test_format_chunks_numbering_is_sequential():
    """
    Chunks must be numbered [1], [2], [3]... in order.
    Position in the prompt implicitly encodes relevance rank.
    """
    chunks = [make_chunk(name=f"func_{i}") for i in range(3)]
    result = _format_chunks(chunks)

    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" in result


# ── _deduplicate tests ────────────────────────────────────────────

def test_deduplicate_empty_input_returns_empty_dict():
    retriever = make_retriever()
    assert retriever._deduplicate([]) == {}


def test_deduplicate_unique_chunks_all_survive():
    """Chunks with different ids should all appear in the output."""
    retriever = make_retriever()
    chunks = [make_chunk(similarity=0.80), make_chunk(similarity=0.75)]

    result = retriever._deduplicate(chunks)

    assert len(result) == 2


def test_deduplicate_keeps_higher_score_for_duplicate_chunk():
    """
    When the same chunk id appears from two queries,
    the higher similarity score must be kept as _best_score.
    This preserves the cross-query relevance signal.
    """
    retriever  = make_retriever()
    chunk_id   = str(uuid.uuid4())

    low_score  = make_chunk(chunk_id=chunk_id, similarity=0.72)
    high_score = make_chunk(chunk_id=chunk_id, similarity=0.95)

    result = retriever._deduplicate([low_score, high_score])

    assert len(result) == 1
    assert result[chunk_id]["_best_score"] == 0.95


def test_deduplicate_keeps_higher_score_regardless_of_order():
    """
    Score selection must not depend on insertion order —
    the higher score wins whether it arrives first or second.
    """
    retriever = make_retriever()
    chunk_id  = str(uuid.uuid4())

    high_score = make_chunk(chunk_id=chunk_id, similarity=0.95)
    low_score  = make_chunk(chunk_id=chunk_id, similarity=0.72)

    result = retriever._deduplicate([high_score, low_score])

    assert result[chunk_id]["_best_score"] == 0.95


def test_deduplicate_injects_best_score_key():
    """
    Every surviving chunk must have _best_score set —
    _rank_and_cap depends on this key being present.
    """
    retriever = make_retriever()
    chunk     = make_chunk(similarity=0.88)

    result = retriever._deduplicate([chunk])

    surviving = next(iter(result.values()))
    assert "_best_score" in surviving
    assert surviving["_best_score"] == 0.88


# ── _rank_and_cap tests ───────────────────────────────────────────

def test_rank_and_cap_sorts_by_best_score_descending():
    """
    Chunks must come out highest score first — position in the
    prompt is the only relevance signal the LLM sees.
    """
    retriever = make_retriever(top_k=10)

    chunk_low  = make_chunk(name="low",  similarity=0.75)
    chunk_high = make_chunk(name="high", similarity=0.95)

    # inject _best_score as _deduplicate would
    chunk_low["_best_score"]  = 0.75
    chunk_high["_best_score"] = 0.95

    deduped = {
        chunk_low["id"]:  chunk_low,
        chunk_high["id"]: chunk_high,
    }
    result = retriever._rank_and_cap(deduped)

    assert result[0]["name"] == "high"
    assert result[1]["name"] == "low"


def test_rank_and_cap_respects_global_cap():
    """
    Output must never exceed RETRIEVAL_TOP_K regardless of
    how many unique chunks deduplication produced.
    """
    retriever = make_retriever(top_k=3)

    chunks = {}
    for i in range(10):
        c = make_chunk(name=f"func_{i}", similarity=0.80)
        c["_best_score"] = 0.80
        chunks[c["id"]] = c

    result = retriever._rank_and_cap(chunks)

    assert len(result) == 3


def test_rank_and_cap_strips_internal_keys():
    """
    _best_score, _query_index, and similarity must all be
    stripped before returning — none of these should reach
    the formatter or the LLM.
    """
    retriever = make_retriever(top_k=10)
    chunk     = make_chunk(similarity=0.88)
    chunk["_best_score"]  = 0.88
    chunk["_query_index"] = 0

    deduped = {chunk["id"]: chunk}
    result  = retriever._rank_and_cap(deduped)

    assert "_best_score"  not in result[0]
    assert "_query_index" not in result[0]
    assert "similarity"   not in result[0]


# ── search integration tests ──────────────────────────────────────

async def test_search_empty_queries_returns_empty_string():
    """
    Empty query list must short-circuit immediately.
    No calls to embedder or store should be made.
    """
    retriever = make_retriever()
    retriever.embedder.embed_texts = AsyncMock()
    retriever.store.similarity_search = AsyncMock()

    result = await retriever.search(queries=[], repo_id=uuid.uuid4())

    assert result == ""
    retriever.embedder.embed_texts.assert_not_called()
    retriever.store.similarity_search.assert_not_called()


async def test_search_no_results_returns_empty_string():
    """
    If similarity_search returns nothing above min_score,
    search must return empty string — not raise, not return None.
    """
    retriever = make_retriever()
    retriever.embedder.embed_texts    = AsyncMock(return_value=[[0.1] * 768])
    retriever.store.similarity_search = AsyncMock(return_value=[])

    result = await retriever.search(
        queries = ["validate token"],
        repo_id = uuid.uuid4(),
    )

    assert result == ""


async def test_search_happy_path_returns_formatted_string():
    """
    Happy path — one query, one result, formatted string returned.
    Confirms the full pipeline wires together correctly.
    """
    retriever = make_retriever()
    repo_id   = uuid.uuid4()
    chunk     = make_chunk(
        name       = "validate_token",
        chunk_type = "function",
        file_path  = "src/auth.py",
        content    = "def validate_token(token): pass",
        similarity = 0.90,
    )

    retriever.embedder.embed_texts    = AsyncMock(return_value=[[0.1] * 768])
    retriever.store.similarity_search = AsyncMock(return_value=[chunk])

    result = await retriever.search(
        queries = ["validate token authentication"],
        repo_id = repo_id,
    )

    assert "validate_token" in result
    assert "src/auth.py"    in result
    assert "0.90"           not in result   # score must be stripped


async def test_search_deduplicates_across_queries():
    """
    The same chunk returned by two different queries must appear
    only once in the output — not duplicated in the prompt.
    """
    retriever = make_retriever(top_k=10)
    repo_id   = uuid.uuid4()
    chunk_id  = str(uuid.uuid4())

    # same chunk id returned by both queries with different scores
    chunk_q1 = make_chunk(chunk_id=chunk_id, name="validate_token", similarity=0.95)
    chunk_q2 = make_chunk(chunk_id=chunk_id, name="validate_token", similarity=0.80)

    retriever.embedder.embed_texts = AsyncMock(
        return_value=[[0.1] * 768, [0.2] * 768]
    )
    retriever.store.similarity_search = AsyncMock(
        side_effect=[[chunk_q1], [chunk_q2]]
    )

    result = await retriever.search(
        queries = ["validate token", "authentication token"],
        repo_id = repo_id,
    )

    # chunk should appear exactly once — [1] but not [2]
    assert result.count("[1]") == 1
    assert "[2]"               not in result


async def test_search_calls_similarity_search_once_per_query():
    """
    similarity_search must be called exactly N times for N queries —
    one search per query embedding, no more, no less.
    """
    retriever = make_retriever()
    retriever.embedder.embed_texts = AsyncMock(
        return_value=[[0.1] * 768, [0.2] * 768, [0.3] * 768]
    )
    retriever.store.similarity_search = AsyncMock(return_value=[])

    await retriever.search(
        queries = ["query one", "query two", "query three"],
        repo_id = uuid.uuid4(),
    )

    assert retriever.store.similarity_search.call_count == 3