import textwrap
from pathlib import Path

import pytest

from app.ingestion.chunker import ASTChunker, CodeChunk


@pytest.fixture
def chunker():
    return ASTChunker()


@pytest.fixture
def tmp_repo(tmp_path):
    """Creates a minimal fake repo structure for testing."""
    # simple Python file with function and class
    (tmp_path / "main.py").write_text(textwrap.dedent('''
        def add(a, b):
            """Adds two numbers."""
            return a + b

        def subtract(a, b):
            return a - b

        class Calculator:
            """A simple calculator."""

            def multiply(self, a, b):
                return a * b

            def divide(self, a, b):
                if b == 0:
                    raise ValueError("division by zero")
                return a / b
    ''').strip())

    # constants file — no functions, should become module chunk
    (tmp_path / "constants.py").write_text(textwrap.dedent('''
        MAX_RETRIES = 3
        TIMEOUT     = 30
        BASE_URL    = "https://api.example.com"
    ''').strip())

    # empty file — should produce no chunks
    (tmp_path / "empty.py").write_text("")

    # javascript file
    (tmp_path / "utils.js").write_text(textwrap.dedent('''
        function fetchUser(userId) {
            return fetch(`/api/users/${userId}`);
        }

        const formatName = (first, last) => {
            return `${first} ${last}`;
        };
    ''').strip())

    # file in a skipped directory — should never be indexed
    node_modules = tmp_path / "node_modules" / "lodash"
    node_modules.mkdir(parents=True)
    (node_modules / "index.js").write_text("module.exports = {};")

    return tmp_path


# ── chunk_repo tests ──────────────────────────────────────────────

def test_chunk_repo_finds_all_files(chunker, tmp_repo):
    chunks = chunker.chunk_repo(tmp_repo)
    assert len(chunks) > 0


def test_chunk_repo_skips_node_modules(chunker, tmp_repo):
    chunks = chunker.chunk_repo(tmp_repo)
    paths  = [c.file_path for c in chunks]
    assert not any("node_modules" in p for p in paths)


def test_chunk_repo_skips_empty_files(chunker, tmp_repo):
    chunks = chunker.chunk_repo(tmp_repo)
    paths  = [c.file_path for c in chunks]
    assert not any("empty.py" in p for p in paths)


# ── Python chunking tests ─────────────────────────────────────────

def test_python_functions_extracted(chunker, tmp_repo):
    chunks = chunker.chunk_repo(tmp_repo)
    py_fns = [
        c for c in chunks
        if c.language == "py" and c.chunk_type == "function"
    ]
    names = [c.name for c in py_fns]
    assert "add"      in names
    assert "subtract" in names


def test_python_class_extracted(chunker, tmp_repo):
    chunks = chunker.chunk_repo(tmp_repo)
    classes = [c for c in chunks if c.chunk_type == "class"]
    assert any(c.name == "Calculator" for c in classes)


def test_python_methods_have_parent_class(chunker, tmp_repo):
    chunks  = chunker.chunk_repo(tmp_repo)
    methods = [c for c in chunks if c.chunk_type == "method"]
    assert len(methods) > 0
    for m in methods:
        assert m.parent_class == "Calculator"


def test_python_docstring_extracted(chunker, tmp_repo):
    chunks = chunker.chunk_repo(tmp_repo)
    add_fn = next(c for c in chunks if c.name == "add")
    assert "Adds two numbers" in add_fn.metadata.get("docstring", "")


def test_content_hash_is_stable(chunker, tmp_repo):
    chunks1 = chunker.chunk_repo(tmp_repo)
    chunks2 = chunker.chunk_repo(tmp_repo)
    hashes1 = {c.content_hash for c in chunks1}
    hashes2 = {c.content_hash for c in chunks2}
    assert hashes1 == hashes2


def test_line_numbers_are_correct(chunker, tmp_repo):
    chunks = chunker.chunk_repo(tmp_repo)
    for chunk in chunks:
        assert chunk.line_start >= 1
        assert chunk.line_end   >= chunk.line_start


# ── Module chunk tests ────────────────────────────────────────────

def test_constants_file_becomes_module_chunk(chunker, tmp_repo):
    chunks      = chunker.chunk_repo(tmp_repo)
    module_chunks = [
        c for c in chunks
        if c.chunk_type == "module" and "constants" in c.file_path
    ]
    assert len(module_chunks) == 1
    assert module_chunks[0].name == "constants"


# ── JavaScript chunking tests ─────────────────────────────────────

def test_javascript_functions_extracted(chunker, tmp_repo):
    chunks = chunker.chunk_repo(tmp_repo)
    js_fns = [
        c for c in chunks
        if c.language == "js" and c.chunk_type == "function"
    ]
    assert len(js_fns) > 0


# ── Embedding text tests ──────────────────────────────────────────

def test_embedding_text_contains_context(chunker, tmp_repo):
    chunks = chunker.chunk_repo(tmp_repo)
    add_fn = next(c for c in chunks if c.name == "add")
    text   = add_fn.to_embedding_text()

    assert "file:"     in text
    assert "type:"     in text
    assert "name:"     in text
    assert "add"       in text


def test_method_embedding_text_contains_class(chunker, tmp_repo):
    chunks  = chunker.chunk_repo(tmp_repo)
    methods = [c for c in chunks if c.chunk_type == "method"]
    assert len(methods) > 0
    for m in methods:
        text = m.to_embedding_text()
        assert "class: Calculator" in text