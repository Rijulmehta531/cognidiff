import logging
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
from tree_sitter import Language, Parser

from app.config import get_settings
from app.ingestion.cloner import compute_content_hash

logger = logging.getLogger(__name__)

# ── Language setup ────────────────────────────────────────────────
# Initialised once at module load — not per file or per chunk.
# Parser objects are reused across all files.
PY_LANGUAGE = Language(tspython.language())
JS_LANGUAGE = Language(tsjavascript.language())

PARSERS: dict[str, Parser] = {
    ".py":  Parser(PY_LANGUAGE),
    ".js":  Parser(JS_LANGUAGE),
    ".ts":  Parser(JS_LANGUAGE),  # TS grammar close enough for chunking
    ".tsx": Parser(JS_LANGUAGE),
    ".jsx": Parser(JS_LANGUAGE),
}

# tree-sitter node types we extract per language
# any node type not listed here is ignored during walk
CHUNK_NODE_TYPES: dict[str, dict[str, str]] = {
    ".py": {
        "function_definition":       "function",
        "async_function_definition": "function",
        "class_definition":          "class",
    },
    ".js": {
        "function_declaration":      "function",
        "function_expression":       "function",
        "arrow_function":            "function",
        "class_declaration":         "class",
        "method_definition":         "method",
    },
    ".ts": {
        "function_declaration":      "function",
        "function_expression":       "function",
        "arrow_function":            "function",
        "class_declaration":         "class",
        "method_definition":         "method",
    },
}
# jsx and tsx share js/ts node types
CHUNK_NODE_TYPES[".tsx"] = CHUNK_NODE_TYPES[".ts"]
CHUNK_NODE_TYPES[".jsx"] = CHUNK_NODE_TYPES[".js"]

# files and directories we never want to index
SKIP_DIRS = {
    ".git", ".github", "node_modules", "__pycache__",
    ".venv", "venv", "dist", "build", ".next",
    "coverage", ".pytest_cache", "migrations",
}
SKIP_FILES = {
    ".env", ".env.example", ".gitignore",
    "package-lock.json", "yarn.lock", "poetry.lock",
}


# ── Data model ────────────────────────────────────────────────────

@dataclass
class CodeChunk:
    """
    One indexable unit extracted from a source file.
    Maps directly to a row in the code_chunks table.
    """
    file_path:    str
    chunk_type:   str       # function | class | method | module
    name:         str
    parent_class: str
    language:     str       # py | js | ts
    content:      str
    content_hash: str
    line_start:   int
    line_end:     int
    metadata:     dict = field(default_factory=dict)

    @property
    def line_count(self) -> int:
        return self.line_end - self.line_start + 1

    def to_embedding_text(self) -> str:
        """
        What actually gets embedded — not just raw code.
        Prepending context improves retrieval quality because
        the embedding captures semantics of the full description,
        not just the syntax of the code.
        """
        parts = [
            f"file: {self.file_path}",
            f"type: {self.chunk_type}",
            f"name: {self.name}",
        ]
        if self.parent_class:
            parts.append(f"class: {self.parent_class}")
        if self.metadata.get("docstring"):
            parts.append(f"description: {self.metadata['docstring']}")
        if self.metadata.get("calls"):
            parts.append(f"calls: {', '.join(self.metadata['calls'][:10])}")
        parts.append(f"\n{self.content}")
        return "\n".join(parts)


# ── Chunker ───────────────────────────────────────────────────────

class ASTChunker:

    def __init__(self):
        self.settings = get_settings()

    def chunk_repo(self, repo_path: Path) -> list[CodeChunk]:
        """
        Walks a repo directory and chunks all supported files.
        Returns a flat list of all chunks across all files.
        """
        chunks  = []
        skipped = 0

        for file_path in repo_path.rglob("*"):
            # skip directories
            if file_path.is_dir():
                continue

            # skip ignored directories anywhere in the path
            if any(part in SKIP_DIRS for part in file_path.parts):
                skipped += 1
                continue

            # skip ignored filenames
            if file_path.name in SKIP_FILES:
                skipped += 1
                continue

            # skip unsupported extensions
            if file_path.suffix not in PARSERS:
                continue

            try:
                file_chunks = self.chunk_file(file_path, repo_path)
                chunks.extend(file_chunks)
            except Exception as e:
                # log but don't crash — one bad file shouldn't
                # stop the entire repo from being indexed
                logger.warning(f"failed to chunk {file_path}: {e}")

        logger.info(
            f"chunked repo: {len(chunks)} chunks, "
            f"{skipped} files skipped"
        )
        return chunks

    def chunk_file(
        self,
        file_path: Path,
        repo_root: Path,
    ) -> list[CodeChunk]:
        """
        Parses one file and returns its chunks.
        Falls back to a single module chunk if:
          - no functions/classes found
          - file is unsupported but has content
        """
        suffix  = file_path.suffix
        source  = self._read_source(file_path)

        if not source.strip():
            return []   # empty file — nothing to index

        # relative path for storage — no machine-specific prefix
        rel_path = str(file_path.relative_to(repo_root))
        language = suffix.lstrip(".")

        if suffix not in PARSERS:
            return []

        parser   = PARSERS[suffix]
        node_map = CHUNK_NODE_TYPES.get(suffix, {})
        tree     = parser.parse(source.encode())
        chunks   = []

        self._walk(
            node       = tree.root_node,
            source     = source,
            rel_path   = rel_path,
            language   = language,
            node_map   = node_map,
            chunks     = chunks,
            parent_cls = "",
        )

        # no structural chunks found — capture as module chunk
        if not chunks:
            chunks = self._make_module_chunk(
                source   = source,
                rel_path = rel_path,
                language = language,
            )

        return chunks

    def _walk(
        self,
        node:       object,
        source:     str,
        rel_path:   str,
        language:   str,
        node_map:   dict,
        chunks:     list,
        parent_cls: str,
    ) -> None:
        """
        Recursively walks the CST and extracts chunk nodes.
        When a class is found, recurses into it with the class
        name as parent_cls so methods know their parent.
        """
        node_type = node.type

        if node_type in node_map:
            chunk_type = node_map[node_type]
            content    = source[node.start_byte:node.end_byte]
            name       = self._extract_name(node, source)
            line_start = node.start_point[0] + 1  # tree-sitter is 0-indexed
            line_end   = node.end_point[0]   + 1

            chunk = CodeChunk(
                file_path    = rel_path,
                chunk_type   = chunk_type,
                name         = name,
                parent_class = parent_cls,
                language     = language,
                content      = content,
                content_hash = compute_content_hash(content),
                line_start   = line_start,
                line_end     = line_end,
                metadata     = self._build_metadata(
                    node, source, language, line_start, line_end
                ),
            )
            if chunk.language == "py" and chunk.chunk_type == "function" and chunk.parent_class:
                chunk.chunk_type = "method"
            chunks.append(chunk)

            # if this is a class — recurse into it so methods
            # know their parent class name
            if chunk_type == "class":
                for child in node.children:
                    self._walk(
                        child, source, rel_path, language,
                        node_map, chunks, name,
                    )
                return  # don't double-walk children below

        for child in node.children:
            self._walk(
                child, source, rel_path, language,
                node_map, chunks, parent_cls,
            )

    def _make_module_chunk(
        self,
        source:   str,
        rel_path: str,
        language: str,
    ) -> list[CodeChunk]:
        """
        Creates a single module-level chunk for files with
        no extractable functions or classes — e.g. constants.py,
        __init__.py with only imports.
        """
        settings   = self.settings
        line_count = source.count("\n") + 1
        oversized  = line_count > settings.CHUNK_MAX_LINES

        return [CodeChunk(
            file_path    = rel_path,
            chunk_type   = "module",
            name         = Path(rel_path).stem,
            parent_class = "",
            language     = language,
            content      = source,
            content_hash = compute_content_hash(source),
            line_start   = 1,
            line_end     = line_count,
            metadata     = {
                "oversized":  oversized,
                "line_count": line_count,
            },
        )]

    def _build_metadata(
        self,
        node:       object,
        source:     str,
        language:   str,
        line_start: int,
        line_end:   int,
    ) -> dict:
        """
        Builds the metadata dict stored alongside the chunk.
        Includes docstring, call graph, and size signals.
        """
        settings   = self.settings
        line_count = line_end - line_start + 1
        oversized  = line_count > settings.CHUNK_MAX_LINES

        metadata = {
            "line_count": line_count,
            "oversized":  oversized,
        }

        # extract docstring if Python
        if language == "py":
            docstring = self._extract_python_docstring(node, source)
            if docstring:
                metadata["docstring"] = docstring

        # extract function calls — useful context for RAG
        calls = self._extract_calls(node, source)
        if calls:
            metadata["calls"] = calls

        return metadata

    def _extract_name(self, node, source: str) -> str:
        """Extracts the identifier name from a node."""
        for child in node.children:
            if child.type == "identifier":
                return source[child.start_byte:child.end_byte]
        return "anonymous"

    def _extract_python_docstring(self, node, source: str) -> str:
        """
        Extracts the first string literal after a function/class
        signature — Python's docstring convention.
        Only applied to Python files.
        """
        for child in node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        for s in stmt.children:
                            if s.type == "string":
                                raw = source[s.start_byte:s.end_byte]
                                # strip quotes and whitespace
                                return raw.strip("\"' \n\t")
        return ""

    def _extract_calls(self, node, source: str) -> list[str]:
        """
        Walks the subtree collecting function call identifiers.
        Capped at 20 — beyond that it's noise not signal.
        """
        calls: list[str] = []
        self._collect_calls(node, source, calls)
        # deduplicate while preserving order
        seen = set()
        unique = []
        for c in calls:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique[:20]

    def _collect_calls(
        self, node, source: str, calls: list[str]
    ) -> None:
        if node.type == "call":
            for child in node.children:
                if child.type == "identifier":
                    name = source[child.start_byte:child.end_byte]
                    if name not in {"print", "len", "str",
                                    "int", "float", "bool",
                                    "list", "dict", "set"}:
                        calls.append(name)
                elif child.type == "attribute":
                    name = source[child.start_byte:child.end_byte]
                    calls.append(name)
        for child in node.children:
            self._collect_calls(child, source, calls)

    @staticmethod
    def _read_source(file_path: Path) -> str:
        """
        Reads source with fallback encoding.
        Some files in the wild are not UTF-8.
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding="latin-1", errors="ignore")