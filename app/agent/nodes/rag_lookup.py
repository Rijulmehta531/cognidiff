import logging
import re

from app.agent.state import AgentState
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

# file extensions we skip retrieval for
_NON_CODE_EXTENSIONS = {
    ".md", ".rst", ".txt",           # documentation
    ".json", ".yaml", ".yml",        # config
    ".lock", ".toml", ".cfg", ".ini", # dependency/config files
    ".png", ".jpg", ".jpeg", ".svg",  # assets
}

_DEF_RE = re.compile(r"(?:async\s+)?def\s+([a-zA-Z_]\w*)")
_CLASS_RE = re.compile(r"class\s+([a-zA-Z_]\w*)")
_IDENT_RE = re.compile(r"[a-zA-Z_]\w*")

async def rag_lookup(state: AgentState) -> AgentState:
    """
    Node 2: Constructs targeted queries from the PR diff and
    retrieves relevant codebase context from the vector DB.

    Reads:  pr_diff, repo_id
    Writes: retrieved_context  — formatted string for LLM prompt
            retrieval_skipped  — True if nothing was found or
                                 retrieval failed softly

    Soft failure (no results) → sets retrieval_skipped=True, continues.
    Hard failure (embedder/DB down) → sets error, routes to END.

    """
    pr_diff   = state["pr_diff"]
    repo_id   = state["repo_id"]
    prefix    = f"[rag_lookup:{state['full_name']}#{state['pr_number']}]"

    logger.info(f"{prefix} constructing queries from diff")

    # ── Step 1: Extract queries from diff ────────────────────────
    queries = _build_queries(pr_diff)

    if not queries:
        logger.info(f"{prefix} no queries extracted — skipping retrieval")
        return {
            **state,
            "retrieved_context": "",
            "retrieval_skipped": True,
        }

    logger.info(f"{prefix} {len(queries)} queries constructed")

    # ── Step 2: Retrieve codebase context ────────────────────────
    try:
        retriever = Retriever()
        context   = await retriever.search(
            queries = queries,
            repo_id = repo_id,
        )

        if not context:
            logger.info(
                f"{prefix} retrieval returned no results "
                f"above min score — continuing without context"
            )
            return {
                **state,
                "retrieved_context": "",
                "retrieval_skipped": True,
            }

        logger.info(f"{prefix} retrieval complete — context ready")
        return {
            **state,
            "retrieved_context": context,
            "retrieval_skipped": False,
        }

    except Exception as e:
        logger.error(
            f"{prefix} retrieval failed — {e}. "
            f"continuing without context",
            exc_info=True,
        )
        return {
            **state,
            "retrieved_context": "",
            "retrieval_skipped": True,
        }


# ── Query construction ────────────────────────────────────────────

def _build_queries(pr_diff) -> list[str]:
    """
    Extracts targeted search queries from the PR diff.

    Query construction strategy:
      - One query per changed code file (skips non-code files)
      - Query is built from: filename stem + symbols changed
    """
    queries  = []

    for diff_file in pr_diff.files:
        if diff_file.status == "removed":
            continue

        ext = _get_extension(diff_file.filename)
        if ext in _NON_CODE_EXTENSIONS:
            continue

        query = _build_file_query(diff_file)
        if query:
            queries.append(query)

    return queries


def _build_file_query(diff_file) -> str:
    """
    Builds a single search query for one changed file.

    Combines:
      - The file's module name (stem of the filename) — gives
        the embedding a topic anchor
      - Symbols extracted from the diff hunks — function names,
        class names, and identifiers that appear in changed lines

    Example output:
      "auth validate_token AuthService check_permissions"

    The symbols are extracted from added/removed lines only —
    context lines (prefixed with space) represent unchanged code
    and add noise rather than signal.
    """
    # use the last two path components for context —
    # "auth/validator" is more meaningful than just "validator"
    path_parts  = diff_file.filename.replace("\\", "/").split("/")
    module_name = "/".join(path_parts[-2:]) if len(path_parts) > 1 else path_parts[0]
    # strip extension from last component
    module_name = re.sub(r"\.[^/]+$", "", module_name)

    symbols = _extract_symbols(diff_file)

    if not symbols:
        return module_name

    # capping symbols to avoid query becoming too broad
    symbol_str = " ".join(symbols[:8])
    return f"{module_name} {symbol_str}"


def _extract_symbols(diff_file) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()

    for line in _iter_changed_lines(diff_file):
        name = _match_function_name(line)
        if name:
            _add_symbol(name, symbols, seen)
            continue

        name = _match_class_name(line)
        if name:
            _add_symbol(name, symbols, seen)
            continue

        for token in _IDENT_RE.findall(line):
            if _is_meaningful_identifier(token):
                _add_symbol(token, symbols, seen)

    return symbols


def _add_symbol(name: str, symbols: list[str], seen: set[str]) -> None:
    if name in seen or _is_noise(name):
        return
    seen.add(name)
    symbols.append(name)

def _iter_changed_lines(diff_file):
    for hunk in diff_file.hunks:
        for line in hunk.content.splitlines():
            if line.startswith(("+", "-")):
                yield line[1:].strip()

def _match_function_name(line: str) -> str | None:
    match = _DEF_RE.match(line)
    return match.group(1) if match else None


def _match_class_name(line: str) -> str | None:
    match = _CLASS_RE.match(line)
    return match.group(1) if match else None

def _is_noise(name: str) -> bool:
    """
    Returns True for names that add no retrieval signal.
    """
    _NOISE = {
        # Python builtins
        "self", "cls", "None", "True", "False",
        "return", "import", "from", "pass", "raise",
        "if", "else", "elif", "for", "while", "with",
        "try", "except", "finally", "and", "or", "not",
        "in", "is", "as", "lambda", "yield", "await",
        "async", "def", "class",
        # common short names
        "str", "int", "float", "bool", "list", "dict",
        "set", "tuple", "len", "range", "print", "type",
        "super", "property", "staticmethod", "classmethod",
    }
    return name in _NOISE or len(name) <= 3


def _is_meaningful_identifier(token: str) -> bool:
    """
    Returns True for identifiers worth including in a query.
    """
    # CamelCase — likely a class or type reference
    if re.match(r"[A-Z][a-zA-Z0-9]+", token):
        return True
    # snake_case longer than 5 — likely a meaningful domain name
    if "_" in token and len(token) > 5:
        return True
    return False


def _get_extension(filename: str) -> str:
    """Returns the lowercased file extension including the dot."""
    parts = filename.rsplit(".", 1)
    return f".{parts[1].lower()}" if len(parts) > 1 else ""