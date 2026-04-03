# CogniDiff

CogniDiff reviews GitHub pull requests automatically. You open a PR, it fetches the diff, searches your codebase for relevant context, runs it through an LLM, and posts a structured review - APPROVE, REQUEST_CHANGES, or COMMENT - directly on the PR.

---

## How it works

![Architecture Diagram](./assets/architecture.svg)

The codebase index is built separately, triggered on every push to your default branch. The agent queries it at review time to understand what the changed code sits next to.

---

## Key decisions

Some key decisions I was deliberate about, worth knowing before you read the code

### 1. AST-aware chunking

Most RAG pipelines split code by character count. This one uses tree-sitter to parse the actual syntax tree and extract complete units - functions, methods, classes = as individual chunks.

A line-based split can cut a function in half. The embedding then represents neither its intent nor its structure, which makes retrieval noisy. AST chunking guarantees every chunk is a complete, meaningful unit.

### 2. Incremental re-embedding

Every chunk is hashed from its content and file path. On re-index, unchanged chunks are copied forward - they're never re-embedded. Only genuinely modified code hits the embedding model.

The file path is included in the hash deliberately. A function that moves files produces a different hash even if the body is identical - the file context changed.

### 3. Transient vs permanent failures

The exception hierarchy separates failures worth retrying from ones that aren't - at the type level, not via string matching.

```
TransientError  →  retry  (rate limits, network issues, service down)
PermanentError  →  stop   (bad credentials, 404, malformed data)
```

Blindly retrying a 401 wastes time and burns API quota. Retrying a 429 with backoff is exactly right. Retry logic only triggers on transient failures.

### 4. RAG scoping for token efficiency

The retriever runs one query per changed file, then applies two caps before anything reaches the LLM: one per query (to limit DB load) and one globally (to control prompt size). Deduplication keeps the highest similarity score when the same chunk appears in multiple queries.

Similarity scores are stripped before the LLM sees the context. They're internal routing signals - exposing them risks the model over-weighting a 0.97 chunk vs a 0.71 one regardless of actual relevance.

### 5. Raw unified diff over the files API

GitHub exposes diffs two ways: a JSON `/files` endpoint and a raw unified diff. The files endpoint silently truncates at 300 files and drops the `patch` field for large diffs. A partial review is worse than no review - a reviewer who missed half the changes is a false signal of quality. CogniDiff uses the raw diff, which has no such limits.

### 6. Structured output via constrained decoding

The LLM produces a typed `PullRequestReview` object, not free-form text. For Ollama this uses `json_schema` mode (constrained decoding at the token level). For Azure OpenAI it uses native structured output. The interface is identical - switching providers requires no code changes.

Prompt-level instructions like "respond in JSON" fail silently. Constrained decoding makes malformed output structurally impossible.


---

## Tech stack

| | |
|---|---|
| API server | FastAPI |
| Agent framework | LangGraph |
| Job queue | ARQ + Redis |
| Vector database | pgvector (PostgreSQL) |
| Code parsing | tree-sitter |
| LLM — local | Ollama (`qwen2.5-coder`) |
| LLM — cloud | Azure OpenAI (`gpt-4o`) |
| Embeddings — local | Ollama (`nomic-embed-text`) |
| Embeddings — cloud | Azure (`text-embedding-3-small`) |

Switch between local and cloud with one environment variable. No code changes.

---

## Getting started

### Requirements

- Python 3.11+
- PostgreSQL with pgvector
- Redis
- Ollama (local) or Azure OpenAI credentials (cloud)

### Install

```bash
git clone https://github.com/Rijulmehta531/cognidiff.git
cd cognidiff

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
```

Minimum required:

```env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/cognidiff
SYNC_DATABASE_URL=postgresql+psycopg2://user:pass@localhost/cognidiff
REDIS_URL=redis://localhost:6379
GITHUB_WEBHOOK_SECRET=your-webhook-secret
GITHUB_TOKEN=your-personal-access-token

# switch between local and cloud here
LLM_PROVIDER=ollama

# local
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5-coder:3b
OLLAMA_EMBED_MODEL=nomic-embed-text

# cloud (uncomment if LLM_PROVIDER=azure)
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_API_KEY=your-key
```

### Database

```bash
psql -U postgres -c "CREATE DATABASE cognidiff;"
psql -U postgres -d cognidiff -f init.sql
```

### Run

```bash
# terminal 1
uvicorn app.main:app --reload

# terminal 2
python -m arq worker.WorkerSettings
```

### Webhook setup

In your GitHub repo settings, add a webhook:

- Payload URL: `https://your-domain.com/webhooks/github`
- Content type: `application/json`
- Secret: same value as `GITHUB_WEBHOOK_SECRET`
- Events: `Pushes` and `Pull requests`

### Tests

```bash
pytest tests/ -v
```

---
