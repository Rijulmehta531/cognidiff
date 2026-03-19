import hashlib
import logging
import tarfile
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
import httpx

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


@asynccontextmanager
async def download_repo(
    full_name: str,
    ref:       str,
    token:     str,
):
    """
    Downloads a GitHub repo as a tarball and extracts it
    to a temp directory. Cleans up on exit - even on error.

    Usage:
        async with download_repo("owner/repo", "main", token) as path:
            # path is the extracted repo root
            # temp dir deleted automatically after this block
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="cognidiff_"))
    logger.info(f"created temp dir {tmpdir} for {full_name}@{ref}")

    try:
        repo_path = await _download_and_extract(
            full_name=full_name,
            ref=ref,
            token=token,
            dest=tmpdir,
        )
        yield repo_path

    finally:
        # guaranteed cleanup — runs even if caller raises
        _cleanup(tmpdir)
        logger.info(f"cleaned up {tmpdir}")


async def _download_and_extract(
    full_name: str,
    ref:       str,
    token:     str,
    dest:      Path,
) -> Path:
    """
    Downloads the tarball and returns the path to the
    extracted repo root directory.
    """
    from app.config import get_settings
    settings = get_settings()

    url     = f"{GITHUB_API}/repos/{full_name}/tarball/{ref}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/vnd.github+json",
        "X-GitHub-API-Version": settings.GITHUB_API_VERSION,
    }

    tarball_path = dest / "repo.tar.gz"

    async with httpx.AsyncClient(
        follow_redirects=True,   # GitHub redirects to S3 for the actual file
        timeout=120.0,           # large repos can be slow
    ) as client:
        logger.info(f"downloading {full_name}@{ref}")
        await _stream_download(client, url, headers, tarball_path)

    logger.info(f"extracting {tarball_path}")
    repo_root = _extract_tarball(tarball_path, dest)

    # delete the archive — we only need the extracted files
    tarball_path.unlink()

    return repo_root


async def _stream_download(
    client:       httpx.AsyncClient,
    url:          str,
    headers:      dict,
    destination:  Path,
) -> None:
    """
    Streams the download to disk in chunks.
    Avoids loading the entire tarball into memory —
    important for large repos.
    """
    async with client.stream("GET", url, headers=headers) as response:
        response.raise_for_status()

        total    = int(response.headers.get("content-length", 0))
        received = 0

        async with aiofiles.open(destination, "wb") as f:
            async for chunk in response.aiter_bytes(chunk_size=8192):
                await f.write(chunk)
                received += len(chunk)

        if total and received != total:
            raise ValueError(
                f"incomplete download: got {received} of {total} bytes"
            )

    logger.info(f"downloaded {received:,} bytes to {destination}")


def _extract_tarball(tarball_path: Path, dest: Path) -> Path:
    """
    Extracts the tarball and returns the repo root directory.

    GitHub tarballs extract to a single top-level directory
    named something like "owner-repo-abc1234/".
    We return that directory as the repo root.
    """
    with tarfile.open(tarball_path, "r:gz") as tar:
        # GitHub tarballs are safe but we filter anyway —
        # guards against path traversal in malicious archives
        members = [
            m for m in tar.getmembers()
            if not _is_unsafe_path(m.name)
        ]
        tar.extractall(dest, members=members, filter="data")

    # find the single top-level directory GitHub creates
    extracted = [
        p for p in dest.iterdir()
        if p.is_dir() and p.name != "__MACOSX"
    ]

    if len(extracted) != 1:
        raise ValueError(
            f"expected 1 top-level dir after extraction, "
            f"found {len(extracted)}: {extracted}"
        )

    return extracted[0]

async def get_commit_sha(
    full_name: str,
    ref:       str,
    token:     str,
) -> str:
    """
    Fetches the current commit SHA for a branch.
    Call this before download_repo so the index_run
    has the exact commit that was indexed.
    """
    from app.config import get_settings
    settings = get_settings()

    url     = f"{GITHUB_API}/repos/{full_name}/commits/{ref}"
    headers = {
        "Authorization":        f"Bearer {token}",
        "Accept":               "application/vnd.github+json",
        "X-GitHub-API-Version": settings.GITHUB_API_VERSION,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()["sha"]
    
def _is_unsafe_path(name: str) -> bool:
    """
    Guards against path traversal attacks in tarballs.
    Rejects any member whose path starts with / or contains ..
    """
    return name.startswith("/") or ".." in name.split("/")


def _cleanup(path: Path) -> None:
    """
    Recursively deletes a directory.
    Silently ignores errors — cleanup should never crash the app.
    """
    if not path.exists():
        return
    try:
        import shutil
        shutil.rmtree(path)
    except Exception as e:
        # log but don't raise — cleanup failure is not fatal
        logger.warning(f"cleanup failed for {path}: {e}")


def compute_content_hash(content: str) -> str:
    """
    SHA-256 hash of a chunk's content.
    Used for idempotency — if this hash matches an existing
    chunk in the current run, we skip re-embedding it.
    """
    return hashlib.sha256(content.encode()).hexdigest()