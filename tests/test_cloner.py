import pytest
from pathlib import Path

from app.config import get_settings
from app.ingestion.cloner import download_repo, compute_content_hash, get_commit_sha


async def test_download_public_repo():
    """
    Downloads a small public repo and verifies
    the extracted directory structure.
    Uses psf/requests — small, stable, public.
    """
    settings = get_settings()

    async with download_repo(
        full_name="psf/requests",
        ref="main",
        token=settings.GITHUB_TOKEN,
    ) as repo_path:
        assert repo_path.exists()
        assert repo_path.is_dir()

        py_files = list(repo_path.rglob("*.py"))
        assert len(py_files) > 0, "no Python files found"

        print(f"\nrepo extracted to: {repo_path}")
        print(f"python files found: {len(py_files)}")

        tmpdir = repo_path.parent
        assert tmpdir.exists()

    assert not tmpdir.exists(), "temp dir not cleaned up"


def test_compute_content_hash():
    h1 = compute_content_hash("def foo(): pass")
    h2 = compute_content_hash("def foo(): pass")
    h3 = compute_content_hash("def bar(): pass")

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64

async def test_get_commit_sha():
    settings = get_settings()

    sha = await get_commit_sha(
        full_name="psf/requests",
        ref="main",
        token=settings.GITHUB_TOKEN,
    )

    assert sha                  # not empty
    assert len(sha) == 40       # full SHA-1 hash
    assert sha.isalnum()        # hex characters only
    print(f"\ncommit sha: {sha}")