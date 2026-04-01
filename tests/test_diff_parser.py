import logging

import pytest

from app.github.diff_parser import (
    _extract_hunks,
    _parse_file_block,
    _split_into_file_blocks,
    parse_diff,
)

@pytest.fixture
def pr_meta():
    return {
        "full_name": "owner/repo",
        "pr_number": 123,
        "commit_sha": "abc123",
    }

@pytest.fixture
def modified_file_block():
    return (
        "diff --git a/src/test.py b/src/test.py\n"
        "index 123..456 100644\n"
        "--- a/src/test.py\n"
        "+++ b/src/test.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n"
    )

@pytest.fixture
def added_file_block():
    return (
        "diff --git a/src/new_file.py b/src/new_file.py\n"
        "new file mode 100644\n"
        "--- a/src/new_file.py\n"
        "+++ b/src/new_file.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+line1\n"
        "+line2\n"
    )

@pytest.fixture
def removed_file_block():
    return (
        "diff --git a/src/old_file.py b/src/old_file.py\n"
        "deleted file mode 100644\n"
        "--- a/src/old_file.py\n"
        "+++ b/src/old_file.py\n"
        "@@ -1,2 +0,0 @@\n"
        "-line1\n"
        "-line2\n"
    )

@pytest.fixture
def renamed_file_block():
    return (
        "diff --git a/src/old_name.py b/src/new_name.py\n"
        "similarity index 100%\n"
        "rename from src/old_name.py\n"
        "rename to src/new_name.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

@pytest.fixture
def multi_file_diff(modified_file_block, added_file_block):
    return modified_file_block + added_file_block

@pytest.fixture
def single_hunk_lines():
    return [
        "diff --git a/test.py b/test.py",
        "--- a/test.py",
        "+++ b/test.py",
        "@@ -1,3 +1,3 @@",
        " line1",
        "-old_line",
        "+new_line",
        " line3",
    ]

@pytest.fixture
def multi_hunk_lines():
    return [
        "diff --git a/test.py b/test.py",
        "--- a/test.py",
        "+++ b/test.py",
        "@@ -1,2 +1,2 @@",
        "-old1",
        "+new1",
        "@@ -10,2 +10,3 @@",
        " context",
        "-old2",
        "+new2",
        "+new3",
    ]

def test_parse_diff_returns_empty_pull_request_diff_for_blank_input(caplog, pr_meta):
    caplog.set_level(logging.WARNING)

    result = parse_diff(raw_diff="   \n", **pr_meta)

    assert result.full_name == pr_meta["full_name"]
    assert result.pr_number == pr_meta["pr_number"]
    assert result.commit_sha == pr_meta["commit_sha"]
    assert result.files == []
    assert "empty diff received" in caplog.text


def test_split_into_file_blocks_splits_multiple_files(multi_file_diff):
    blocks = _split_into_file_blocks(multi_file_diff)

    assert len(blocks) == 2
    assert blocks[0].startswith("diff --git a/src/test.py b/src/test.py")
    assert blocks[1].startswith("diff --git a/src/new_file.py b/src/new_file.py")


def test_split_into_file_blocks_returns_single_block_when_only_one_file(modified_file_block):
    blocks = _split_into_file_blocks(modified_file_block)

    assert len(blocks) == 1
    assert blocks[0] == modified_file_block


def test_extract_hunks_returns_single_hunk_and_counts_changes(single_hunk_lines):
    hunks, additions, deletions = _extract_hunks(single_hunk_lines)

    assert len(hunks) == 1
    assert hunks[0].header == "@@ -1,3 +1,3 @@"
    assert hunks[0].content == (
        " line1\n"
        "-old_line\n"
        "+new_line\n"
        " line3\n"
    )
    assert additions == 1
    assert deletions == 1


def test_extract_hunks_handles_multiple_hunks(multi_hunk_lines):
    hunks, additions, deletions = _extract_hunks(multi_hunk_lines)

    assert len(hunks) == 2
    assert hunks[0].header == "@@ -1,2 +1,2 @@"
    assert hunks[0].content == "-old1\n+new1\n"

    assert hunks[1].header == "@@ -10,2 +10,3 @@"
    assert hunks[1].content == " context\n-old2\n+new2\n+new3\n"

    assert additions == 3
    assert deletions == 2


def test_parse_file_block_parses_modified_file(modified_file_block):
    diff_file = _parse_file_block(modified_file_block)

    assert diff_file is not None
    assert diff_file.filename == "src/test.py"
    assert diff_file.old_filename == ""
    assert diff_file.status == "modified"
    assert diff_file.additions == 1
    assert diff_file.deletions == 1
    assert len(diff_file.hunks) == 1


def test_parse_file_block_parses_added_file(added_file_block):
    diff_file = _parse_file_block(added_file_block)

    assert diff_file is not None
    assert diff_file.filename == "src/new_file.py"
    assert diff_file.old_filename == ""
    assert diff_file.status == "added"
    assert diff_file.additions == 2
    assert diff_file.deletions == 0


def test_parse_file_block_parses_removed_file(removed_file_block):
    diff_file = _parse_file_block(removed_file_block)

    assert diff_file is not None
    assert diff_file.filename == "src/old_file.py"
    assert diff_file.old_filename == ""
    assert diff_file.status == "removed"
    assert diff_file.additions == 0
    assert diff_file.deletions == 2


def test_parse_file_block_parses_renamed_file(renamed_file_block):
    diff_file = _parse_file_block(renamed_file_block)

    assert diff_file is not None
    assert diff_file.filename == "src/new_name.py"
    assert diff_file.old_filename == "src/old_name.py"
    assert diff_file.status == "renamed"
    assert diff_file.additions == 1
    assert diff_file.deletions == 1


def test_parse_file_block_returns_none_when_filename_cannot_be_extracted(caplog):
    caplog.set_level(logging.WARNING)

    block = (
        "index 123..456 100644\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    diff_file = _parse_file_block(block)

    assert diff_file is None
    assert "could not extract filename from block" in caplog.text


def test_parse_diff_parses_multiple_files(caplog, pr_meta, multi_file_diff):
    caplog.set_level(logging.INFO)

    result = parse_diff(raw_diff=multi_file_diff, **pr_meta)

    assert result.full_name == pr_meta["full_name"]
    assert result.pr_number == pr_meta["pr_number"]
    assert result.commit_sha == pr_meta["commit_sha"]
    assert len(result.files) == 2

    assert result.files[0].filename == "src/test.py"
    assert result.files[0].status == "modified"
    assert result.files[0].additions == 1
    assert result.files[0].deletions == 1

    assert result.files[1].filename == "src/new_file.py"
    assert result.files[1].status == "added"
    assert result.files[1].additions == 2
    assert result.files[1].deletions == 0

    assert "parsed 2 files" in caplog.text