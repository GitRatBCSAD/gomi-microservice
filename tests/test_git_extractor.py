"""Unit tests for app/ml/git_extractor.py — get_git_stats()."""
import math
from unittest.mock import patch, MagicMock
import pytest
from app.ml.git_extractor import get_git_stats


def _make_proc(stdout: str) -> MagicMock:
    """Build a fake subprocess.CompletedProcess with the given stdout."""
    proc = MagicMock()
    proc.stdout = stdout
    return proc


def _single_commit_stdout(
    hash_: str = "abc123",
    author: str = "dev@example.com",
    timestamp: int = 1700000100,
    message: str = "fix: resolve null pointer in auth middleware",
    filepath: str = "src/auth.py",
    added: int = 10,
    deleted: int = 2,
) -> str:
    header = f"\x1eCOMMIT\t{hash_}\t{author}\t{timestamp}\t{message}"
    numstat = f"{added}\t{deleted}\t{filepath}"
    return f"{header}\n{numstat}\n"


@patch("subprocess.run")
def test_get_git_stats_normal(mock_run):
    mock_run.return_value = _make_proc(_single_commit_stdout())
    result = get_git_stats("/fake/repo")

    assert "src/auth.py" in result
    entry = result["src/auth.py"]
    assert entry["ndev"] == 1
    assert entry["nf"] == 1
    assert entry["age_days"] == 0  # single commit → no span
    assert len(entry["commits"]) == 1
    assert entry["commits"][0]["message"] == "fix: resolve null pointer in auth middleware"


@patch("subprocess.run")
def test_get_git_stats_deduplication(mock_run):
    """Same commit message on the same file twice → commits list dedups to one entry, nf stays 2."""
    ts1, ts2 = 1700000000, 1700001000
    stdout = (
        _single_commit_stdout(hash_="aaa", timestamp=ts1, message="fix: same message")
        + _single_commit_stdout(hash_="bbb", timestamp=ts2, message="fix: same message")
    )
    mock_run.return_value = _make_proc(stdout)
    result = get_git_stats("/fake/repo")

    entry = result["src/auth.py"]
    assert entry["nf"] == 2           # both changes counted
    assert len(entry["commits"]) == 1  # duplicate message deduplicated


@patch("subprocess.run")
def test_get_git_stats_malformed_numstat(mock_run):
    """Lines with wrong number of tab-separated parts are silently skipped."""
    stdout = "\x1eCOMMIT\tabc\tdev@x.com\t1700000000\tfix something\nBAD_LINE_NO_TABS\n"
    mock_run.return_value = _make_proc(stdout)
    result = get_git_stats("/fake/repo")
    # The malformed line produces no file entry
    assert result == {}


@patch("subprocess.run")
def test_get_git_stats_empty_repo(mock_run):
    """No commits → empty dict returned."""
    mock_run.return_value = _make_proc("")
    result = get_git_stats("/fake/repo")
    assert result == {}


@patch("subprocess.run")
def test_entropy_zero_for_single_commit(mock_run):
    """Single commit to a file → entropy = 0 (only one churn event, no spread)."""
    mock_run.return_value = _make_proc(_single_commit_stdout(added=50, deleted=0))
    result = get_git_stats("/fake/repo")
    # Single commit: prob = 50/50 = 1.0; -1.0 * log2(1.0) = 0.0
    assert result["src/auth.py"]["ent"] == pytest.approx(0.0)


@patch("subprocess.run")
def test_entropy_positive_for_multiple_commits(mock_run):
    """Two commits with equal churn → entropy > 0."""
    stdout = (
        _single_commit_stdout(hash_="aaa", timestamp=1700000000, message="first commit", added=10, deleted=0)
        + _single_commit_stdout(hash_="bbb", timestamp=1700001000, message="second commit", added=10, deleted=0)
    )
    mock_run.return_value = _make_proc(stdout)
    result = get_git_stats("/fake/repo")
    assert result["src/auth.py"]["ent"] > 0.0


@patch("subprocess.run")
def test_ndev_counts_distinct_authors(mock_run):
    """Two different authors on the same file → ndev=2."""
    stdout = (
        _single_commit_stdout(hash_="aaa", author="alice@x.com", timestamp=1700000000, message="alice commit")
        + _single_commit_stdout(hash_="bbb", author="bob@x.com", timestamp=1700001000, message="bob commit")
    )
    mock_run.return_value = _make_proc(stdout)
    result = get_git_stats("/fake/repo")
    assert result["src/auth.py"]["ndev"] == 2


@patch("subprocess.run")
def test_age_days_computed_from_timestamp_span(mock_run):
    """age_days = (latest_ts - earliest_ts) / 86400."""
    ts_old = 1700000000
    ts_new = 1700086400  # exactly 1 day later
    stdout = (
        _single_commit_stdout(hash_="aaa", timestamp=ts_old, message="old commit")
        + _single_commit_stdout(hash_="bbb", timestamp=ts_new, message="new commit")
    )
    mock_run.return_value = _make_proc(stdout)
    result = get_git_stats("/fake/repo")
    assert result["src/auth.py"]["age_days"] == pytest.approx(1.0)
