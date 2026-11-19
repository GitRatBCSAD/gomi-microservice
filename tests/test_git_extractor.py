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
    assert entry["nf"] == 1
    assert entry["commits"][0]["ndev"] == 0  # no prior authors
    assert entry["commits"][0]["age_days"] == 0
    assert len(entry["commits"]) == 1
    assert entry["commits"][0]["message"] == "fix: resolve null pointer in auth middleware"


@patch("subprocess.run")
def test_get_git_stats_no_dedup(mock_run):
    """Same message twice on one file -> both commits kept (training never dedups)."""
    stdout = (
        _single_commit_stdout(hash_="aaa", timestamp=1700000000, message="fix: same message")
        + _single_commit_stdout(hash_="bbb", timestamp=1700001000, message="fix: same message")
    )
    mock_run.return_value = _make_proc(stdout)
    entry = get_git_stats("/fake/repo")["src/auth.py"]
    assert entry["nf"] == 2
    assert len(entry["commits"]) == 2


@patch("subprocess.run")
def test_git_log_uses_no_renames(mock_run):
    mock_run.return_value = _make_proc("")
    get_git_stats("/fake/repo")
    assert "--no-renames" in mock_run.call_args[0][0]


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


def _multi_file_commit(hash_, author, ts, files):
    header = f"\x1eCOMMIT\t{hash_}\t{author}\t{ts}\tmsg {hash_}"
    return header + "\n" + "".join(f"{a}\t{d}\t{fp}\n" for fp, a, d in files)


@patch("subprocess.run")
def test_entropy_is_per_commit_and_normalised(mock_run):
    """Equal churn across 2 files -> log2(2)/log2(2) = 1.0; single-file commit -> 0."""
    stdout = (
        _multi_file_commit("aaa", "a@x.com", 1700000000, [("a.py", 5, 0), ("b.py", 5, 0)])
        + _multi_file_commit("bbb", "a@x.com", 1700001000, [("a.py", 9, 0)])
    )
    mock_run.return_value = _make_proc(stdout)
    result = get_git_stats("/fake/repo")
    by_hash = {c["hash"]: c for c in result["a.py"]["commits"]}
    assert by_hash["aaa"]["ent"] == pytest.approx(1.0)
    assert by_hash["bbb"]["ent"] == pytest.approx(0.0)


@patch("subprocess.run")
def test_ndev_counts_distinct_prior_authors(mock_run):
    """ndev = distinct authors who changed the touched files BEFORE this commit."""
    stdout = (
        _single_commit_stdout(hash_="aaa", author="alice@x.com", timestamp=1700000000, message="one")
        + _single_commit_stdout(hash_="bbb", author="bob@x.com", timestamp=1700001000, message="two")
        + _single_commit_stdout(hash_="ccc", author="carol@x.com", timestamp=1700002000, message="three")
    )
    mock_run.return_value = _make_proc(stdout)
    by_hash = {c["hash"]: c for c in get_git_stats("/fake/repo")["src/auth.py"]["commits"]}
    assert [by_hash[h]["ndev"] for h in ("aaa", "bbb", "ccc")] == [0, 1, 2]


@patch("subprocess.run")
def test_age_is_mean_days_since_prior_change(mock_run):
    """age_days = days since prior change (0 for the first touch)."""
    stdout = (
        _single_commit_stdout(hash_="aaa", timestamp=1700000000, message="old commit")
        + _single_commit_stdout(hash_="bbb", timestamp=1700086400, message="new commit")
    )
    mock_run.return_value = _make_proc(stdout)
    by_hash = {c["hash"]: c for c in get_git_stats("/fake/repo")["src/auth.py"]["commits"]}
    assert by_hash["aaa"]["age_days"] == 0.0
    assert by_hash["bbb"]["age_days"] == pytest.approx(1.0)
