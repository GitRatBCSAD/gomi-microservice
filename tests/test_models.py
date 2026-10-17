"""Unit tests for app/ml/models.py — commit filtering logic (no model loading required)."""
import pytest
from app.ml.models import MLEngine, SentimentResult
from app.ml.constants import MIN_COMMIT_WORDS


def _make_commit(message: str, hash_: str = "abc123", ts: int = 1700000000) -> dict:
    return {
        "hash": hash_,
        "message": message,
        "committed_at": ts,
        "author": "dev@example.com",
        "lines_added": 5,
        "lines_deleted": 2,
        "co_changed_files": 1,
    }


def _mock_pipeline(messages):
    """Fake sentiment pipeline: always returns caution=0.8, neutral=0.15, satisfaction=0.05."""
    return [
        [
            {"label": "caution", "score": 0.8},
            {"label": "neutral", "score": 0.15},
            {"label": "satisfaction", "score": 0.05},
        ]
        for _ in messages
    ]


def _engine_with_mock_pipeline() -> MLEngine:
    engine = MLEngine()
    engine.sentiment_pipeline = _mock_pipeline
    return engine


# ---------------------------------------------------------------------------
# Low-info filtering
# ---------------------------------------------------------------------------

def test_empty_commits_returns_zero_sentiment():
    engine = _engine_with_mock_pipeline()
    result = engine.compute_sentiment([])
    assert result == SentimentResult(0.0, 0.0, [])


def test_low_info_commits_flagged_correctly():
    """Messages with fewer than MIN_COMMIT_WORDS words are marked low_info=True."""
    short_msg = " ".join(["word"] * (MIN_COMMIT_WORDS - 1))  # 4 words
    engine = _engine_with_mock_pipeline()
    result = engine.compute_sentiment([_make_commit(short_msg)])

    assert len(result.commit_sentiments) == 1
    cs = result.commit_sentiments[0]
    assert cs.low_info is True
    assert cs.risk_probability is None
    assert cs.code is None


def test_valid_commits_classified():
    """Messages with >= MIN_COMMIT_WORDS words are classified and flagged low_info=False."""
    long_msg = " ".join(["word"] * MIN_COMMIT_WORDS)  # exactly 5 words
    engine = _engine_with_mock_pipeline()
    result = engine.compute_sentiment([_make_commit(long_msg)])

    assert len(result.commit_sentiments) == 1
    cs = result.commit_sentiments[0]
    assert cs.low_info is False
    assert cs.risk_probability is not None
    assert cs.code is not None


def test_low_info_ratio_computation():
    """3 valid + 1 low-info → low_info_ratio = 0.25."""
    valid_msg = "fix null pointer exception in auth middleware handler"  # 8 words
    short_msg = "fix bug"  # 2 words
    commits = [
        _make_commit(valid_msg, hash_="a"),
        _make_commit(valid_msg, hash_="b"),
        _make_commit(valid_msg, hash_="c"),
        _make_commit(short_msg, hash_="d"),
    ]
    engine = _engine_with_mock_pipeline()
    result = engine.compute_sentiment(commits)
    assert result.low_info_ratio == pytest.approx(0.25)


def test_all_low_info_returns_zero_sentiment_score():
    """All commits low-info → sentiment_score=0.0, ratio=1.0."""
    commits = [_make_commit("wip", hash_="a"), _make_commit("fix", hash_="b")]
    engine = _engine_with_mock_pipeline()
    result = engine.compute_sentiment(commits)
    assert result.sentiment_score == pytest.approx(0.0)
    assert result.low_info_ratio == pytest.approx(1.0)


def test_sentiment_score_is_mean_of_risk_probs():
    """With mock pipeline returning caution=0.8 for all valid commits, sentiment_score≈0.8."""
    valid_msg = "fix null pointer exception in auth middleware handler"
    commits = [
        _make_commit(valid_msg, hash_="a"),
        _make_commit(valid_msg, hash_="b"),
    ]
    engine = _engine_with_mock_pipeline()
    result = engine.compute_sentiment(commits)
    # risk_probability = sum of caution scores = 0.8 for each commit
    assert result.sentiment_score == pytest.approx(0.8)


def test_risk_probability_capped_at_one():
    """risk_probability = sum of caution scores, capped at 1.0."""
    def over_one_pipeline(messages):
        return [
            [{"label": "caution", "score": 0.99}, {"label": "neutral", "score": 0.99}]
            for _ in messages
        ]

    valid_msg = "fix null pointer exception in auth middleware handler"
    engine = MLEngine()
    engine.sentiment_pipeline = over_one_pipeline
    result = engine.compute_sentiment([_make_commit(valid_msg)])
    assert result.commit_sentiments[0].risk_probability <= 1.0


def test_commit_sentiment_metadata_preserved():
    """author, lines_added, lines_deleted, co_changed_files preserved in output."""
    valid_msg = "fix null pointer exception in auth middleware handler"
    commit = _make_commit(valid_msg)
    commit["author"] = "alice@example.com"
    commit["lines_added"] = 42
    commit["lines_deleted"] = 7
    commit["co_changed_files"] = 3

    engine = _engine_with_mock_pipeline()
    result = engine.compute_sentiment([commit])
    cs = result.commit_sentiments[0]
    assert cs.author == "alice@example.com"
    assert cs.lines_added == 42
    assert cs.lines_deleted == 7
    assert cs.co_changed_files == 3
