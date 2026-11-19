import os

# ML Engine Configuration
DEFAULT_RISK_THRESHOLD = 0.63  # fallback only; risk_threshold.json "threshold" (MCC-optimal) overrides it
BERT_MAX_LENGTH = 128
MIN_COMMIT_WORDS = 5
PERCENTILE_PRECISION = 4
BERT_BATCH_SIZE = int(os.getenv("BERT_BATCH_SIZE", "16"))  # default 16 (CPU-optimal); tune via env var for GPU or low-RAM deployments
MAX_COMMITS_PER_FILE = 100    # cap per-file commits before sentiment; recent history is most relevant

# Git Analysis Configuration
DEFAULT_WINDOW_DAYS = 180
SECONDS_PER_DAY = 86400.0

# Analyzer Configuration
LOW_CONFIDENCE_COMMIT_THRESHOLD = 1  # fallback only; risk_threshold.json "min_commits" (notebook gate sweep) overrides it

# Risk-model features in training order; risk_threshold.json "features" overrides it.
RISK_FEATURES = [
    "sentiment_score", "change_entropy", "ndev_score", "age_score",
    "complexity_score", "low_info_ratio", "log_commits",
]
