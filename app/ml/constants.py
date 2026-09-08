# ML Engine Configuration
DEFAULT_RISK_THRESHOLD = 0.3
BERT_MAX_LENGTH = 128
MIN_COMMIT_WORDS = 5
PERCENTILE_PRECISION = 4
BERT_BATCH_SIZE = 16          # optimal for CPU inference; larger batches increase memory pressure without throughput gain on CPU
MAX_COMMITS_PER_FILE = 100    # cap per-file commits before sentiment; recent history is most relevant

# Git Analysis Configuration
DEFAULT_WINDOW_DAYS = 180
SECONDS_PER_DAY = 86400.0

# Analyzer Configuration
LOW_CONFIDENCE_COMMIT_THRESHOLD = 10
