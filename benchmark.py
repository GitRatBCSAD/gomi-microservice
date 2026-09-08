"""
Gomi ML Inference Benchmark — Track 2 (ML component timing).
Measures DistilBERT sentiment and LR+SHAP risk model in isolation.

Usage:
    cd gomi-microservice
    uv run python benchmark.py

Requirements:
    - .env file with HF_TOKEN (and optionally HF_SENTIMENT_REPO, HF_RISK_REPO)
    - All microservice dependencies installed
    - First run downloads model weights (~250MB); subsequent runs use HF cache

Output:
    Timing table (mean/median/min/max in ms) for use in thesis benchmark section.
"""
import os
import statistics
import tempfile
import textwrap
import time

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from joblib import Parallel, delayed

from app.core.config import *  # noqa: F401, F403 — ensures app path is importable
from app.ml.constants import BERT_BATCH_SIZE
from app.ml.lizard_scanner import compute_complexity, get_source_files
from app.ml.models import MLEngine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_SENTIMENT_REPO = os.environ.get("HF_SENTIMENT_REPO", "GitRatBCSAD/gomi-sentiment")
HF_RISK_REPO = os.environ.get("HF_RISK_REPO", "GitRatBCSAD/gomi-risk")
RUNS = 5  # repetitions per benchmark; results are averaged


# ---------------------------------------------------------------------------
# Synthetic test data (no real repo needed)
# ---------------------------------------------------------------------------

_COMMIT_MESSAGES = [
    "fix: resolve null pointer exception in authentication middleware",
    "refactor: extract payment processing into dedicated service class",
    "hotfix: critical crash on empty token validation in oauth flow",
    "feat: add repository analysis endpoint with async job queue",
    "chore: update dependencies to latest stable versions",
    "wip",
    "fix bug",
    "merge branch main into feature branch",
    "add tests for user registration flow",
    "cleanup old unused code and remove dead imports",
] * 100  # 1000 messages — representative of a real company repo:
            # ~100 source files × ~10 valid commits/file over 6 months (after low-info filtering)


def _make_commit_dicts(messages: list[str]) -> list[dict]:
    return [
        {
            "hash": f"deadbeef{i:08x}",
            "message": msg,
            "committed_at": 1700000000 + i * 3600,
            "author": f"dev{i % 5}@example.com",
            "lines_added": (i % 20) + 1,
            "lines_deleted": (i % 10),
            "co_changed_files": (i % 5) + 1,
        }
        for i, msg in enumerate(messages)
    ]


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def _fmt(times_ms: list[float]) -> str:
    return (
        f"mean={statistics.mean(times_ms):7.1f}ms  "
        f"median={statistics.median(times_ms):7.1f}ms  "
        f"min={min(times_ms):7.1f}ms  "
        f"max={max(times_ms):7.1f}ms"
    )


def benchmark_sentiment(engine: MLEngine, commits: list[dict], batch_sizes: list[int]) -> None:
    from app.ml.constants import MIN_COMMIT_WORDS

    # Extract valid messages only (matches what compute_sentiment sends to the pipeline)
    valid_msgs = [c["message"] for c in commits if len(c["message"].split()) >= MIN_COMMIT_WORDS]

    print("\n[1] DistilBERT Sentiment — batch inference (pipeline called directly)")
    print(f"    Input: {len(valid_msgs)} valid messages (filtered from {len(commits)} commits), {RUNS} runs each")

    for bs in batch_sizes:
        times: list[float] = []
        for _ in range(RUNS):
            start = time.perf_counter()
            # Call pipeline directly so batch_size is actually varied
            engine.sentiment_pipeline(valid_msgs, batch_size=bs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        print(f"    batch_size={bs:>3}  |  {_fmt(times)}")


def _make_synthetic_py_file(path: str, n_branches: int = 5) -> None:
    """Write a Python file with one function containing n_branches if/elif blocks."""
    branches = "\n".join(
        f"    {'if' if i == 0 else 'elif'} x == {i}:\n        return {i}"
        for i in range(n_branches)
    )
    source = textwrap.dedent(f"""\
        def classify(x):
        {branches}
            else:
                return -1
    """)
    with open(path, "w") as f:
        f.write(source)


def benchmark_lizard(file_counts: list[int]) -> None:
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    print(f"\n[3] Lizard Parallel Complexity Scan — {n_cores} CPU cores (joblib loky)")

    for n in file_counts:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate n synthetic Python files with varied branch counts
            for i in range(n):
                _make_synthetic_py_file(
                    os.path.join(tmpdir, f"module_{i:04d}.py"),
                    n_branches=(i % 8) + 2,  # 2–9 branches, varied CCN
                )
            files = get_source_files(tmpdir)

            times: list[float] = []
            for _ in range(RUNS):
                start = time.perf_counter()
                Parallel(n_jobs=-1, backend="loky")(
                    delayed(compute_complexity)(f) for f in files
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                times.append(elapsed_ms)

        print(f"    n_files={n:>3}       |  {_fmt(times)}")


def benchmark_risk_model(engine: MLEngine, file_counts: list[int]) -> None:
    print("\n[2] Risk Model (LR predict_proba + SHAP) — single call per N files")

    for n in file_counts:
        # Synthetic feature matrix: [sentiment, entropy, ndev, age, complexity, low_info, commits]
        rng = np.random.default_rng(seed=42)
        X_raw = rng.random((n, 7)).astype(float)
        X_scaled = engine.scaler.transform(X_raw)

        times: list[float] = []
        for _ in range(RUNS):
            start = time.perf_counter()
            engine.risk_model.predict_proba(X_scaled)
            engine.shap_explainer.shap_values(X_scaled)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        print(f"    n_files={n:>3}       |  {_fmt(times)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set.")
        print("       Add HF_TOKEN=<your_token> to gomi-microservice/.env and re-run.")
        raise SystemExit(1)

    print("=" * 70)
    print("GOMI ML INFERENCE BENCHMARK")
    print(f"Sentiment repo : {HF_SENTIMENT_REPO}")
    print(f"Risk repo      : {HF_RISK_REPO}")
    print(f"Runs per bench : {RUNS}")
    print("=" * 70)

    print("\nLoading models (first run downloads from HF Hub, subsequent runs use cache)...")
    engine = MLEngine()
    engine.load_models(
        hf_sentiment_repo=HF_SENTIMENT_REPO,
        hf_risk_repo=HF_RISK_REPO,
        hf_token=HF_TOKEN,
    )
    print(f"Models loaded. F1-optimal threshold: {engine.threshold:.4f}")

    commits = _make_commit_dicts(_COMMIT_MESSAGES)

    # Warm-up pass (avoid cold JIT / cache effects on first inference)
    print("\nWarm-up pass...")
    engine.compute_sentiment(commits[:10])

    # Run benchmarks
    benchmark_sentiment(engine, commits, batch_sizes=[8, 16, 32])
    benchmark_lizard(file_counts=[100, 200, 500])
    benchmark_risk_model(engine, file_counts=[50, 200, 500])

    print("\n" + "=" * 70)
    print("Done. Copy results into thesis benchmark table (Chapter 4 / Appendix).")
    print("Footnote: benchmarks run on CPU; GPU would reduce DistilBERT latency.")
    print("=" * 70)


if __name__ == "__main__":
    main()
